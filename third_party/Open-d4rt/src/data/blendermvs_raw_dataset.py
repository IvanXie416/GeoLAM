"""BlenderMVS raw dataset adapter with dense depth reprojection supervision."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bad_sample_registry import (
    BadSampleRegistry,
    RetryableSampleError,
    failed_paths_from_exception,
    is_retryable_data_error,
)
from .depth_query_builder import build_queries_from_depth
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    count_valid_depth_frames,
    sample_frame_indices_with_stride,
    sanitize_depth_map,
)
from .seeding import SeededDatasetMixin, stable_split_bucket


def _to_split_bucket(name: str, modulo: int = 20) -> int:
    return stable_split_bucket(name, modulo=modulo)


def _frame_id_from_stem(stem: str) -> int | None:
    token = str(stem).strip()
    if token.endswith("_masked"):
        token = token[: -len("_masked")]
    if token.endswith("_cam"):
        token = token[: -len("_cam")]
    if not token.isdigit():
        return None
    return int(token)


def _read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _next_non_comment_line(handle) -> str:
    while True:
        raw = handle.readline()
        if raw == b"":
            raise ValueError("Unexpected EOF while reading PFM header")
        line = raw.decode("ascii", errors="ignore").strip()
        if line and not line.startswith("#"):
            return line


def _read_depth_pfm(path: Path, width: int, height: int) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            header = _next_non_comment_line(handle)
            if header not in {"Pf", "PF"}:
                raise ValueError(f"Unsupported PFM header: {header}")
            dim_line = _next_non_comment_line(handle)
            match = re.match(r"^(\d+)\s+(\d+)$", dim_line)
            if match is None:
                raise ValueError(f"Invalid PFM size line: {dim_line}")
            src_w = int(match.group(1))
            src_h = int(match.group(2))
            scale_line = _next_non_comment_line(handle)
            scale = float(scale_line)
            dtype = np.dtype("<f4" if scale < 0.0 else ">f4")
            channels = 3 if header == "PF" else 1
            count = src_w * src_h * channels
            raw = np.fromfile(handle, dtype=dtype, count=count)
            if raw.size != count:
                raise ValueError(f"PFM payload size mismatch: got {raw.size}, expected {count}")
            if channels == 3:
                depth = raw.reshape(src_h, src_w, 3)[..., 0]
            else:
                depth = raw.reshape(src_h, src_w)
            depth = np.flipud(depth).astype(np.float32, copy=False)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read PFM depth: {path}: {exc}", failed_paths=[str(path)]) from exc

    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32, copy=False)


def _parse_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read camera file: {path}: {exc}", failed_paths=[str(path)]) from exc

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lower = [line.lower() for line in lines]

    def _parse_row(row: str, expected: int) -> list[float]:
        parts = row.replace(",", " ").split()
        if len(parts) < expected:
            raise ValueError(f"Expected {expected} values, got {len(parts)} in row: {row}")
        return [float(parts[i]) for i in range(expected)]

    try:
        if "extrinsic" in lower and "intrinsic" in lower:
            eidx = lower.index("extrinsic")
            iidx = lower.index("intrinsic")
            extrinsic = np.asarray([_parse_row(lines[eidx + 1 + i], 4) for i in range(4)], dtype=np.float32)
            intrinsic = np.asarray([_parse_row(lines[iidx + 1 + i], 3) for i in range(3)], dtype=np.float32)
        else:
            values: list[float] = []
            for line in lines:
                for token in line.split():
                    try:
                        values.append(float(token))
                    except ValueError:
                        continue
            if len(values) < 25:
                raise ValueError(f"Insufficient numeric values in camera file: {len(values)}")
            extrinsic = np.asarray(values[:16], dtype=np.float32).reshape(4, 4)
            intrinsic = np.asarray(values[16:25], dtype=np.float32).reshape(3, 3)
        t_wc = np.linalg.inv(extrinsic).astype(np.float32)
        return intrinsic.astype(np.float32), t_wc
    except Exception as exc:
        raise RetryableSampleError(f"Failed to parse camera file: {path}: {exc}", failed_paths=[str(path)]) from exc


@dataclass
class BlendermvsRawConfig:
    root: Path
    split: str
    clip_frames: int
    image_size: tuple[int, int]  # (H, W)
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    training: bool
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    max_scenes: int | None = None
    split_modulo: int = 20
    split_map: dict[str, str] | None = None
    use_masked_images: bool = False
    max_depth_m: float = 1e5
    depth_clip_percentile: float = 0.0
    min_depth_valid_ratio: float = 0.0
    min_valid_frames_ratio: float = 0.0
    require_complete_frames: bool = False
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64
    roots: tuple[Path, ...] | None = None


@dataclass
class _FrameMeta:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    cam_path: Path
    k_src: np.ndarray
    t_wc: np.ndarray


@dataclass
class _Scene:
    scene_id: str
    scene_dir: Path
    frames: list[_FrameMeta]
    src_h: int
    src_w: int


class _FilteredDepthSampleError(RetryableSampleError):
    """Sample-level quality filter; retry without persisting a bad-sample entry."""


class BlendermvsRawDataset(SeededDatasetMixin, Dataset):
    """Loads BlenderMVS scene folders and builds D4RT-compatible supervision."""

    def __init__(self, config: BlendermvsRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="blendermvs_raw", default_seed=20260321)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        if not config.training:
            self.augment = RawAugmentConfig()

        root_entries = self._configured_roots()
        self.scenes: list[_Scene] = []
        multi_root = len(root_entries) > 1
        for root_tag, root_dir in root_entries:
            scene_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
            for scene_dir in scene_dirs:
                split_key = scene_dir.name if not multi_root else f"{root_tag}::{scene_dir.name}"
                if not self._in_split(split_key):
                    continue
                scene_id = scene_dir.name if not multi_root else f"{root_tag}/{scene_dir.name}"
                scene = self._load_scene(scene_dir=scene_dir, scene_id=scene_id)
                if scene is None:
                    continue
                self.scenes.append(scene)
                if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                    break
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break

        if not self.scenes:
            roots_summary = ", ".join(str(root_dir) for _, root_dir in root_entries)
            raise ValueError(f"No valid BlenderMVS scenes found for split={config.split} under [{roots_summary}]")

    def _configured_roots(self) -> list[tuple[str, Path]]:
        raw_roots = list(self.cfg.roots or ())
        if not raw_roots:
            raw_roots = [self.cfg.root]

        entries: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for raw_root in raw_roots:
            root = Path(raw_root)
            try:
                norm = str(root.resolve())
            except Exception:
                norm = str(root)
            if norm in seen:
                continue
            seen.add(norm)
            if not root.exists():
                raise FileNotFoundError(f"BlenderMVS root not found: {root}")
            entries.append((self._root_tag(root), root))
        return entries

    def _root_tag(self, root: Path) -> str:
        try:
            norm = str(root.resolve())
        except Exception:
            norm = str(root)
        suffix_parts = [part for part in root.parts[-2:] if part not in {"", "/", "."}]
        suffix = "_".join(suffix_parts) if suffix_parts else (root.name or "root")
        suffix = re.sub(r"[^0-9A-Za-z._-]+", "-", suffix).strip("-") or "root"
        digest = hashlib.blake2b(norm.encode("utf-8"), digest_size=4).hexdigest()
        return f"{suffix}-{digest}"

    def _in_split(self, scene_name: str) -> bool:
        split = str(self.cfg.split).strip().lower()
        split_mapping = {"train": "train", "val": "val", "test": "test"}
        if isinstance(self.cfg.split_map, dict):
            raw_mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in self.cfg.split_map.items()}
            split_mapping.update({k: v for k, v in raw_mapping.items() if v in {"train", "val", "test"}})
        split = split_mapping.get(split, split_mapping.get("train", "train"))
        modulo = max(3, int(self.cfg.split_modulo))
        bucket = _to_split_bucket(scene_name, modulo=modulo)
        val_bucket = modulo - 2
        test_bucket = modulo - 1
        if split == "val":
            return bucket == val_bucket
        if split == "test":
            return bucket == test_bucket
        return bucket < val_bucket

    def _load_scene(self, scene_dir: Path, scene_id: str) -> _Scene | None:
        image_dir = scene_dir / "blended_images"
        depth_dir = scene_dir / "rendered_depth_maps"
        cam_dir = scene_dir / "cams"
        if not (image_dir.exists() and depth_dir.exists() and cam_dir.exists()):
            return None

        rgb_by_id: dict[int, Path] = {}
        for p in sorted(image_dir.glob("*.jpg")):
            if not self.cfg.use_masked_images and p.stem.endswith("_masked"):
                continue
            if self.cfg.use_masked_images and not p.stem.endswith("_masked"):
                continue
            frame_id = _frame_id_from_stem(p.stem)
            if frame_id is None:
                continue
            rgb_by_id[frame_id] = p

        depth_by_id: dict[int, Path] = {}
        for p in sorted(depth_dir.glob("*.pfm")):
            frame_id = _frame_id_from_stem(p.stem)
            if frame_id is None:
                continue
            depth_by_id[frame_id] = p

        cam_by_id: dict[int, Path] = {}
        for p in sorted(cam_dir.glob("*_cam.txt")):
            frame_id = _frame_id_from_stem(p.stem)
            if frame_id is None:
                continue
            cam_by_id[frame_id] = p

        if self.cfg.require_complete_frames and not (
            set(rgb_by_id.keys()) == set(depth_by_id.keys()) == set(cam_by_id.keys())
        ):
            return None

        common_ids = sorted(set(rgb_by_id.keys()).intersection(depth_by_id.keys()).intersection(cam_by_id.keys()))
        if len(common_ids) < int(self.cfg.clip_frames):
            return None

        try:
            src_w, src_h = Image.open(rgb_by_id[common_ids[0]]).size
        except Exception:
            return None

        frames: list[_FrameMeta] = []
        for frame_id in common_ids:
            cam_path = cam_by_id[frame_id]
            try:
                k_src, t_wc = _parse_cam(cam_path)
            except Exception:
                continue
            if not np.isfinite(k_src).all() or not np.isfinite(t_wc).all():
                continue
            frames.append(
                _FrameMeta(
                    frame_id=int(frame_id),
                    rgb_path=rgb_by_id[frame_id],
                    depth_path=depth_by_id[frame_id],
                    cam_path=cam_path,
                    k_src=k_src.astype(np.float32),
                    t_wc=t_wc.astype(np.float32),
                )
            )

        if len(frames) < int(self.cfg.clip_frames):
            return None

        return _Scene(
            scene_id=scene_id,
            scene_dir=scene_dir,
            frames=frames,
            src_h=int(src_h),
            src_w=int(src_w),
        )

    def __len__(self) -> int:
        base = len(self.scenes) * 30 if self.cfg.training else len(self.scenes)
        return max(base, len(self.scenes))

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            sid = int(self.rng.integers(0, len(self.scenes)))
            return self.scenes[sid]
        return self.scenes[index % len(self.scenes)]

    def _frame_indices(self, scene_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng, scene_len=scene_len, clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment, training=bool(self.cfg.training), index=index,
        )

    def _sample_key(self, scene: _Scene, idxs: list[int]) -> str:
        frame_token = ",".join(str(int(scene.frames[i].frame_id)) for i in idxs)
        return f"blendermvs_raw::{scene.scene_id}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            frame = scene.frames[i]
            out.append(str(frame.rgb_path))
            out.append(str(frame.depth_path))
            out.append(str(frame.cam_path))
        return out

    def _build_sample(self, scene: _Scene, idxs: list[int], clip_start: int) -> dict[str, Any]:
        video_list: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        depth_valid_list: list[np.ndarray] = []
        k_seq: list[np.ndarray] = []
        t_wc_seq: list[np.ndarray] = []
        camera_valid: list[bool] = []

        for i in idxs:
            frame = scene.frames[i]
            rgb = _read_rgb(frame.rgb_path, width=self.w, height=self.h)
            depth_m = _read_depth_pfm(frame.depth_path, width=self.w, height=self.h)
            depth_m, valid = sanitize_depth_map(
                depth_m,
                max_depth_m=float(self.cfg.max_depth_m),
                depth_clip_percentile=float(self.cfg.depth_clip_percentile),
                min_valid_ratio=float(self.cfg.min_depth_valid_ratio),
            )

            k = frame.k_src.copy()
            sx = float(self.w) / max(float(scene.src_w), 1.0)
            sy = float(self.h) / max(float(scene.src_h), 1.0)
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy

            video_list.append(rgb)
            depth_list.append(depth_m.astype(np.float32, copy=False))
            depth_valid_list.append(valid.astype(np.bool_))
            k_seq.append(k.astype(np.float32))
            t_wc_seq.append(frame.t_wc.astype(np.float32))
            camera_valid.append(bool(np.isfinite(k).all() and np.isfinite(frame.t_wc).all()))

        video = np.stack(video_list, axis=0).astype(np.float32) / 255.0
        video = np.transpose(video, (0, 3, 1, 2))
        depth = np.stack(depth_list, axis=0).astype(np.float32)
        depth_valid = np.stack(depth_valid_list, axis=0).astype(np.bool_)
        k_arr = np.stack(k_seq, axis=0).astype(np.float32)
        t_wc_arr = np.stack(t_wc_seq, axis=0).astype(np.float32)
        cam_valid = np.asarray(camera_valid, dtype=np.bool_)

        aspect_ratio = np.array([scene.src_w / max(float(scene.src_h), 1.0)], dtype=np.float32)
        _crop_info = {}
        if self.cfg.training:
            video = apply_photometric_augment(video_t_chw=video, rng=self.rng, cfg=self.augment)
            (video, depth, depth_valid, k_arr, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video,
                depth_t_hw=depth,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_arr,
                camera_valid_t=cam_valid,
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )

        min_valid_frames_ratio = max(0.0, min(1.0, float(self.cfg.min_valid_frames_ratio)))
        if min_valid_frames_ratio > 0.0:
            min_frames = max(1, int(np.ceil(float(len(idxs)) * min_valid_frames_ratio)))
            valid_frames = count_valid_depth_frames(
                depth_valid,
                min_valid_ratio=float(self.cfg.min_depth_valid_ratio),
            )
            if valid_frames < min_frames:
                raise _FilteredDepthSampleError(
                    f"BlenderMVS clip has too few valid depth frames: {valid_frames}/{len(idxs)} < {min_frames}"
                )

        query, target, mask, query_stats = build_queries_from_depth(
            rng=self.rng,
            depth=depth,
            depth_valid=depth_valid,
            k_seq=k_arr,
            t_wc_seq=t_wc_arr,
            camera_valid=cam_valid,
            queries_per_clip=int(self.cfg.queries_per_clip),
            hard_query_ratio=float(self.cfg.hard_query_ratio),
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )

        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_arr).float(),
                "T_wc": torch.from_numpy(t_wc_arr).float(),
                "camera_valid": torch.from_numpy(cam_valid).bool(),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()},
            "meta": {
                "dataset": "blendermvs_raw",
                "scene_id": scene.scene_id,
                "clip_start": int(clip_start),
                "source_mode": "blendermvs_depth_reproject",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            idxs = self._frame_indices(len(scene.frames), query_index)
            clip_start = int(idxs[0]) if idxs else 0
            sample_key = self._sample_key(scene, idxs)
            sample_paths = self._sample_paths(scene, idxs)

            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue

            try:
                sample = self._build_sample(scene=scene, idxs=idxs, clip_start=clip_start)
            except _FilteredDepthSampleError as exc:
                last_error = exc
                continue
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="blendermvs_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"BlendermvsRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
