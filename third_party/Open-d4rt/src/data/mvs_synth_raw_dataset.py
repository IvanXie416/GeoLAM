"""MVS-Synth raw dataset adapter with dense depth reprojection supervision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
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


_WORLD_REFLECTION_FOR_NEGATIVE_DET = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)


def _to_split_bucket(name: str, modulo: int = 20) -> int:
    return stable_split_bucket(name, modulo=modulo)


def _frame_id_from_stem(stem: str) -> int | None:
    token = str(stem).strip()
    if not token.isdigit():
        return None
    return int(token)


def _read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_depth_exr(path: Path, width: int, height: int) -> np.ndarray:
    try:
        depth = iio.imread(path).astype(np.float32, copy=False)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read EXR depth: {path}: {exc}", failed_paths=[str(path)]) from exc

    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RetryableSampleError(f"Invalid EXR depth shape for {path}: {depth.shape}", failed_paths=[str(path)])
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32, copy=False)


def _parse_pose(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read pose json: {path}: {exc}", failed_paths=[str(path)]) from exc

    try:
        fx = float(raw["f_x"])
        fy = float(raw["f_y"])
        cx = float(raw["c_x"])
        cy = float(raw["c_y"])
        t_cw = np.asarray(raw["extrinsic"], dtype=np.float32).reshape(4, 4)
        if not np.isfinite(t_cw).all():
            raise ValueError("extrinsic contains non-finite values")
        if not np.allclose(t_cw[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
            raise ValueError(f"extrinsic last row is not homogeneous: {t_cw[3].tolist()}")
        det_r = float(np.linalg.det(t_cw[:3, :3].astype(np.float64)))
        if not np.isfinite(det_r):
            raise ValueError("extrinsic rotation determinant is non-finite")
        if abs(abs(det_r) - 1.0) > 1e-2:
            raise ValueError(f"extrinsic rotation determinant magnitude is invalid: det={det_r:.6f}")
        if det_r < 0.0:
            # Raw MVS-Synth poses use a globally reflected world basis.  Compose a
            # fixed world reflection on the right so every exported T_wc has a
            # proper right-handed rotation while camera-to-camera transforms stay
            # unchanged: (T_cw_j H) inv(T_cw_i H) == T_cw_j inv(T_cw_i).
            t_cw = (t_cw @ _WORLD_REFLECTION_FOR_NEGATIVE_DET).astype(np.float32)
            det_r = float(np.linalg.det(t_cw[:3, :3].astype(np.float64)))
        if det_r <= 0.0 or abs(det_r - 1.0) > 1e-2:
            raise ValueError(f"canonicalized extrinsic rotation determinant is invalid: det={det_r:.6f}")
        t_wc = np.linalg.inv(t_cw).astype(np.float32)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to parse pose json: {path}: {exc}", failed_paths=[str(path)]) from exc

    k_src = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return k_src.astype(np.float32), t_wc.astype(np.float32), det_r


@dataclass
class MvsSynthRawConfig:
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
    split_map: dict[str, str] | None = None
    sequence_dir: str = "GTAV_540"
    max_scenes: int | None = None
    split_modulo: int = 20
    depth_scale: float = 1.0
    max_depth_m: float = 1e5
    depth_clip_percentile: float = 0.0
    min_depth_valid_ratio: float = 0.0
    min_valid_frames_ratio: float = 0.0
    require_complete_frames: bool = False
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _FrameMeta:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    pose_path: Path
    k_src: np.ndarray
    t_wc: np.ndarray


@dataclass
class _Scene:
    scene_id: str
    clip_dir: Path
    frames: list[_FrameMeta]
    src_h: int
    src_w: int


class _FilteredDepthSampleError(RetryableSampleError):
    """Sample-level quality filter; retry without persisting a bad-sample entry."""


class MvsSynthRawDataset(SeededDatasetMixin, Dataset):
    """Loads MVS-Synth clips from images/depths/poses and emits D4RT-compatible batches."""

    def __init__(self, config: MvsSynthRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="mvs_synth_raw", default_seed=20260321)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        if not config.training:
            self.augment = RawAugmentConfig()

        if not config.root.exists():
            raise FileNotFoundError(f"MVS-Synth root not found: {config.root}")

        clips_root = config.root / str(config.sequence_dir)
        if not clips_root.exists():
            clips_root = config.root
        if not clips_root.exists():
            raise FileNotFoundError(f"MVS-Synth clips root not found: {clips_root}")

        clip_dirs = sorted([p for p in clips_root.iterdir() if p.is_dir()])
        self.scenes: list[_Scene] = []
        for clip_dir in clip_dirs:
            if not self._in_split(clip_dir.name):
                continue
            scene = self._load_scene(clip_dir)
            if scene is None:
                continue
            self.scenes.append(scene)
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break

        if not self.scenes:
            raise ValueError(f"No valid MVS-Synth scenes found for split={config.split} under {clips_root}")

    def _in_split(self, clip_name: str) -> bool:
        split = str(self.cfg.split).strip().lower()
        split_mapping = {"train": "train", "val": "val", "test": "test"}
        if isinstance(self.cfg.split_map, dict):
            raw_mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in self.cfg.split_map.items()}
            split_mapping.update({k: v for k, v in raw_mapping.items() if v in {"train", "val", "test"}})
        split = split_mapping.get(split, split_mapping.get("train", "train"))
        modulo = max(3, int(self.cfg.split_modulo))
        bucket = _to_split_bucket(clip_name, modulo=modulo)
        val_bucket = modulo - 2
        test_bucket = modulo - 1
        if split == "val":
            return bucket == val_bucket
        if split == "test":
            return bucket == test_bucket
        return bucket < val_bucket

    def _load_scene(self, clip_dir: Path) -> _Scene | None:
        image_dir = clip_dir / "images"
        depth_dir = clip_dir / "depths"
        pose_dir = clip_dir / "poses"
        if not (image_dir.exists() and depth_dir.exists() and pose_dir.exists()):
            return None

        rgb_by_id: dict[int, Path] = {}
        depth_by_id: dict[int, Path] = {}
        pose_by_id: dict[int, Path] = {}

        for p in sorted(image_dir.glob("*.png")):
            frame_id = _frame_id_from_stem(p.stem)
            if frame_id is not None:
                rgb_by_id[frame_id] = p
        for p in sorted(depth_dir.glob("*.exr")):
            frame_id = _frame_id_from_stem(p.stem)
            if frame_id is not None:
                depth_by_id[frame_id] = p
        for p in sorted(pose_dir.glob("*.json")):
            frame_id = _frame_id_from_stem(p.stem)
            if frame_id is not None:
                pose_by_id[frame_id] = p

        if self.cfg.require_complete_frames and not (
            set(rgb_by_id.keys()) == set(depth_by_id.keys()) == set(pose_by_id.keys())
        ):
            return None

        common_ids = sorted(set(rgb_by_id.keys()).intersection(depth_by_id.keys()).intersection(pose_by_id.keys()))
        if len(common_ids) < int(self.cfg.clip_frames):
            return None

        try:
            src_w, src_h = Image.open(rgb_by_id[common_ids[0]]).size
        except Exception:
            return None

        frames: list[_FrameMeta] = []
        for frame_id in common_ids:
            pose_path = pose_by_id[frame_id]
            try:
                k_src, t_wc, _det_r = _parse_pose(pose_path)
            except Exception:
                continue
            if not np.isfinite(k_src).all() or not np.isfinite(t_wc).all():
                continue
            frames.append(
                _FrameMeta(
                    frame_id=int(frame_id),
                    rgb_path=rgb_by_id[frame_id],
                    depth_path=depth_by_id[frame_id],
                    pose_path=pose_path,
                    k_src=k_src.astype(np.float32),
                    t_wc=t_wc.astype(np.float32),
                )
            )

        if len(frames) < int(self.cfg.clip_frames):
            return None

        return _Scene(
            scene_id=clip_dir.name,
            clip_dir=clip_dir,
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
        return f"mvs_synth_raw::{scene.scene_id}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            frame = scene.frames[i]
            out.append(str(frame.rgb_path))
            out.append(str(frame.depth_path))
            out.append(str(frame.pose_path))
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
            depth_m = _read_depth_exr(frame.depth_path, width=self.w, height=self.h)
            depth_m = depth_m * float(self.cfg.depth_scale)
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
            depth_list.append(depth_m)
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
                video_t_chw=video, depth_t_hw=depth, depth_valid_t_hw=depth_valid,
                k_t_33=k_arr, camera_valid_t=cam_valid, rng=self.rng, cfg=self.augment,
                native_aspect_ratio=aspect_ratio, out_info=_crop_info,
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
                    f"MVS-Synth clip has too few valid depth frames: {valid_frames}/{len(idxs)} < {min_frames}"
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
                "dataset": "mvs_synth_raw",
                "scene_id": scene.scene_id,
                "clip_start": int(clip_start),
                "source_mode": "mvs_synth_depth_reproject",
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
                    dataset="mvs_synth_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"MvsSynthRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
