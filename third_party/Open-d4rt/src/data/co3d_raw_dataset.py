"""CO3D raw dataset adapter with dense depth reprojection supervision."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    sample_frame_indices_with_stride,
)
from .seeding import SeededDatasetMixin


def _load_json_gz(path: Path) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read gzip json: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_depth_png(path: Path, width: int, height: int) -> np.ndarray:
    try:
        dep = np.asarray(Image.open(path), dtype=np.uint16)
        dep_img = Image.fromarray(dep, mode="I;16").resize((width, height), resample=Image.Resampling.NEAREST)
        return np.asarray(dep_img, dtype=np.uint16)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read depth image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_mask_png(path: Path, width: int, height: int) -> np.ndarray:
    try:
        raw = np.asarray(Image.open(path))
        mask_u8 = (raw > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_u8, mode="L").resize((width, height), resample=Image.Resampling.NEAREST)
        return (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.bool_)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read mask image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _viewpoint_to_camera(vp: dict[str, Any], image_h: int, image_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert CO3D viewpoint to OpenCV-convention K, T_cw, T_wc.

    CO3D stores R, T in PyTorch3D right-multiply convention:
        X_cam_pt3d = X_world @ R + T   (= R^T @ X_world + T  in column vectors)
    PyTorch3D camera frame: x-left, y-up, z-forward.
    OpenCV camera frame:    x-right, y-down, z-forward.
    Frame flip F = diag(-1, -1, 1) converts between them.
    """
    r = np.asarray(vp["R"], dtype=np.float32)
    t = np.asarray(vp["T"], dtype=np.float32)
    focal = np.asarray(vp.get("focal_length", [1.0, 1.0]), dtype=np.float32)
    principal = np.asarray(vp.get("principal_point", [0.0, 0.0]), dtype=np.float32)
    intrinsics_format = str(vp.get("intrinsics_format", "ndc_norm_image_bounds")).lower()

    # NDC-to-pixel intrinsics conversion.
    # PyTorch3D NDC projects: x_ndc = fx_ndc * X_pt3d/Z + px_ndc
    # Then NDC-to-screen: u = -x_ndc * s + (W-1)/2  (sign flip compensates frame flip)
    # Combining with frame flip gives standard CV intrinsics:
    #   fx_cv = fx_ndc * s,  cx_cv = (W-1)/2 - px_ndc * s
    if intrinsics_format == "ndc_isotropic":
        s = 0.5 * float(min(image_h, image_w))
        fx = float(focal[0]) * s
        fy = float(focal[1]) * s
        cx = 0.5 * float(image_w - 1) - float(principal[0]) * s
        cy = 0.5 * float(image_h - 1) - float(principal[1]) * s
    elif intrinsics_format == "ndc_norm_image_bounds":
        sx = 0.5 * float(image_w)
        sy = 0.5 * float(image_h)
        fx = float(focal[0]) * sx
        fy = float(focal[1]) * sy
        cx = 0.5 * float(image_w - 1) - float(principal[0]) * sx
        cy = 0.5 * float(image_h - 1) - float(principal[1]) * sy
    else:
        fx = float(focal[0])
        fy = float(focal[1])
        cx = float(principal[0])
        cy = float(principal[1])

    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Extrinsics: PyTorch3D right-multiply → OpenCV column-vector world-to-camera.
    # R^T converts right-multiply R to left-multiply rotation.
    # F = diag(-1,-1,1) flips x-left→x-right, y-up→y-down.
    # T_cw_cv = [[F @ R^T, F @ T], [0, 1]]
    _f = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    r_cw = np.diag(_f) @ r.T
    t_cw_vec = _f * t

    t_cw = np.eye(4, dtype=np.float32)
    t_cw[:3, :3] = r_cw
    t_cw[:3, 3] = t_cw_vec
    t_wc = np.linalg.inv(t_cw).astype(np.float32)
    return k, t_cw, t_wc


@dataclass
class Co3dRawConfig:
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
    max_scenes: int | None = None
    categories: list[str] | None = None
    min_viewpoint_quality: float = 0.5
    use_depth_masks: bool = False
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _FrameMeta:
    frame_number: int
    image_rel: str
    depth_rel: str
    depth_mask_rel: str | None
    k_src: np.ndarray
    t_wc: np.ndarray
    depth_scale_adjustment: float
    src_h: int
    src_w: int


@dataclass
class _Scene:
    scene_id: str
    category: str
    sequence_name: str
    frames: list[_FrameMeta]
    src_h: int
    src_w: int


class Co3dRawDataset(SeededDatasetMixin, Dataset):
    """Loads CO3D v2 from raw files and builds D4RT-compatible supervision."""

    def __init__(self, config: Co3dRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="co3d_raw", default_seed=20260320)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        if not config.training:
            self.augment = RawAugmentConfig()

        if not config.root.exists():
            raise FileNotFoundError(f"CO3D root not found: {config.root}")

        self.scenes: list[_Scene] = []
        categories = self._categories()
        for category in categories:
            category_dir = config.root / category
            if not category_dir.exists():
                continue
            remaining = None
            if config.max_scenes is not None:
                remaining = int(config.max_scenes) - len(self.scenes)
                if remaining <= 0:
                    break
            wanted = self._wanted_frames_for_split(category_dir)
            if not wanted:
                continue
            category_scenes = self._load_category_scenes(
                category=category,
                category_dir=category_dir,
                wanted_frames=wanted,
                sequence_limit=remaining,
            )
            self.scenes.extend(category_scenes)
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                self.scenes = self.scenes[: int(config.max_scenes)]
                break

        if not self.scenes:
            raise ValueError(f"No valid CO3D scenes found for split={config.split} under {config.root}")

    def _categories(self) -> list[str]:
        cfg_categories = self.cfg.categories or []
        if cfg_categories:
            return [str(cat).strip() for cat in cfg_categories if str(cat).strip()]
        return sorted([p.name for p in self.cfg.root.iterdir() if p.is_dir()])

    def _wanted_frames_for_split(self, category_dir: Path) -> dict[str, set[int]]:
        split = str(self.cfg.split).strip().lower()
        split_mapping = {"train": "train", "val": "val", "test": "test"}
        if isinstance(self.cfg.split_map, dict):
            raw_mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in self.cfg.split_map.items()}
            split_mapping.update({k: v for k, v in raw_mapping.items() if v in {"train", "val", "test"}})
        split_key = split_mapping.get(split, split_mapping.get("train", "train"))
        candidates = {
            "train": ["set_lists_fewview_train.json"],
            "val": ["set_lists_fewview_dev.json", "set_lists_fewview_train.json"],
            "test": ["set_lists_fewview_test.json", "set_lists_fewview_dev.json"],
        }.get(split_key, ["set_lists_fewview_train.json"])
        set_lists_dir = category_dir / "set_lists"

        picked_path: Path | None = None
        for name in candidates:
            candidate = set_lists_dir / name
            if candidate.exists():
                picked_path = candidate
                break
        if picked_path is None:
            return {}

        try:
            raw = json.loads(picked_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RetryableSampleError(f"Failed to read CO3D set list: {picked_path}: {exc}", failed_paths=[str(picked_path)]) from exc

        entries = raw.get(split_key, [])
        wanted: dict[str, set[int]] = {}
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            seq = str(entry[0]).strip()
            if not seq:
                continue
            try:
                frame_id = int(entry[1])
            except (TypeError, ValueError):
                continue
            wanted.setdefault(seq, set()).add(frame_id)
        return wanted

    def _filter_sequence_names_by_quality(self, category_dir: Path, sequence_names: list[str]) -> list[str]:
        min_quality = float(self.cfg.min_viewpoint_quality)
        if min_quality <= 0.0:
            return sequence_names

        annotation_path = category_dir / "sequence_annotations.jgz"
        if not annotation_path.exists():
            return sequence_names

        raw_items = _load_json_gz(annotation_path)
        if not isinstance(raw_items, list):
            return sequence_names

        quality_by_seq: dict[str, float] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            seq = str(item.get("sequence_name", "")).strip()
            if not seq:
                continue
            score_raw = item.get("viewpoint_quality_score")
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score):
                quality_by_seq[seq] = score

        filtered: list[str] = []
        for seq in sequence_names:
            score = quality_by_seq.get(seq)
            if score is None or score < min_quality:
                continue
            filtered.append(seq)
        return filtered

    def _load_category_scenes(
        self,
        category: str,
        category_dir: Path,
        wanted_frames: dict[str, set[int]],
        sequence_limit: int | None,
    ) -> list[_Scene]:
        sequence_names = sorted(wanted_frames.keys())
        sequence_names = self._filter_sequence_names_by_quality(category_dir, sequence_names)
        if sequence_limit is not None:
            sequence_names = sequence_names[: max(0, int(sequence_limit))]
        selected = set(sequence_names)
        if not selected:
            return []

        annotation_path = category_dir / "frame_annotations.jgz"
        if not annotation_path.exists():
            return []
        raw_items = _load_json_gz(annotation_path)
        if not isinstance(raw_items, list):
            return []

        frames_by_seq: dict[str, dict[int, _FrameMeta]] = {seq: {} for seq in selected}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            seq = str(item.get("sequence_name", "")).strip()
            if seq not in selected:
                continue
            try:
                frame_id = int(item.get("frame_number", -1))
            except (TypeError, ValueError):
                continue
            if frame_id < 0:
                continue
            if wanted_frames.get(seq) and frame_id not in wanted_frames[seq]:
                continue

            image = item.get("image", {})
            depth = item.get("depth", {})
            viewpoint = item.get("viewpoint")
            if not isinstance(image, dict) or not isinstance(depth, dict) or not isinstance(viewpoint, dict):
                continue

            image_rel = str(image.get("path", "")).strip()
            depth_rel = str(depth.get("path", "")).strip()
            if not image_rel or not depth_rel:
                continue
            image_size = image.get("size", [0, 0])
            image_h = int(image_size[0]) if len(image_size) > 0 else 0
            image_w = int(image_size[1]) if len(image_size) > 1 else 0
            if image_h <= 0 or image_w <= 0:
                continue

            try:
                k_src, _, t_wc = _viewpoint_to_camera(viewpoint, image_h=image_h, image_w=image_w)
            except Exception:
                continue
            if not np.isfinite(k_src).all() or not np.isfinite(t_wc).all():
                continue

            depth_mask_rel_raw = depth.get("mask_path")
            depth_mask_rel = str(depth_mask_rel_raw).strip() if depth_mask_rel_raw is not None else None
            if depth_mask_rel == "":
                depth_mask_rel = None
            depth_scale_adjustment = float(depth.get("scale_adjustment", 1.0))
            frames_by_seq[seq][frame_id] = _FrameMeta(
                frame_number=frame_id,
                image_rel=image_rel,
                depth_rel=depth_rel,
                depth_mask_rel=depth_mask_rel,
                k_src=k_src,
                t_wc=t_wc,
                depth_scale_adjustment=depth_scale_adjustment,
                src_h=image_h,
                src_w=image_w,
            )

        scenes: list[_Scene] = []
        for seq in sequence_names:
            frame_map = frames_by_seq.get(seq, {})
            if not frame_map:
                continue
            frames = [frame_map[fid] for fid in sorted(frame_map.keys())]
            if len(frames) < self.cfg.clip_frames:
                continue
            if not self._has_usable_depth(frames):
                continue
            first = frames[0]
            scenes.append(
                _Scene(
                    scene_id=f"{category}/{seq}",
                    category=category,
                    sequence_name=seq,
                    frames=frames,
                    src_h=int(first.src_h),
                    src_w=int(first.src_w),
                )
            )
        return scenes

    def _has_usable_depth(self, frames: list[_FrameMeta]) -> bool:
        if not frames:
            return False
        sample_idx = sorted(set([0, len(frames) // 2, len(frames) - 1]))
        for idx in sample_idx:
            frame = frames[idx]
            depth_path = self.cfg.root / frame.depth_rel
            if not depth_path.exists():
                continue
            try:
                depth_u16 = np.asarray(Image.open(depth_path), dtype=np.uint16)
            except Exception:
                continue
            valid = depth_u16 > 0
            if self.cfg.use_depth_masks and frame.depth_mask_rel:
                mask_path = self.cfg.root / frame.depth_mask_rel
                if mask_path.exists():
                    try:
                        valid &= np.asarray(Image.open(mask_path)) > 0
                    except Exception:
                        pass
            if float(valid.mean()) > 1e-4:
                return True
        return False

    def __len__(self) -> int:
        base = len(self.scenes) * 50 if self.cfg.training else len(self.scenes)
        return max(base, len(self.scenes))

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            sid = int(self.rng.integers(0, len(self.scenes)))
            return self.scenes[sid]
        return self.scenes[index % len(self.scenes)]

    def _frame_indices(self, scene: _Scene, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng, scene_len=len(scene.frames), clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment, training=bool(self.cfg.training), index=index,
        )

    def _sample_key(self, scene: _Scene, idxs: list[int]) -> str:
        frame_token = ",".join(str(int(scene.frames[i].frame_number)) for i in idxs)
        return f"co3d_raw::{scene.scene_id}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            frame = scene.frames[i]
            out.append(str(self.cfg.root / frame.image_rel))
            out.append(str(self.cfg.root / frame.depth_rel))
            if frame.depth_mask_rel:
                out.append(str(self.cfg.root / frame.depth_mask_rel))
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
            image_path = self.cfg.root / frame.image_rel
            depth_path = self.cfg.root / frame.depth_rel
            rgb = _read_rgb(image_path, width=self.w, height=self.h)
            depth_u16 = _read_depth_png(depth_path, width=self.w, height=self.h)

            # CO3D depth PNG stores float16 values bitcast as uint16.
            # Official decode: reinterpret uint16 bytes as float16, then scale.
            depth_u16_c = np.ascontiguousarray(depth_u16)
            depth_m = depth_u16_c.view(np.float16).astype(np.float32) * float(frame.depth_scale_adjustment)
            depth_m[~np.isfinite(depth_m)] = 0.0
            valid = np.isfinite(depth_m) & (depth_m > 0.0)
            if self.cfg.use_depth_masks and frame.depth_mask_rel:
                mask_path = self.cfg.root / frame.depth_mask_rel
                if mask_path.exists():
                    valid &= _read_mask_png(mask_path, width=self.w, height=self.h)

            k = frame.k_src.copy()
            sx = float(self.w) / max(float(frame.src_w), 1.0)
            sy = float(self.h) / max(float(frame.src_h), 1.0)
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy

            video_list.append(rgb)
            depth_list.append(depth_m.astype(np.float32))
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

        sample = {
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
                "dataset": "co3d_raw",
                "scene_id": scene.scene_id,
                "clip_start": int(clip_start),
                "source_mode": "co3d_depth_reproject",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            idxs = self._frame_indices(scene, query_index)
            clip_start = int(idxs[0]) if idxs else 0
            sample_key = self._sample_key(scene, idxs)
            sample_paths = self._sample_paths(scene, idxs)

            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue

            try:
                sample = self._build_sample(scene=scene, idxs=idxs, clip_start=clip_start)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="co3d_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"Co3dRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
