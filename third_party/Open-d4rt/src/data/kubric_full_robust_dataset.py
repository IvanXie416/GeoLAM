"""Kubric MOVi-F full-annotation robust dataset adapter (TFDS-based)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
import re
from typing import Any
import warnings
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .bad_sample_registry import (
    BadSampleRegistry,
    RetryableSampleError,
    failed_paths_from_exception,
    is_retryable_data_error,
)
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    depth_boundary_mask,
    sample_frame_indices_with_stride,
    sample_hard_query_flags,
    sample_t_tgt_t_cam,
)
from .seeding import SeededDatasetMixin


def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = (q / n).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _seg_boundary_mask(seg_hw: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg_hw)
    if seg.ndim != 2:
        return np.zeros_like(seg, dtype=bool)
    out = np.zeros_like(seg, dtype=bool)
    out[:-1, :] |= seg[:-1, :] != seg[1:, :]
    out[1:, :] |= seg[1:, :] != seg[:-1, :]
    out[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    out[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    out_u8 = out.astype(np.uint8)
    out = cv2.dilate(out_u8, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return out


def _resize_nn(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_crop_resize_extra(
    arr_t_hw_or_hwc: np.ndarray,
    crop_xy: tuple[int, int],
    crop_hw: tuple[int, int],
    out_hw: tuple[int, int],
    interp: int,
) -> np.ndarray:
    x0, y0 = int(crop_xy[0]), int(crop_xy[1])
    ch, cw = int(crop_hw[0]), int(crop_hw[1])
    h, w = int(out_hw[0]), int(out_hw[1])
    t = int(arr_t_hw_or_hwc.shape[0])
    out = []
    for i in range(t):
        frame = arr_t_hw_or_hwc[i]
        cropped = frame[y0 : y0 + ch, x0 : x0 + cw]
        resized = cv2.resize(cropped, (w, h), interpolation=interp)
        out.append(resized)
    return np.stack(out, axis=0)


def _decode_depth_u16_to_metric(depth_u16: np.ndarray, depth_range: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_u16, dtype=np.float32)
    lo = float(depth_range[0])
    hi = float(depth_range[1])
    return lo + d / 65535.0 * (hi - lo)


def _decode_object_coordinates_u16(obj_u16: np.ndarray) -> np.ndarray:
    return np.asarray(obj_u16, dtype=np.float32) / 65535.0 - 0.5


def _decode_world_normals_u16(normal_u16: np.ndarray) -> np.ndarray:
    return np.asarray(normal_u16, dtype=np.float32) / 65535.0 * 2.0 - 1.0


def _resolve_tfds_dir(root: Path) -> Path:
    if (root / "dataset_info.json").exists():
        return root
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and (p / "dataset_info.json").exists()])
    if not candidates:
        raise FileNotFoundError(f"No TFDS version dir with dataset_info.json under: {root}")
    return candidates[-1]


_TF_NOT_FOUND_PATTERNS = (
    re.compile(r"NOT_FOUND:\s*([^;\n]+)"),
    re.compile(r"No such file or directory[: ]+([^;\n]+)", re.IGNORECASE),
    re.compile(r"([^\s;]+\.tfrecord-\d+-of-\d+)"),
)


def _dedup_str_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        s = str(raw).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_failed_paths_from_error_text(message: str) -> list[str]:
    if not message:
        return []
    found: list[str] = []
    for pattern in _TF_NOT_FOUND_PATTERNS:
        for match in pattern.finditer(message):
            token = str(match.group(1)).strip().strip("'").strip('"')
            if token:
                found.append(token)
    return _dedup_str_list(found)


@dataclass
class KubricFullRobustConfig:
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
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64
    tfds_split_map: dict[str, str] | None = None
    shuffle_buffer_size: int = 256
    eval_cache_max_items: int = 4
    benchmark_tracking_enabled: bool = False
    benchmark_max_queries: int = 4096


class KubricFullRobustDataset(SeededDatasetMixin, Dataset):
    """Loads Kubric MOVi-F full annotations from TFDS and builds robust 3D supervision."""

    REQUIRED_TOP_KEYS = {
        "video",
        "segmentations",
        "depth",
        "normal",
        "object_coordinates",
        "camera",
        "instances",
        "metadata",
    }

    REQUIRED_CAMERA_KEYS = {"positions", "quaternions", "field_of_view"}
    REQUIRED_INSTANCE_KEYS = {"bboxes_3d", "positions", "quaternions"}
    REQUIRED_METADATA_KEYS = {"depth_range", "num_frames", "num_instances", "video_name", "height", "width"}

    def __init__(self, config: KubricFullRobustConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="kubric_full_robust", default_seed=20260327)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        self._warned_skip_keys: set[str] = set()
        if not config.training:
            self.augment = RawAugmentConfig()

        tfds_dir = _resolve_tfds_dir(config.root)
        self.tfds_dir = tfds_dir

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except Exception as exc:
            raise ImportError(
                "KubricFullRobustDataset requires tensorflow and tensorflow_datasets. "
                "Please install them in the running environment."
            ) from exc

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        self.tfds = tfds
        self.tf = tf
        self.builder = tfds.builder_from_directory(str(tfds_dir))
        split_map = dict(config.tfds_split_map or {"train": "train", "val": "validation", "test": "validation"})
        tfds_split = split_map.get(str(config.split).lower(), str(config.split))
        if tfds_split not in self.builder.info.splits:
            raise ValueError(f"TFDS split '{tfds_split}' not found. available={list(self.builder.info.splits.keys())}")
        self.tfds_split = tfds_split
        self.num_examples = int(self.builder.info.splits[tfds_split].num_examples)
        if self.num_examples <= 0:
            raise ValueError(f"Empty TFDS split: {tfds_split}")
        if config.max_scenes is not None:
            self.num_examples = min(self.num_examples, int(config.max_scenes))

        read_cfg = tfds.ReadConfig(try_autocache=False)
        ds = self.builder.as_dataset(split=tfds_split, shuffle_files=bool(config.training), read_config=read_cfg)
        ds = self._with_ignore_errors(ds)
        if config.max_scenes is not None:
            ds = ds.take(int(config.max_scenes))
        if config.training:
            self._train_iter = None
            self._eval_ds = None
            self._eval_cache = None
            self._reset_train_iter()
        else:
            self._train_iter = None
            self._eval_ds = ds
            self._eval_cache_max_items = max(0, int(config.eval_cache_max_items))
            self._eval_cache: dict[int, dict[str, Any]] = {}
            self._eval_cache_order: deque[int] = deque()
            self._eval_next_index = 0
            self._reset_eval_iter()

        try:
            probe_ds = self._with_ignore_errors(self.builder.as_dataset(split=f"{tfds_split}[:1]"))
            sample_for_schema = next(iter(tfds.as_numpy(probe_ds)))
            self._validate_schema(sample_for_schema)
        except Exception as exc:
            failed_paths = self._failed_paths_from_exception(exc, default=[str(self.tfds_dir)])
            self.bad_registry.mark_bad(
                dataset="kubric_full_robust",
                sample_key="kubric_full_robust::__schema_probe__",
                sample_paths=[str(self.tfds_dir)],
                failed_paths=failed_paths,
                error=f"{type(exc).__name__}: {exc}",
            )
            warnings.warn(
                "KubricFullRobustDataset schema probe failed; continue with runtime per-sample validation. "
                f"reason={type(exc).__name__}: {exc}",
                stacklevel=2,
            )

    def _reset_train_iter(self) -> None:
        if not self.cfg.training:
            return
        shuffle_seed = int(self._seed_material(index=0, attempt=0, stream=97).generate_state(1, dtype=np.uint32)[0])
        read_cfg = self.tfds.ReadConfig(try_autocache=False)
        ds = self.builder.as_dataset(split=self.tfds_split, shuffle_files=True, read_config=read_cfg)
        ds = self._with_ignore_errors(ds)
        if self.cfg.max_scenes is not None:
            ds = ds.take(int(self.cfg.max_scenes))
        ds = ds.shuffle(max(16, int(self.cfg.shuffle_buffer_size)), seed=shuffle_seed, reshuffle_each_iteration=True).repeat()
        self._train_iter = iter(self.tfds.as_numpy(ds))

    def _reset_eval_iter(self) -> None:
        if self.cfg.training:
            return
        assert self._eval_ds is not None
        self._eval_iter = iter(self.tfds.as_numpy(self._eval_ds))
        self._eval_next_index = 0
        self._eval_cache.clear()
        self._eval_cache_order.clear()

    def configure_dataset_seed(self, base_seed: int) -> None:
        super().configure_dataset_seed(base_seed)
        self._reset_train_iter()

    def set_dataset_epoch(self, epoch: int) -> None:
        super().set_dataset_epoch(epoch)
        self._reset_train_iter()

    def set_dataset_worker(self, worker_id: int) -> None:
        super().set_dataset_worker(worker_id)
        self._reset_train_iter()

    def _with_ignore_errors(self, ds: Any) -> Any:
        try:
            return ds.ignore_errors()
        except Exception:
            try:
                return ds.apply(self.tf.data.experimental.ignore_errors())
            except Exception:
                return ds

    def _failed_paths_from_exception(self, exc: Exception, default: list[str] | None = None) -> list[str]:
        paths = failed_paths_from_exception(exc)
        if paths:
            return _dedup_str_list(paths)
        paths = _extract_failed_paths_from_error_text(str(exc))
        if paths:
            return _dedup_str_list(paths)
        return _dedup_str_list(list(default or []))

    def _warn_skip(self, sample_key: str, reason: str, failed_paths: list[str]) -> None:
        key = str(sample_key).strip() or "__unknown_sample__"
        if key in self._warned_skip_keys:
            return
        self._warned_skip_keys.add(key)
        failed = ", ".join(str(p) for p in failed_paths) if failed_paths else "N/A"
        warnings.warn(
            f"[kubric_full_robust] skip bad sample: sample_key={key}; reason={reason}; failed_paths={failed}",
            stacklevel=2,
        )

    def _validate_schema(self, sample: dict[str, Any]) -> None:
        top = set(sample.keys())
        missing = sorted(self.REQUIRED_TOP_KEYS - top)
        if missing:
            raise ValueError(f"Kubric full sample missing keys: {missing}")
        camera_keys = set(sample["camera"].keys())
        missing_cam = sorted(self.REQUIRED_CAMERA_KEYS - camera_keys)
        if missing_cam:
            raise ValueError(f"Kubric full sample missing camera keys: {missing_cam}")
        inst_keys = set(sample["instances"].keys())
        missing_inst = sorted(self.REQUIRED_INSTANCE_KEYS - inst_keys)
        if missing_inst:
            raise ValueError(f"Kubric full sample missing instances keys: {missing_inst}")
        meta_keys = set(sample["metadata"].keys())
        missing_meta = sorted(self.REQUIRED_METADATA_KEYS - meta_keys)
        if missing_meta:
            raise ValueError(f"Kubric full sample missing metadata keys: {missing_meta}")

    def _validate_runtime_sample(self, sample: dict[str, Any]) -> None:
        try:
            self._validate_schema(sample)
        except Exception as exc:
            raise RetryableSampleError(
                f"Incomplete Kubric full sample schema: {exc}",
                failed_paths=[str(self.tfds_dir)],
            ) from exc

        try:
            video = np.asarray(sample["video"])
            depth = np.asarray(sample["depth"])
            seg = np.asarray(sample["segmentations"])
            normal = np.asarray(sample["normal"])
            obj = np.asarray(sample["object_coordinates"])
            cam_pos = np.asarray(sample["camera"]["positions"])
            cam_quat = np.asarray(sample["camera"]["quaternions"])
        except Exception as exc:
            raise RetryableSampleError(
                f"Failed to access Kubric sample arrays: {type(exc).__name__}: {exc}",
                failed_paths=[str(self.tfds_dir)],
            ) from exc

        if video.ndim != 4 or video.shape[-1] != 3:
            raise RetryableSampleError(
                f"Invalid video shape: {video.shape}",
                failed_paths=[str(self.tfds_dir)],
            )
        if depth.ndim != 4 or depth.shape[-1] != 1:
            raise RetryableSampleError(
                f"Invalid depth shape: {depth.shape}",
                failed_paths=[str(self.tfds_dir)],
            )
        if seg.ndim != 4 or seg.shape[-1] != 1:
            raise RetryableSampleError(
                f"Invalid segmentations shape: {seg.shape}",
                failed_paths=[str(self.tfds_dir)],
            )
        if normal.ndim != 4 or normal.shape[-1] != 3:
            raise RetryableSampleError(
                f"Invalid normal shape: {normal.shape}",
                failed_paths=[str(self.tfds_dir)],
            )
        if obj.ndim != 4 or obj.shape[-1] != 3:
            raise RetryableSampleError(
                f"Invalid object_coordinates shape: {obj.shape}",
                failed_paths=[str(self.tfds_dir)],
            )
        if cam_pos.ndim != 2 or cam_pos.shape[-1] != 3:
            raise RetryableSampleError(
                f"Invalid camera.positions shape: {cam_pos.shape}",
                failed_paths=[str(self.tfds_dir)],
            )
        if cam_quat.ndim != 2 or cam_quat.shape[-1] != 4:
            raise RetryableSampleError(
                f"Invalid camera.quaternions shape: {cam_quat.shape}",
                failed_paths=[str(self.tfds_dir)],
            )

    def __len__(self) -> int:
        base = self.num_examples * 40 if self.cfg.training else self.num_examples
        return max(base, self.num_examples)

    def _next_train_sample(self) -> dict[str, Any]:
        assert self._train_iter is not None
        try:
            return next(self._train_iter)
        except StopIteration:
            ds = self.builder.as_dataset(split=self.tfds_split, shuffle_files=True)
            ds = self._with_ignore_errors(ds).shuffle(256).repeat()
            self._train_iter = iter(self.tfds.as_numpy(ds))
            return next(self._train_iter)
        except Exception as exc:
            failed_paths = self._failed_paths_from_exception(exc, default=[str(self.tfds_dir)])
            raise RetryableSampleError(
                f"TFDS train iterator failure: {type(exc).__name__}: {exc}",
                failed_paths=failed_paths,
            ) from exc

    def _eval_sample_by_index(self, index: int) -> dict[str, Any]:
        assert self._eval_ds is not None
        if index in self._eval_cache:
            return self._eval_cache[index]
        if index < self._eval_next_index:
            self._reset_eval_iter()
        while self._eval_next_index <= index:
            try:
                sample = next(self._eval_iter)
            except StopIteration as exc:
                raise IndexError(f"Eval index out of range: {index}") from exc
            except Exception as exc:
                failed_paths = self._failed_paths_from_exception(exc, default=[str(self.tfds_dir)])
                raise RetryableSampleError(
                    f"TFDS eval iterator failure: {type(exc).__name__}: {exc}",
                    failed_paths=failed_paths,
                ) from exc
            current_index = self._eval_next_index
            if self._eval_cache_max_items > 0:
                self._eval_cache[current_index] = sample
                self._eval_cache_order.append(current_index)
                while len(self._eval_cache_order) > self._eval_cache_max_items:
                    evict_index = self._eval_cache_order.popleft()
                    self._eval_cache.pop(evict_index, None)
            self._eval_next_index += 1
            if current_index == index:
                return sample
        raise IndexError(f"Eval index out of range: {index}")

    def _select_clip(self, sample: dict[str, Any], index: int) -> list[int]:
        scene_len = int(sample["video"].shape[0])
        return sample_frame_indices_with_stride(
            rng=self.rng,
            scene_len=scene_len,
            clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment,
            training=bool(self.cfg.training),
            index=index,
        )

    def _camera_mats_and_depth(
        self,
        sample: dict[str, Any],
        idxs: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        video = np.asarray(sample["video"])[idxs]  # [T,H0,W0,3]
        depth_u16 = np.asarray(sample["depth"])[idxs, ..., 0]  # [T,H0,W0]
        seg = np.asarray(sample["segmentations"])[idxs, ..., 0]  # [T,H0,W0]
        obj_coord_u16 = np.asarray(sample["object_coordinates"])[idxs]  # [T,H0,W0,3]
        normal_u16 = np.asarray(sample["normal"])[idxs]  # [T,H0,W0,3]

        h0, w0 = int(video.shape[1]), int(video.shape[2])
        depth_range = np.asarray(sample["metadata"]["depth_range"], dtype=np.float32)
        depth_range_m = _decode_depth_u16_to_metric(depth_u16, depth_range)  # radial distance

        cam_pos = np.asarray(sample["camera"]["positions"], dtype=np.float32)[idxs]
        cam_quat = np.asarray(sample["camera"]["quaternions"], dtype=np.float32)[idxs]
        fov = float(sample["camera"]["field_of_view"])
        fx = 0.5 * float(w0) / max(np.tan(0.5 * fov), 1e-6)
        fy = 0.5 * float(h0) / max(np.tan(0.5 * fov), 1e-6)
        cx = (float(w0) - 1.0) * 0.5
        cy = (float(h0) - 1.0) * 0.5

        # Kubric uses Blender camera convention (front=-Z, y-up). Convert to CV (+Z forward, y-down).
        s_blender_to_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        t = len(idxs)
        k_seq = np.tile(np.eye(3, dtype=np.float32)[None], (t, 1, 1))
        k_seq[:, 0, 0] = fx
        k_seq[:, 1, 1] = fy
        k_seq[:, 0, 2] = cx
        k_seq[:, 1, 2] = cy
        t_wc_seq = np.tile(np.eye(4, dtype=np.float32)[None], (t, 1, 1))
        for i in range(t):
            r_bl = _quat_wxyz_to_rot(cam_quat[i])
            r_cv = r_bl @ s_blender_to_cv
            t_wc_seq[i, :3, :3] = r_cv
            t_wc_seq[i, :3, 3] = cam_pos[i]

        # Convert radial depth -> z depth in CV camera frame.
        uu = np.arange(w0, dtype=np.float32)[None, :]
        vv = np.arange(h0, dtype=np.float32)[:, None]
        x = (uu - cx) / max(fx, 1e-6)
        y = (vv - cy) / max(fy, 1e-6)
        ray_norm = np.sqrt(x * x + y * y + 1.0).astype(np.float32)
        z_factor = 1.0 / np.maximum(ray_norm, 1e-6)
        depth_z = depth_range_m * z_factor[None, :, :]

        return (
            video,
            depth_z.astype(np.float32),
            depth_range_m.astype(np.float32),
            seg.astype(np.int32),
            _decode_object_coordinates_u16(obj_coord_u16).astype(np.float32),
            _decode_world_normals_u16(normal_u16).astype(np.float32),
            k_seq.astype(np.float32),
            t_wc_seq.astype(np.float32),
        )

    def _resize_to_target(
        self,
        video: np.ndarray,
        depth_z: np.ndarray,
        depth_range_m: np.ndarray,
        seg: np.ndarray,
        obj_coord: np.ndarray,
        normal_world: np.ndarray,
        k_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t, h0, w0, _ = video.shape
        if h0 == self.h and w0 == self.w:
            return (
                video.astype(np.float32) / 255.0,
                depth_z.astype(np.float32),
                depth_range_m.astype(np.float32),
                seg.astype(np.int32),
                obj_coord.astype(np.float32),
                normal_world.astype(np.float32),
                k_seq.astype(np.float32),
            )
        out_video = np.empty((t, self.h, self.w, 3), dtype=np.uint8)
        out_depth_z = np.empty((t, self.h, self.w), dtype=np.float32)
        out_depth_range = np.empty((t, self.h, self.w), dtype=np.float32)
        out_seg = np.empty((t, self.h, self.w), dtype=np.int32)
        out_obj = np.empty((t, self.h, self.w, 3), dtype=np.float32)
        out_normal = np.empty((t, self.h, self.w, 3), dtype=np.float32)
        for i in range(t):
            out_video[i] = cv2.resize(video[i], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            out_depth_z[i] = _resize_nn(depth_z[i], (self.h, self.w))
            out_depth_range[i] = _resize_nn(depth_range_m[i], (self.h, self.w))
            out_seg[i] = _resize_nn(seg[i], (self.h, self.w)).astype(np.int32)
            out_obj[i] = _resize_nn(obj_coord[i], (self.h, self.w)).astype(np.float32)
            out_normal[i] = _resize_nn(normal_world[i], (self.h, self.w)).astype(np.float32)

        sx = float(self.w) / max(float(w0), 1.0)
        sy = float(self.h) / max(float(h0), 1.0)
        k = k_seq.copy()
        k[:, 0, 0] *= sx
        k[:, 0, 2] *= sx
        k[:, 1, 1] *= sy
        k[:, 1, 2] *= sy
        return (
            out_video.astype(np.float32) / 255.0,
            out_depth_z,
            out_depth_range,
            out_seg,
            out_obj,
            out_normal,
            k.astype(np.float32),
        )

    @staticmethod
    def _project_point(k: np.ndarray, t_cw: np.ndarray, p_world: np.ndarray) -> tuple[float, float, float]:
        p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32)
        p_cam = (t_cw @ p_h)[:3]
        z = float(p_cam[2])
        if not np.isfinite(z):
            return np.nan, np.nan, np.nan
        if z <= 1e-6:
            return np.nan, np.nan, z
        proj = k @ p_cam
        u = float(proj[0] / z)
        v = float(proj[1] / z)
        return u, v, z

    @staticmethod
    def _depth_seg_occlusion(
        depth_z_hw: np.ndarray,
        depth_valid_hw: np.ndarray,
        seg_hw: np.ndarray,
        u: float,
        v: float,
        z_proj: float,
        seg_id: int,
    ) -> bool:
        h, w = depth_z_hw.shape
        if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(z_proj) or z_proj <= 1e-6:
            return True
        x = float(u)
        y = float(v)
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        coords = ((y0, x0), (y1, x0), (y0, x1), (y1, x1))

        ds = []
        seg_match = []
        for yy, xx in coords:
            if bool(depth_valid_hw[yy, xx]):
                ds.append(float(depth_z_hw[yy, xx]))
            seg_match.append(int(seg_hw[yy, xx]) == int(seg_id))
        if not ds:
            return True
        depth_nn = max(ds)
        depth_occluded = depth_nn < (z_proj * 0.99)
        if int(seg_id) > 0:
            seg_occluded = not any(seg_match)
        else:
            seg_occluded = False
        return bool(depth_occluded or seg_occluded)

    def _build_object_local_to_world(
        self,
        bboxes_3d_ot83: np.ndarray,
    ) -> np.ndarray:
        num_obj, t_clip = int(bboxes_3d_ot83.shape[0]), int(bboxes_3d_ot83.shape[1])
        local_box = np.array(
            [
                [-0.5, -0.5, -0.5, 1.0],
                [-0.5, -0.5, 0.5, 1.0],
                [-0.5, 0.5, -0.5, 1.0],
                [-0.5, 0.5, 0.5, 1.0],
                [0.5, -0.5, -0.5, 1.0],
                [0.5, -0.5, 0.5, 1.0],
                [0.5, 0.5, -0.5, 1.0],
                [0.5, 0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        out = np.full((num_obj, t_clip, 4, 4), np.nan, dtype=np.float32)
        for oi in range(num_obj):
            for ti in range(t_clip):
                bbox = np.asarray(bboxes_3d_ot83[oi, ti], dtype=np.float32)  # [8,3]
                if bbox.shape != (8, 3) or not np.isfinite(bbox).all():
                    continue
                bbox_h = np.concatenate([bbox, np.ones((8, 1), dtype=np.float32)], axis=-1)  # [8,4]
                try:
                    m, *_ = np.linalg.lstsq(local_box, bbox_h, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                out[oi, ti] = m.astype(np.float32)
        return out

    @staticmethod
    def _sample_rows(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        if count <= 0 or pool.shape[0] <= 0:
            return np.zeros((0, 2), dtype=np.int64)
        if pool.shape[0] <= count:
            return pool.astype(np.int64, copy=False)
        picks = rng.choice(pool.shape[0], size=int(count), replace=False)
        return pool[np.asarray(picks, dtype=np.int64)].astype(np.int64, copy=False)

    def _build_benchmark_tracking(
        self,
        *,
        depth_z: np.ndarray,
        depth_range_m: np.ndarray,
        depth_valid: np.ndarray,
        seg: np.ndarray,
        obj_coord_local: np.ndarray,
        k_seq: np.ndarray,
        t_wc_seq: np.ndarray,
        bboxes_3d_ot83: np.ndarray,
        sample_key: str,
    ) -> dict[str, np.ndarray] | None:
        t_clip, h, w = depth_z.shape
        max_queries = int(self.cfg.benchmark_max_queries)
        if t_clip <= 0 or h <= 0 or w <= 0 or max_queries == 0:
            return None

        cam_valid = np.isfinite(t_wc_seq).all(axis=(1, 2)) & np.isfinite(k_seq).all(axis=(1, 2))
        t_cw_seq = np.full((t_clip, 4, 4), np.nan, dtype=np.float32)
        for ti in range(t_clip):
            if not bool(cam_valid[ti]):
                continue
            try:
                t_cw_seq[ti] = np.linalg.inv(t_wc_seq[ti]).astype(np.float32)
            except np.linalg.LinAlgError:
                cam_valid[ti] = False
                continue

        obj_l2w = self._build_object_local_to_world(bboxes_3d_ot83=bboxes_3d_ot83)
        num_obj = int(obj_l2w.shape[0])
        rng = np.random.default_rng(zlib.crc32(sample_key.encode("utf-8")) & 0xFFFFFFFF)
        quotas = np.full((t_clip,), max_queries // max(t_clip, 1), dtype=np.int64)
        quotas[: max_queries % max(t_clip, 1)] += 1

        query_points: list[np.ndarray] = []
        tracks_xyz: list[np.ndarray] = []
        visibility: list[np.ndarray] = []

        for fs in range(t_clip):
            quota = int(quotas[fs])
            if quota <= 0 or not bool(cam_valid[fs]):
                continue
            valid = depth_valid[fs].astype(bool)
            if not np.any(valid):
                continue
            fg_pool = np.argwhere(valid & (seg[fs] > 0))
            bg_pool = np.argwhere(valid & (seg[fs] <= 0))
            fg_quota = min(int(quota * 0.5), int(fg_pool.shape[0]))
            chosen = [self._sample_rows(fg_pool, fg_quota, rng)]
            chosen_count = int(chosen[0].shape[0])
            bg_quota = min(quota - chosen_count, int(bg_pool.shape[0]))
            chosen.append(self._sample_rows(bg_pool, bg_quota, rng))
            chosen_count += int(chosen[-1].shape[0])
            if chosen_count < quota:
                all_pool = np.argwhere(valid)
                chosen.append(self._sample_rows(all_pool, quota - chosen_count, rng))
            picks = np.concatenate([arr for arr in chosen if arr.shape[0] > 0], axis=0) if chosen else np.zeros((0, 2), dtype=np.int64)
            if picks.shape[0] == 0:
                continue

            # De-duplicate repeated picks from fill-in sampling.
            unique_coords: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for coord in picks.tolist():
                key = (int(coord[0]), int(coord[1]))
                if key in seen:
                    continue
                seen.add(key)
                unique_coords.append(key)

            for v_src, u_src in unique_coords:
                seg_id = int(seg[fs, v_src, u_src])
                p_world_seq = np.full((t_clip, 3), np.nan, dtype=np.float32)
                if seg_id > 0 and (seg_id - 1) < num_obj:
                    obj_idx = seg_id - 1
                    local = np.concatenate([obj_coord_local[fs, v_src, u_src], np.array([1.0], dtype=np.float32)], axis=0)
                    for ti in range(t_clip):
                        m_tgt = obj_l2w[obj_idx, ti]
                        if not np.isfinite(m_tgt).all():
                            continue
                        world_h = local @ m_tgt
                        if abs(float(world_h[3])) <= 1e-6:
                            continue
                        p_world_seq[ti] = (world_h[:3] / world_h[3]).astype(np.float32)
                else:
                    p_world = self._unproject_background_world(
                        u=int(u_src),
                        v=int(v_src),
                        depth_range_value=float(depth_range_m[fs, v_src, u_src]),
                        k=k_seq[fs],
                        t_wc=t_wc_seq[fs],
                    )
                    if np.isfinite(p_world).all():
                        p_world_seq[:] = p_world.astype(np.float32)

                if not np.isfinite(p_world_seq[fs]).all():
                    continue

                track_local = np.full((t_clip, 3), np.nan, dtype=np.float32)
                vis = np.zeros((t_clip,), dtype=np.bool_)
                for ti in range(t_clip):
                    if not bool(cam_valid[ti]) or not np.isfinite(p_world_seq[ti]).all():
                        continue
                    p_h = np.array([p_world_seq[ti, 0], p_world_seq[ti, 1], p_world_seq[ti, 2], 1.0], dtype=np.float32)
                    p_cam = (t_cw_seq[ti] @ p_h)[:3]
                    if np.isfinite(p_cam).all():
                        track_local[ti] = p_cam.astype(np.float32)
                    u_tgt, v_tgt, z_tgt = self._project_point(k_seq[ti], t_cw_seq[ti], p_world_seq[ti])
                    in_img = (
                        np.isfinite(u_tgt)
                        and np.isfinite(v_tgt)
                        and np.isfinite(z_tgt)
                        and (z_tgt > 1e-6)
                        and (0.0 <= u_tgt <= float(w - 1))
                        and (0.0 <= v_tgt <= float(h - 1))
                    )
                    if not in_img:
                        continue
                    vis[ti] = not self._depth_seg_occlusion(
                        depth_z_hw=depth_z[ti],
                        depth_valid_hw=depth_valid[ti],
                        seg_hw=seg[ti],
                        u=float(u_tgt),
                        v=float(v_tgt),
                        z_proj=float(z_tgt),
                        seg_id=int(seg_id),
                    )

                if not bool(vis[fs]):
                    continue
                query_points.append(np.array([float(u_src), float(v_src), float(fs)], dtype=np.float32))
                tracks_xyz.append(track_local.astype(np.float32))
                visibility.append(vis.astype(np.bool_))

        if not query_points:
            return None

        first_valid_k = np.where(cam_valid.astype(bool))[0]
        if first_valid_k.size == 0:
            return None
        k0 = k_seq[int(first_valid_k[0])].astype(np.float32)
        return {
            "queries_xyt": np.stack(query_points, axis=0).astype(np.float32),
            "tracks_xyz": np.stack(tracks_xyz, axis=0).astype(np.float32),
            "visibility": np.stack(visibility, axis=0).astype(np.bool_),
            "intrinsics_params": np.array(
                [float(k0[0, 0]), float(k0[1, 1]), float(k0[0, 2]), float(k0[1, 2])],
                dtype=np.float32,
            ),
            "camera_valid": cam_valid.astype(np.bool_),
            "t_wc": t_wc_seq.astype(np.float32),
        }

    def _unproject_background_world(
        self,
        u: int,
        v: int,
        depth_range_value: float,
        k: np.ndarray,
        t_wc: np.ndarray,
    ) -> np.ndarray:
        if not np.isfinite(depth_range_value) or depth_range_value <= 0.0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        x = (float(u) - cx) / max(fx, 1e-6)
        y = (float(v) - cy) / max(fy, 1e-6)
        ray = np.array([x, y, 1.0], dtype=np.float32)
        ray = ray / max(float(np.linalg.norm(ray)), 1e-6)
        p_cam = ray * float(depth_range_value)
        p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float32)
        p_w = (t_wc @ p_h)[:3]
        return p_w.astype(np.float32)

    def _build_queries(
        self,
        video_t_chw: np.ndarray,
        depth_z: np.ndarray,
        depth_range_m: np.ndarray,
        depth_valid: np.ndarray,
        seg: np.ndarray,
        obj_coord_local: np.ndarray,
        normal_world: np.ndarray,
        k_seq: np.ndarray,
        t_wc_seq: np.ndarray,
        bboxes_3d_ot83: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        del video_t_chw
        t_clip, h, w = depth_z.shape
        m = int(self.cfg.queries_per_clip)

        q_t_src = self.rng.integers(0, t_clip, size=(m,), dtype=np.int64)
        q_t_tgt, q_t_cam, _ = sample_t_tgt_t_cam(
            rng=self.rng,
            queries_per_clip=m,
            clip_frames=t_clip,
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            q_t_src=q_t_src,
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )
        use_hard = sample_hard_query_flags(self.rng, m, float(self.cfg.hard_query_ratio))

        q_u = np.zeros((m,), dtype=np.float32)
        q_v = np.zeros((m,), dtype=np.float32)
        y_uv = np.zeros((m, 2), dtype=np.float32)
        y_xyz = np.zeros((m, 3), dtype=np.float32)
        y_disp = np.zeros((m, 3), dtype=np.float32)
        y_normal = np.zeros((m, 3), dtype=np.float32)
        y_vis = np.zeros((m,), dtype=np.float32)

        m_uv = np.zeros((m,), dtype=np.bool_)
        m_xyz = np.zeros((m,), dtype=np.bool_)
        m_disp = np.zeros((m,), dtype=np.bool_)
        m_vis = np.zeros((m,), dtype=np.bool_)
        m_normal = np.zeros((m,), dtype=np.bool_)
        is_hard_query = np.zeros((m,), dtype=np.bool_)

        t_cw_seq = np.full((t_clip, 4, 4), np.nan, dtype=np.float32)
        cam_valid = np.isfinite(t_wc_seq).all(axis=(1, 2)) & np.isfinite(k_seq).all(axis=(1, 2))
        for i in range(t_clip):
            if not bool(cam_valid[i]):
                continue
            try:
                t_cw_seq[i] = np.linalg.inv(t_wc_seq[i]).astype(np.float32)
            except np.linalg.LinAlgError:
                cam_valid[i] = False

        obj_l2w = self._build_object_local_to_world(bboxes_3d_ot83)
        num_obj = obj_l2w.shape[0]

        hard_pix: list[np.ndarray] = []
        easy_pix: list[np.ndarray] = []
        for ti in range(t_clip):
            valid = depth_valid[ti] & bool(cam_valid[ti])
            sb = _seg_boundary_mask(seg[ti])
            db = depth_boundary_mask(depth_z[ti], depth_valid[ti], q=0.9)
            hb = valid & (sb | db)
            hard_pix.append(np.argwhere(hb))
            easy_pix.append(np.argwhere(valid))

        w_norm = max(1.0, float(w - 1))
        h_norm = max(1.0, float(h - 1))

        for i in range(m):
            fs = int(q_t_src[i])
            ft = int(q_t_tgt[i])
            fc = int(q_t_cam[i])
            if not (bool(cam_valid[fs]) and bool(cam_valid[ft]) and bool(cam_valid[fc])):
                continue

            pool = hard_pix[fs] if bool(use_hard[i]) and hard_pix[fs].shape[0] > 0 else easy_pix[fs]
            if pool.shape[0] == 0:
                continue
            is_hard_query[i] = bool(use_hard[i]) and hard_pix[fs].shape[0] > 0

            solved = False
            max_tries = 24
            for _ in range(max_tries):
                pick = int(self.rng.integers(0, pool.shape[0]))
                v_src = int(pool[pick, 0])
                u_src = int(pool[pick, 1])

                seg_id = int(seg[fs, v_src, u_src])
                p_world_src = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                p_world_tgt = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

                if seg_id > 0 and (seg_id - 1) < num_obj:
                    obj_idx = seg_id - 1
                    local = np.concatenate([obj_coord_local[fs, v_src, u_src], np.array([1.0], dtype=np.float32)], axis=0)
                    m_src = obj_l2w[obj_idx, fs]
                    m_tgt = obj_l2w[obj_idx, ft]
                    if np.isfinite(m_src).all() and np.isfinite(m_tgt).all():
                        w_src_h = local @ m_src
                        w_tgt_h = local @ m_tgt
                        if abs(float(w_src_h[3])) > 1e-6 and abs(float(w_tgt_h[3])) > 1e-6:
                            p_world_src = (w_src_h[:3] / w_src_h[3]).astype(np.float32)
                            p_world_tgt = (w_tgt_h[:3] / w_tgt_h[3]).astype(np.float32)

                if not np.isfinite(p_world_src).all():
                    # Background (or fallback): one-shot unprojection with radial depth.
                    d = float(depth_range_m[fs, v_src, u_src])
                    p_world_src = self._unproject_background_world(
                        u=u_src, v=v_src, depth_range_value=d, k=k_seq[fs], t_wc=t_wc_seq[fs]
                    )
                    p_world_tgt = p_world_src.copy()
                    seg_id = 0

                if not (np.isfinite(p_world_src).all() and np.isfinite(p_world_tgt).all()):
                    continue

                q_u[i] = float(u_src) / w_norm
                q_v[i] = float(v_src) / h_norm

                # xyz_3d in t_cam frame
                p_tgt_h = np.array([p_world_tgt[0], p_world_tgt[1], p_world_tgt[2], 1.0], dtype=np.float32)
                xyz_cam = (t_cw_seq[fc] @ p_tgt_h)[:3]
                if not np.isfinite(xyz_cam).all():
                    continue
                y_xyz[i] = xyz_cam.astype(np.float32)
                m_xyz[i] = True

                # displacement in t_cam
                delta_world = p_world_tgt - p_world_src
                disp_cam = t_cw_seq[fc, :3, :3] @ delta_world
                if np.isfinite(disp_cam).all():
                    y_disp[i] = disp_cam.astype(np.float32)
                    m_disp[i] = True

                # target uv + visibility
                u_tgt, v_tgt, z_tgt = self._project_point(k_seq[ft], t_cw_seq[ft], p_world_tgt)
                in_img = (
                    np.isfinite(u_tgt)
                    and np.isfinite(v_tgt)
                    and np.isfinite(z_tgt)
                    and (z_tgt > 1e-6)
                    and (0.0 <= u_tgt <= (w - 1))
                    and (0.0 <= v_tgt <= (h - 1))
                )
                if in_img:
                    y_uv[i, 0] = np.clip(float(u_tgt) / w_norm, 0.0, 1.0)
                    y_uv[i, 1] = np.clip(float(v_tgt) / h_norm, 0.0, 1.0)
                    m_uv[i] = True
                    occluded = self._depth_seg_occlusion(
                        depth_z_hw=depth_z[ft],
                        depth_valid_hw=depth_valid[ft],
                        seg_hw=seg[ft],
                        u=float(u_tgt),
                        v=float(v_tgt),
                        z_proj=float(z_tgt),
                        seg_id=int(seg_id),
                    )
                    y_vis[i] = 0.0 if occluded else 1.0
                    m_vis[i] = True
                else:
                    y_vis[i] = 0.0
                    m_vis[i] = True

                # normal: world normal at source pixel -> t_cam frame
                n_world = normal_world[fs, v_src, u_src]
                n_norm = float(np.linalg.norm(n_world))
                if np.isfinite(n_world).all() and n_norm > 1e-6:
                    n_world = n_world / n_norm
                    n_cam = t_cw_seq[fc, :3, :3] @ n_world
                    n_cam_norm = float(np.linalg.norm(n_cam))
                    if np.isfinite(n_cam).all() and n_cam_norm > 1e-6:
                        y_normal[i] = (n_cam / n_cam_norm).astype(np.float32)
                        m_normal[i] = True

                solved = True
                break

            if not solved:
                continue

        query = {"u": q_u, "v": q_v, "t_src": q_t_src, "t_tgt": q_t_tgt, "t_cam": q_t_cam}
        query_stats = {"is_hard_query": is_hard_query}
        target = {"xyz_3d": y_xyz, "uv_2d": y_uv, "visibility": y_vis, "displacement": y_disp, "normal": y_normal}
        mask = {"xyz_3d": m_xyz, "uv_2d": m_uv, "visibility": m_vis, "displacement": m_disp, "normal": m_normal}
        return query, target, mask, query_stats

    def _build_sample(self, sample: dict[str, Any], idxs: list[int], clip_start: int) -> dict[str, Any]:
        (
            video,
            depth_z,
            depth_range_m,
            seg,
            obj_coord_local,
            normal_world,
            k_seq,
            t_wc_seq,
        ) = self._camera_mats_and_depth(sample=sample, idxs=idxs)

        (
            video,
            depth_z,
            depth_range_m,
            seg,
            obj_coord_local,
            normal_world,
            k_seq,
        ) = self._resize_to_target(
            video=video,
            depth_z=depth_z,
            depth_range_m=depth_range_m,
            seg=seg,
            obj_coord=obj_coord_local,
            normal_world=normal_world,
            k_seq=k_seq,
        )
        depth_valid = np.isfinite(depth_z) & (depth_z > 0.0)
        t_clip = len(idxs)
        cam_valid = np.isfinite(t_wc_seq).all(axis=(1, 2)) & np.isfinite(k_seq).all(axis=(1, 2))

        video_t_chw = np.transpose(video, (0, 3, 1, 2)).astype(np.float32)
        src_h = int(np.asarray(sample["metadata"]["height"]).item())
        src_w = int(np.asarray(sample["metadata"]["width"]).item())
        aspect_ratio = np.array([float(src_w) / max(float(src_h), 1.0)], dtype=np.float32)

        _crop_info: dict[str, Any] = {}
        if self.cfg.training:
            video_t_chw = apply_photometric_augment(video_t_chw=video_t_chw, rng=self.rng, cfg=self.augment)
            (video_t_chw, depth_z, depth_valid, k_seq, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video_t_chw,
                depth_t_hw=depth_z,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_seq,
                camera_valid_t=cam_valid,
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )
            if "crop_hw" in _crop_info and "crop_xy" in _crop_info and "image_hw" in _crop_info:
                crop_hw = tuple(_crop_info["crop_hw"])
                crop_xy = tuple(_crop_info["crop_xy"])
                image_hw = tuple(_crop_info["image_hw"])
                seg = _apply_crop_resize_extra(seg, crop_xy=crop_xy, crop_hw=crop_hw, out_hw=image_hw, interp=cv2.INTER_NEAREST)
                obj_coord_local = _apply_crop_resize_extra(
                    obj_coord_local, crop_xy=crop_xy, crop_hw=crop_hw, out_hw=image_hw, interp=cv2.INTER_NEAREST
                )
                normal_world = _apply_crop_resize_extra(
                    normal_world, crop_xy=crop_xy, crop_hw=crop_hw, out_hw=image_hw, interp=cv2.INTER_NEAREST
                )
                depth_range_m = _apply_crop_resize_extra(
                    depth_range_m, crop_xy=crop_xy, crop_hw=crop_hw, out_hw=image_hw, interp=cv2.INTER_NEAREST
                )
                seg = seg.astype(np.int32)
                obj_coord_local = obj_coord_local.astype(np.float32)
                normal_world = normal_world.astype(np.float32)
                depth_range_m = depth_range_m.astype(np.float32)

        instances = sample["instances"]
        bboxes_all = np.asarray(instances["bboxes_3d"], dtype=np.float32)  # [N_obj,T_all,8,3]
        bboxes_clip = bboxes_all[:, idxs]

        query, target, mask, query_stats = self._build_queries(
            video_t_chw=video_t_chw,
            depth_z=depth_z,
            depth_range_m=depth_range_m,
            depth_valid=depth_valid,
            seg=seg,
            obj_coord_local=obj_coord_local,
            normal_world=normal_world,
            k_seq=k_seq,
            t_wc_seq=t_wc_seq,
            bboxes_3d_ot83=bboxes_clip,
        )

        video_name_raw = sample["metadata"]["video_name"]
        if isinstance(video_name_raw, bytes):
            video_name = video_name_raw.decode("utf-8", errors="ignore")
        else:
            video_name = str(video_name_raw)

        benchmark_payload: dict[str, Any] = {}
        if bool(self.cfg.benchmark_tracking_enabled):
            benchmark_tracking = self._build_benchmark_tracking(
                depth_z=depth_z,
                depth_range_m=depth_range_m,
                depth_valid=depth_valid,
                seg=seg,
                obj_coord_local=obj_coord_local,
                k_seq=k_seq,
                t_wc_seq=t_wc_seq,
                bboxes_3d_ot83=bboxes_clip,
                sample_key=f"{video_name}::start={int(clip_start)}",
            )
            if benchmark_tracking is not None:
                benchmark_payload["benchmark_tracking"] = {
                    key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for key, value in benchmark_tracking.items()
                }

        return {
            **benchmark_payload,
            "video": torch.from_numpy(video_t_chw).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth_z.astype(np.float32)).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_seq).float(),
                "T_wc": torch.from_numpy(t_wc_seq).float(),
                "camera_valid": torch.from_numpy(cam_valid).bool(),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()},
            "meta": {
                "dataset": "kubric_full_robust",
                "scene_id": video_name,
                "clip_start": int(clip_start),
                "source_mode": "kubric_movi_full_objectcoord",
                "native_hw": (src_h, src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))
        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            sample_key = ""
            sample_paths = [str(self.tfds_dir)]
            try:
                if self.cfg.training:
                    sample = self._next_train_sample()
                else:
                    eval_index = query_index % self.num_examples
                    sample = self._eval_sample_by_index(eval_index)

                self._validate_runtime_sample(sample)
                idxs = self._select_clip(sample=sample, index=query_index)
                clip_start = int(idxs[0]) if idxs else 0
                video_name_raw = sample["metadata"]["video_name"]
                video_name = (
                    video_name_raw.decode("utf-8", errors="ignore")
                    if isinstance(video_name_raw, bytes)
                    else str(video_name_raw)
                )
                sample_key = f"kubric_full_robust::{video_name}::frames={','.join(str(int(v)) for v in idxs)}"
                if self.bad_registry.is_bad_sample(sample_key):
                    continue
                out = self._build_sample(sample=sample, idxs=idxs, clip_start=clip_start)
                out["meta"]["sample_key"] = sample_key
                return out
            except Exception as exc:
                failed_paths = self._failed_paths_from_exception(exc, default=[str(self.tfds_dir)])
                if not is_retryable_data_error(exc):
                    retry_exc = RetryableSampleError(
                        f"Converted non-retryable sample failure: {type(exc).__name__}: {exc}",
                        failed_paths=failed_paths,
                    )
                    last_error = retry_exc
                    self.bad_registry.mark_bad(
                        dataset="kubric_full_robust",
                        sample_key=sample_key or f"kubric_full_robust::index={query_index}",
                        sample_paths=_dedup_str_list(sample_paths + failed_paths),
                        failed_paths=failed_paths,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self._warn_skip(
                        sample_key=sample_key or f"kubric_full_robust::index={query_index}",
                        reason=f"{type(exc).__name__}: {exc}",
                        failed_paths=failed_paths,
                    )
                    continue
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="kubric_full_robust",
                    sample_key=sample_key or f"kubric_full_robust::index={query_index}",
                    sample_paths=_dedup_str_list(sample_paths + failed_paths),
                    failed_paths=failed_paths,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._warn_skip(
                    sample_key=sample_key or f"kubric_full_robust::index={query_index}",
                    reason=f"{type(exc).__name__}: {exc}",
                    failed_paths=failed_paths,
                )
                continue

        raise RuntimeError(
            f"KubricFullRobustDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
