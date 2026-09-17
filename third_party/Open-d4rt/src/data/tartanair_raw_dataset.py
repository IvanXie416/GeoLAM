"""TartanAir V2 raw dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    sample_frame_indices_with_stride,
)
from .seeding import SeededDatasetMixin, stable_split_bucket


# Fixed intrinsics for TartanAir V2 (640x640, 90-deg FOV).
_SRC_W = 640
_SRC_H = 640
_FX = 320.0
_FY = 320.0
_CX = 320.0
_CY = 240.0  # V1 default; overridden for V2 square images

# NED-body → standard camera frame:
#   cam_x (right)   = ned_y (east)
#   cam_y (down)    = ned_z (down)
#   cam_z (forward) = ned_x (north)
_T_CAM_FROM_NED = np.array(
    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
_T_NED_FROM_CAM = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _ned_pose_to_t_wc(pose_7: np.ndarray) -> np.ndarray:
    """Convert TartanAir NED pose (tx ty tz qx qy qz qw) to T_wc in standard camera frame.

    The pose describes camera-to-world in the NED coordinate system.  We compose
    with the NED↔camera rotation so that the resulting T_wc maps from the
    standard camera frame (x-right, y-down, z-forward) to world NED.
    """
    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in pose_7)
    r_wc_ned = _quat_to_rotation(qx, qy, qz, qw)
    t_wc_ned = np.eye(4, dtype=np.float64)
    t_wc_ned[:3, :3] = r_wc_ned
    t_wc_ned[:3, 3] = [tx, ty, tz]
    # T_wc_cam = T_wc_ned @ T_ned_from_cam
    t_wc = (t_wc_ned @ _T_NED_FROM_CAM).astype(np.float32)
    return t_wc


def _read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_depth_rgba_png(path: Path, width: int, height: int) -> np.ndarray:
    """Read TartanAir V2 RGBA-encoded depth PNG and return depth in meters.

    The 4 RGBA uint8 channels encode a float32 depth value (little-endian).
    We must use cv2.IMREAD_UNCHANGED to preserve the BGRA byte order that
    matches the original encoding; PIL reads RGBA which swaps bytes 0 and 2,
    producing garbled float32 values.
    """
    try:
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise RetryableSampleError(f"cv2.imread returned None: {path}", failed_paths=[str(path)])
        depth = bgra.view("<f4").squeeze(-1).astype(np.float32)
        if (width, height) != (depth.shape[1], depth.shape[0]):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        return depth
    except RetryableSampleError:
        raise
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read depth image: {path}: {exc}", failed_paths=[str(path)]) from exc


@dataclass
class TartanairRawConfig:
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
    camera_name: str = "lcam_front"
    difficulties: list[str] = field(default_factory=lambda: ["Data_easy", "Data_hard"])
    max_scenes: int | None = None
    split_modulo: int = 20
    max_depth_m: float = 1000.0
    intrinsics: list[float] | None = None  # [fx, fy, cx, cy]; None → auto from V2 defaults
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _Trajectory:
    scene_id: str
    traj_dir: Path
    rgb_paths: list[Path]
    depth_paths: list[Path]
    t_wc_seq: np.ndarray  # (N, 4, 4) float32
    frame_count: int
    src_h: int
    src_w: int


class TartanairRawDataset(SeededDatasetMixin, Dataset):
    """Loads TartanAir V2 RGBD + poses and builds depth-projected supervision."""

    def __init__(self, config: TartanairRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="tartanair_raw", default_seed=20260321)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        if not config.training:
            self.augment = RawAugmentConfig()

        if not config.root.exists():
            raise FileNotFoundError(f"TartanAir root not found: {config.root}")

        # Build per-pixel intrinsics matrix for the *source* resolution (before resize).
        if config.intrinsics and len(config.intrinsics) >= 4:
            fx, fy, cx, cy = (float(v) for v in config.intrinsics[:4])
        else:
            fx, fy, cx, cy = _FX, _FY, _SRC_W / 2.0, _SRC_H / 2.0
        self._k_src = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        self.trajectories: list[_Trajectory] = []
        self._enumerate_trajectories()

        if not self.trajectories:
            raise ValueError(f"No valid TartanAir trajectories found for split={config.split} under {config.root}")

    # ------------------------------------------------------------------
    # Scene / trajectory enumeration
    # ------------------------------------------------------------------

    def _enumerate_trajectories(self) -> None:
        cam = str(self.cfg.camera_name)
        scene_dirs = sorted([p for p in self.cfg.root.iterdir() if p.is_dir()])

        for scene_dir in scene_dirs:
            for diff in self.cfg.difficulties:
                diff_dir = scene_dir / diff
                if not diff_dir.is_dir():
                    continue
                traj_dirs = sorted([p for p in diff_dir.iterdir() if p.is_dir() and p.name.startswith("P")])
                for traj_dir in traj_dirs:
                    self._try_add_trajectory(scene_dir.name, diff, traj_dir, cam)
                    if self.cfg.max_scenes is not None and len(self.trajectories) >= int(self.cfg.max_scenes):
                        return

    def _try_add_trajectory(self, scene_name: str, difficulty: str, traj_dir: Path, cam: str) -> None:
        rgb_dir = traj_dir / f"image_{cam}"
        depth_dir = traj_dir / f"depth_{cam}"
        pose_file = traj_dir / f"pose_{cam}.txt"
        if not (rgb_dir.is_dir() and depth_dir.is_dir() and pose_file.exists()):
            return

        scene_id = f"{scene_name}/{difficulty}/{traj_dir.name}/{cam}"

        # Split by stable hash modulo so all processes see the same partition.
        h = stable_split_bucket(scene_id, modulo=self.cfg.split_modulo)
        split = self._effective_split()
        if split == "train" and h == 0:
            return
        if split in ("val", "test") and h != 0:
            return

        # Enumerate frames.
        rgb_files = sorted(rgb_dir.glob("*.png"))
        depth_files = sorted(depth_dir.glob("*.png"))
        if len(rgb_files) < self.cfg.clip_frames or len(depth_files) < self.cfg.clip_frames:
            return

        # Match frame indices via sorted filename prefix (XXXXXX_*).
        def _frame_idx(p: Path) -> int:
            return int(p.stem.split("_")[0])

        rgb_by_idx = {_frame_idx(p): p for p in rgb_files}
        depth_by_idx = {_frame_idx(p): p for p in depth_files}

        # Load poses (one per frame, line-indexed).
        try:
            raw_poses = np.loadtxt(pose_file)
        except Exception:
            return
        if raw_poses.ndim != 2 or raw_poses.shape[1] != 7:
            return
        num_poses = raw_poses.shape[0]

        # Only keep frames for which rgb, depth, and pose all exist.
        common = sorted(
            idx
            for idx in set(rgb_by_idx.keys()).intersection(depth_by_idx.keys())
            if idx < num_poses
        )
        if len(common) < self.cfg.clip_frames:
            return

        rgb_paths = [rgb_by_idx[idx] for idx in common]
        depth_paths = [depth_by_idx[idx] for idx in common]

        # Convert all poses from NED → standard camera frame.
        t_wc_seq = np.stack([_ned_pose_to_t_wc(raw_poses[idx]) for idx in common], axis=0)

        # Determine source resolution from first image.
        try:
            src_w, src_h = Image.open(rgb_paths[0]).size
        except Exception:
            src_w, src_h = _SRC_W, _SRC_H

        self.trajectories.append(
            _Trajectory(
                scene_id=scene_id,
                traj_dir=traj_dir,
                rgb_paths=rgb_paths,
                depth_paths=depth_paths,
                t_wc_seq=t_wc_seq.astype(np.float32),
                frame_count=len(common),
                src_h=int(src_h),
                src_w=int(src_w),
            )
        )

    def _effective_split(self) -> str:
        split = str(self.cfg.split).strip().lower()
        if isinstance(self.cfg.split_map, dict):
            mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in self.cfg.split_map.items()}
            return mapping.get(split, mapping.get("train", split))
        return split

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        base = len(self.trajectories) * 30 if self.cfg.training else len(self.trajectories)
        return max(base, len(self.trajectories))

    def _trajectory(self, index: int) -> _Trajectory:
        if self.cfg.training:
            return self.trajectories[int(self.rng.integers(0, len(self.trajectories)))]
        return self.trajectories[index % len(self.trajectories)]

    def _frame_indices(self, traj_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng, scene_len=traj_len, clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment, training=bool(self.cfg.training), index=index,
        )

    def _sample_key(self, traj: _Trajectory, idxs: list[int]) -> str:
        frame_token = ",".join(str(i) for i in idxs)
        return f"tartanair_raw::{traj.scene_id}::frames={frame_token}"

    def _sample_paths(self, traj: _Trajectory, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            out.append(str(traj.rgb_paths[i]))
            out.append(str(traj.depth_paths[i]))
        return out

    # ------------------------------------------------------------------
    # Sample building
    # ------------------------------------------------------------------

    def _build_sample(self, traj: _Trajectory, idxs: list[int], clip_start: int) -> dict[str, Any]:
        video_list: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        depth_valid_list: list[np.ndarray] = []
        k_seq: list[np.ndarray] = []
        t_wc_seq: list[np.ndarray] = []
        camera_valid: list[bool] = []

        # Scale intrinsics from source resolution to target resolution.
        sx = float(self.w) / max(float(traj.src_w), 1.0)
        sy = float(self.h) / max(float(traj.src_h), 1.0)
        k_resized = self._k_src.copy()
        k_resized[0, 0] *= sx
        k_resized[0, 2] *= sx
        k_resized[1, 1] *= sy
        k_resized[1, 2] *= sy

        for i in idxs:
            rgb = _read_rgb(traj.rgb_paths[i], width=self.w, height=self.h)
            depth_m = _read_depth_rgba_png(traj.depth_paths[i], width=self.w, height=self.h)
            valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < float(self.cfg.max_depth_m))

            video_list.append(rgb)
            depth_list.append(depth_m)
            depth_valid_list.append(valid.astype(np.bool_))
            k_seq.append(k_resized.copy())
            t_wc_seq.append(traj.t_wc_seq[i])
            camera_valid.append(bool(np.isfinite(traj.t_wc_seq[i]).all()))

        video = np.stack(video_list, axis=0).astype(np.float32) / 255.0
        video = np.transpose(video, (0, 3, 1, 2))
        depth = np.stack(depth_list, axis=0).astype(np.float32)
        depth_valid = np.stack(depth_valid_list, axis=0).astype(np.bool_)
        k_arr = np.stack(k_seq, axis=0).astype(np.float32)
        t_wc_arr = np.stack(t_wc_seq, axis=0).astype(np.float32)
        cam_valid = np.asarray(camera_valid, dtype=np.bool_)

        aspect_ratio = np.array([traj.src_w / max(float(traj.src_h), 1.0)], dtype=np.float32)
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
                "dataset": "tartanair_raw",
                "scene_id": traj.scene_id,
                "clip_start": int(clip_start),
                "source_mode": "tartanair_depth_reproject",
                "native_hw": (traj.src_h, traj.src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            traj = self._trajectory(query_index)
            idxs = self._frame_indices(traj.frame_count, query_index)
            clip_start = int(idxs[0]) if idxs else 0
            sample_key = self._sample_key(traj, idxs)
            sample_paths = self._sample_paths(traj, idxs)

            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue

            try:
                sample = self._build_sample(traj=traj, idxs=idxs, clip_start=clip_start)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="tartanair_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"TartanairRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
