"""Dynamic Replica raw dataset adapter with query supervision in t_cam coordinates."""

from __future__ import annotations

import json
import warnings
import zlib
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
from .depth_query_builder import build_queries_from_depth, build_queries_from_trajectories
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    sample_frame_indices_with_stride,
)
from .seeding import SeededDatasetMixin


def _resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGBA")
        rgb = np.array(img, dtype=np.uint8)[..., :3]
        pil = Image.fromarray(rgb, mode="RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.array(pil, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _resize_depth(path: Path, width: int, height: int) -> np.ndarray:
    try:
        dep = np.array(Image.open(path), dtype=np.uint16)
        pil = Image.fromarray(dep, mode="I;16").resize((width, height), resample=Image.Resampling.NEAREST)
        return np.array(pil, dtype=np.uint16)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read depth image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _decode_depth_float16_bitcast(depth_u16: np.ndarray) -> np.ndarray:
    dep = np.ascontiguousarray(depth_u16, dtype=np.uint16)
    depth = dep.view(np.float16).astype(np.float32)
    return depth.reshape(dep.shape)


def _decode_depth_uint16_divisor(depth_u16: np.ndarray, depth_divisor: float) -> np.ndarray:
    depth = depth_u16.astype(np.float32) / max(float(depth_divisor), 1e-6)
    # Common invalid sentinels in Dynamic Replica releases.
    invalid = (depth_u16 == 0) | (depth_u16 == 32768) | (depth_u16 == 65535)
    depth[invalid] = 0.0
    return depth


def _depth_decode_plausibility(depth_m: np.ndarray, eps: float = 1e-6) -> float:
    finite = np.isfinite(depth_m)
    if finite.mean() < 0.5:
        return -1e9
    pos = depth_m[finite & (depth_m > float(eps))]
    if pos.size < 2048:
        return -1e9
    q01, q50, q99 = np.percentile(pos, [1.0, 50.0, 99.0]).astype(np.float32)
    valid_ratio = float(pos.size) / float(depth_m.size)
    score = 0.0
    if 0.02 <= float(q01) <= 5.0:
        score += 1.0
    if 0.05 <= float(q50) <= 20.0:
        score += 2.0
    if 0.1 <= float(q99) <= 200.0:
        score += 2.0
    if 0.05 <= valid_ratio <= 0.995:
        score += 1.0
    if float(q99) > float(q50) * 1.2:
        score += 0.5
    return score


def _load_traj(path: Path) -> dict[str, np.ndarray]:
    try:
        pack = torch.load(path, map_location="cpu")
        return {
            "traj_2d": pack["traj_2d"].numpy().astype(np.float32),
            "traj_3d_world": pack["traj_3d_world"].numpy().astype(np.float32),
            "verts_inds_vis": pack["verts_inds_vis"].numpy().astype(np.bool_),
        }
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read trajectory file: {path}: {exc}", failed_paths=[str(path)]) from exc


def _annotation_filename(split: str) -> str:
    mapping = {"train": "frame_annotations_train.json", "valid": "frame_annotations_valid.json", "test": "frame_annotations_test.json"}
    return mapping.get(split, f"frame_annotations_{split}.json")


def _viewpoint_to_camera(
    vp: dict[str, Any],
    image_h: int,
    image_w: int,
    camera_convention: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(vp["R"], dtype=np.float32)
    t = np.asarray(vp["T"], dtype=np.float32)
    focal = np.asarray(vp.get("focal_length", [1.0, 1.0]), dtype=np.float32)
    principal = np.asarray(vp.get("principal_point", [0.0, 0.0]), dtype=np.float32)
    intrinsics_format = str(vp.get("intrinsics_format", "ndc_isotropic")).lower()
    convention = str(camera_convention).strip().lower()

    if intrinsics_format == "ndc_isotropic":
        if convention == "dynamic_replica_v2":
            scale = 0.5 * float(min(image_h, image_w))
        elif convention == "legacy":
            scale = 0.5 * float(max(image_h, image_w))
        else:
            raise ValueError(f"Unsupported camera convention: {camera_convention}")
        fx = float(focal[0] * scale)
        fy = float(focal[1] * scale)
        cx = 0.5 * float(image_w - 1) * (1.0 - float(principal[0]))
        cy = 0.5 * float(image_h - 1) * (1.0 - float(principal[1]))
    else:
        fx = float(focal[0])
        fy = float(focal[1])
        cx = float(principal[0])
        cy = float(principal[1])

    if convention == "dynamic_replica_v2":
        # Dynamic Replica viewpoint uses a flipped camera basis in this raw release.
        # Empirically consistent with traj_2d/traj_3d_world when using R^T plus XY flip.
        axis_flip = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        r_cw = axis_flip @ r.T
        t_cw_vec = axis_flip @ t
    elif convention == "legacy":
        r_cw = r
        t_cw_vec = t
    else:
        raise ValueError(f"Unsupported camera convention: {camera_convention}")

    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    t_cw = np.eye(4, dtype=np.float32)
    t_cw[:3, :3] = r_cw
    t_cw[:3, 3] = t_cw_vec
    t_wc = np.linalg.inv(t_cw).astype(np.float32)
    return k, t_cw, t_wc


def _world_to_cam(x_world: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    x_h = np.concatenate([x_world.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)
    x_cam = t_cw @ x_h
    return x_cam[:3]


@dataclass
class DynamicReplicaRawConfig:
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
    camera_convention: str = "dynamic_replica_v2"
    depth_decode_mode: str = "auto"  # auto|float16_bitcast|uint16_divisor
    depth_divisor: float = 10000.0
    reprojection_self_check_enabled: bool = True
    reprojection_self_check_mode: str = "warn"  # off|warn|raise
    reprojection_self_check_median_threshold_px: float = 5.0
    reprojection_self_check_max_scenes: int = 1
    reprojection_self_check_max_frames: int = 4
    reprojection_self_check_max_points: int = 4096
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64
    benchmark_tracking_enabled: bool = False
    benchmark_max_queries: int = 0


@dataclass
class _FrameCamera:
    k_src: np.ndarray
    t_cw: np.ndarray
    t_wc: np.ndarray
    depth_scale_adjustment: float
    src_h: int
    src_w: int
    valid: bool


@dataclass
class _Scene:
    name: str
    path: Path
    frame_count: int
    src_h: int
    src_w: int


class DynamicReplicaRawDataset(SeededDatasetMixin, Dataset):
    """Builds D4RT samples from Dynamic Replica with explicit t_cam geometry transforms."""

    def __init__(self, config: DynamicReplicaRawConfig) -> None:
        self.cfg = config
        self._validate_config()
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="dynamic_replica_raw", default_seed=20260319)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        self._resolved_depth_decode_mode: str | None = None
        self._warned_depth_fallback_scenes: set[str] = set()
        if not config.training:
            self.augment = RawAugmentConfig()
        split_dir = config.root / config.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dynamic Replica split dir not found: {split_dir}")

        self.camera_index = self._load_camera_index(split_dir, config.split)
        self.camera_cache: dict[tuple[str, int], _FrameCamera] = {}

        self.scenes: list[_Scene] = []
        for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            image_count = len(list((scene_dir / "images").glob("*.png")))
            traj_count = len(list((scene_dir / "trajectories").glob("*.pth")))
            depth_count = len(list((scene_dir / "depths").glob("*.geometric.png")))
            n = min(image_count, traj_count, depth_count)
            if n < config.clip_frames:
                continue

            src_h, src_w = self._scene_size(scene_dir.name, scene_dir)
            self.scenes.append(
                _Scene(
                    name=scene_dir.name,
                    path=scene_dir,
                    frame_count=n,
                    src_h=src_h,
                    src_w=src_w,
                )
            )
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break

        if not self.scenes:
            raise ValueError(f"No valid Dynamic Replica scenes with >= {config.clip_frames} frames in {split_dir}")
        self._run_reprojection_self_check()

    def _warn_depth_fallback(self, scene: _Scene, reason: str) -> None:
        if scene.name in self._warned_depth_fallback_scenes:
            return
        self._warned_depth_fallback_scenes.add(scene.name)
        warnings.warn(
            "DynamicReplicaRawDataset falling back to depth-based query construction "
            f"for scene '{scene.name}' because trajectory-based 3D supervision is unavailable: {reason}",
            stacklevel=2,
        )

    def _build_benchmark_tracking(
        self,
        *,
        scene: _Scene,
        traj_2d: np.ndarray,
        traj_3d_world: np.ndarray,
        traj_visible: np.ndarray,
        k_seq: np.ndarray,
        t_wc_seq: np.ndarray,
        camera_valid: np.ndarray,
        sample_key: str,
    ) -> dict[str, np.ndarray] | None:
        t_clip, n_pts = int(traj_visible.shape[0]), int(traj_visible.shape[1])
        if t_clip <= 0 or n_pts <= 0:
            return None

        t_cw_seq = np.full((t_clip, 4, 4), np.nan, dtype=np.float32)
        for ti in range(t_clip):
            if not bool(camera_valid[ti]):
                continue
            try:
                t_cw_seq[ti] = np.linalg.inv(t_wc_seq[ti]).astype(np.float32)
            except np.linalg.LinAlgError:
                continue

        sx = float(self.w) / max(float(scene.src_w), 1.0)
        sy = float(self.h) / max(float(scene.src_h), 1.0)
        traj_uv = traj_2d[..., :2].astype(np.float32).copy()
        traj_uv[..., 0] *= sx
        traj_uv[..., 1] *= sy

        tracks_local = np.full((t_clip, n_pts, 3), np.nan, dtype=np.float32)
        visibility = np.zeros((t_clip, n_pts), dtype=np.bool_)
        for ti in range(t_clip):
            if not bool(camera_valid[ti]) or not np.isfinite(t_cw_seq[ti]).all():
                continue
            pts_world = traj_3d_world[ti]
            valid_pts = np.isfinite(pts_world).all(axis=1)
            if not np.any(valid_pts):
                continue
            ids = np.where(valid_pts)[0]
            pts_h = np.concatenate(
                [pts_world[ids].astype(np.float32), np.ones((ids.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            pts_cam = (t_cw_seq[ti] @ pts_h.T).T[:, :3]
            tracks_local[ti, ids] = pts_cam.astype(np.float32)
            uv = traj_uv[ti, ids]
            in_img = (
                np.isfinite(uv).all(axis=1)
                & np.isfinite(pts_cam).all(axis=1)
                & (pts_cam[:, 2] > 1e-6)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= float(self.w - 1))
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= float(self.h - 1))
            )
            visibility[ti, ids] = traj_visible[ti, ids].astype(bool) & in_img

        first_visible = np.full((n_pts,), -1, dtype=np.int64)
        for ti in range(t_clip):
            update = (first_visible < 0) & visibility[ti]
            first_visible[update] = int(ti)
        point_ids = np.where(first_visible >= 0)[0].astype(np.int64)
        if point_ids.size == 0:
            return None

        max_queries = int(self.cfg.benchmark_max_queries)
        if max_queries > 0 and point_ids.size > max_queries:
            rng = np.random.default_rng(zlib.crc32(sample_key.encode("utf-8")) & 0xFFFFFFFF)
            point_ids = np.sort(rng.choice(point_ids, size=max_queries, replace=False).astype(np.int64))

        src_frames = first_visible[point_ids]
        query_uv = traj_uv[src_frames, point_ids]
        first_valid_k = np.where(camera_valid.astype(bool))[0]
        if first_valid_k.size == 0:
            return None
        k0 = k_seq[int(first_valid_k[0])].astype(np.float32)
        return {
            "queries_xyt": np.stack(
                [query_uv[:, 0], query_uv[:, 1], src_frames.astype(np.float32)],
                axis=1,
            ).astype(np.float32),
            "tracks_xyz": np.transpose(tracks_local[:, point_ids], (1, 0, 2)).astype(np.float32),
            "visibility": np.transpose(visibility[:, point_ids], (1, 0)).astype(np.bool_),
            "intrinsics_params": np.array(
                [float(k0[0, 0]), float(k0[1, 1]), float(k0[0, 2]), float(k0[1, 2])],
                dtype=np.float32,
            ),
            "camera_valid": camera_valid.astype(np.bool_),
            "t_wc": t_wc_seq.astype(np.float32),
        }

    def _validate_config(self) -> None:
        if not np.isfinite(float(self.cfg.depth_divisor)) or float(self.cfg.depth_divisor) <= 0.0:
            raise ValueError(f"data.dynamic_replica.depth_divisor must be > 0, got {self.cfg.depth_divisor}")
        decode_mode = str(self.cfg.depth_decode_mode).strip().lower()
        if decode_mode not in {"auto", "float16_bitcast", "uint16_divisor"}:
            raise ValueError(
                "data.dynamic_replica.depth_decode_mode must be one of: auto|float16_bitcast|uint16_divisor, "
                f"got {self.cfg.depth_decode_mode}"
            )
        mode = str(self.cfg.reprojection_self_check_mode).strip().lower()
        if mode not in {"off", "warn", "raise"}:
            raise ValueError(
                "data.dynamic_replica.reprojection_self_check.mode must be one of: off|warn|raise, "
                f"got {self.cfg.reprojection_self_check_mode}"
            )
        convention = str(self.cfg.camera_convention).strip().lower()
        if convention not in {"dynamic_replica_v2", "legacy"}:
            raise ValueError(
                "data.dynamic_replica.camera_convention must be one of: dynamic_replica_v2|legacy, "
                f"got {self.cfg.camera_convention}"
            )

    def _decode_depth(self, depth_u16: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.depth_decode_mode).strip().lower()
        if mode == "float16_bitcast":
            return _decode_depth_float16_bitcast(depth_u16)
        if mode == "uint16_divisor":
            return _decode_depth_uint16_divisor(depth_u16, float(self.cfg.depth_divisor))

        bitcast = _decode_depth_float16_bitcast(depth_u16)
        divisor = _decode_depth_uint16_divisor(depth_u16, float(self.cfg.depth_divisor))
        if self._resolved_depth_decode_mode is None:
            bitcast_score = _depth_decode_plausibility(bitcast)
            divisor_score = _depth_decode_plausibility(divisor)
            self._resolved_depth_decode_mode = "float16_bitcast" if bitcast_score >= divisor_score else "uint16_divisor"
        if self._resolved_depth_decode_mode == "float16_bitcast":
            return bitcast
        return divisor

    def _load_camera_index(self, split_dir: Path, split: str) -> dict[str, dict[int, _FrameCamera]]:
        annotation_path = split_dir / _annotation_filename(split)
        if not annotation_path.exists():
            return {}

        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        index: dict[str, dict[int, _FrameCamera]] = {}
        for item in raw:
            image_path = str(item.get("image", {}).get("path", ""))
            if "/" not in image_path:
                continue
            scene_name = image_path.split("/", 1)[0]
            frame_id = int(item.get("frame_number", -1))
            if frame_id < 0:
                continue
            image_size = item.get("image", {}).get("size", [0, 0])
            image_h = int(image_size[0]) if len(image_size) > 0 else 0
            image_w = int(image_size[1]) if len(image_size) > 1 else 0
            vp = item.get("viewpoint", None)
            depth_scale = float(item.get("depth", {}).get("scale_adjustment", 1.0))

            if vp is None or image_h <= 0 or image_w <= 0:
                cam = _FrameCamera(
                    k_src=np.full((3, 3), np.nan, dtype=np.float32),
                    t_cw=np.full((4, 4), np.nan, dtype=np.float32),
                    t_wc=np.full((4, 4), np.nan, dtype=np.float32),
                    depth_scale_adjustment=depth_scale,
                    src_h=max(1, image_h),
                    src_w=max(1, image_w),
                    valid=False,
                )
            else:
                k_src, t_cw, t_wc = _viewpoint_to_camera(
                    vp,
                    image_h=image_h,
                    image_w=image_w,
                    camera_convention=self.cfg.camera_convention,
                )
                valid = bool(np.isfinite(k_src).all() and np.isfinite(t_wc).all())
                cam = _FrameCamera(
                    k_src=k_src,
                    t_cw=t_cw,
                    t_wc=t_wc,
                    depth_scale_adjustment=depth_scale,
                    src_h=image_h,
                    src_w=image_w,
                    valid=valid,
                )
            index.setdefault(scene_name, {})[frame_id] = cam
        return index

    def _run_reprojection_self_check(self) -> None:
        if not bool(self.cfg.reprojection_self_check_enabled):
            return
        mode = str(self.cfg.reprojection_self_check_mode).strip().lower()
        if mode == "off":
            return
        max_scenes = max(1, int(self.cfg.reprojection_self_check_max_scenes))
        max_frames = max(1, int(self.cfg.reprojection_self_check_max_frames))
        max_points = max(1, int(self.cfg.reprojection_self_check_max_points))
        local_rng = np.random.default_rng(0)

        errs: list[np.ndarray] = []
        checked_frames = 0
        checked_scenes = 0
        for scene in self.scenes[:max_scenes]:
            checked_scenes += 1
            frame_ids = np.linspace(0, max(0, scene.frame_count - 1), num=max_frames, dtype=np.int64).tolist()
            for fid in frame_ids:
                _, _, traj_path = self._frame_paths(scene, int(fid))
                if not traj_path.exists():
                    continue
                cam = self._frame_camera(scene, int(fid))
                if not cam.valid:
                    continue
                try:
                    traj = _load_traj(traj_path)
                except Exception:
                    continue
                uv = traj["traj_2d"].astype(np.float32)
                xyz_world = traj["traj_3d_world"].astype(np.float32)
                vis = traj["verts_inds_vis"].astype(bool)

                x_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)], axis=1)
                x_cam = (cam.t_cw @ x_h.T).T[:, :3]
                valid = vis & np.isfinite(uv).all(axis=-1) & np.isfinite(x_cam).all(axis=-1) & (x_cam[:, 2] > 1e-6)
                valid_ids = np.flatnonzero(valid)
                if valid_ids.size == 0:
                    continue
                if valid_ids.size > max_points:
                    pick = local_rng.choice(valid_ids.size, size=max_points, replace=False)
                    valid_ids = valid_ids[pick]

                proj_u = cam.k_src[0, 0] * (x_cam[valid_ids, 0] / x_cam[valid_ids, 2]) + cam.k_src[0, 2]
                proj_v = cam.k_src[1, 1] * (x_cam[valid_ids, 1] / x_cam[valid_ids, 2]) + cam.k_src[1, 2]
                proj = np.stack([proj_u, proj_v], axis=-1)
                err = np.linalg.norm(proj - uv[valid_ids, :2], axis=-1).astype(np.float32)
                if err.size > 0:
                    errs.append(err)
                    checked_frames += 1

        if not errs:
            return
        all_err = np.concatenate(errs, axis=0).astype(np.float32)
        median_px = float(np.median(all_err))
        p95_px = float(np.percentile(all_err, 95))
        threshold_px = float(self.cfg.reprojection_self_check_median_threshold_px)
        if median_px <= threshold_px:
            return

        msg = (
            "DynamicReplicaRawDataset reprojection self-check failed: "
            f"median={median_px:.3f}px p95={p95_px:.3f}px threshold={threshold_px:.3f}px "
            f"(checked_scenes={checked_scenes}, checked_frames={checked_frames}, points={int(all_err.size)}). "
            f"camera_convention={self.cfg.camera_convention}, depth_divisor={float(self.cfg.depth_divisor):.1f}"
        )
        if mode == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    def _scene_size(self, scene_name: str, scene_dir: Path) -> tuple[int, int]:
        scene_cams = self.camera_index.get(scene_name, {})
        if scene_cams:
            first = scene_cams[min(scene_cams.keys())]
            return int(first.src_h), int(first.src_w)
        image_files = sorted((scene_dir / "images").glob("*.png"))
        if not image_files:
            return 1, 1
        src_w, src_h = Image.open(image_files[0]).size
        return int(src_h), int(src_w)

    def __len__(self) -> int:
        base = len(self.scenes) * 100 if self.cfg.training else len(self.scenes)
        return max(base, len(self.scenes))

    def _scene_for_index(self, index: int) -> _Scene:
        if self.cfg.training:
            sid = int(self.rng.integers(0, len(self.scenes)))
            return self.scenes[sid]
        return self.scenes[index % len(self.scenes)]

    def _clip_start(self, scene: _Scene, index: int) -> int:
        max_start = scene.frame_count - self.cfg.clip_frames
        if max_start <= 0:
            return 0
        if self.cfg.training:
            return int(self.rng.integers(0, max_start + 1))
        return int((index * self.cfg.clip_frames) % (max_start + 1))

    def _frame_ids(self, scene: _Scene, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng, scene_len=int(scene.frame_count), clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment, training=bool(self.cfg.training), index=index,
        )

    def _frame_paths(self, scene: _Scene, frame_idx: int) -> tuple[Path, Path, Path]:
        name = scene.name
        image_path = scene.path / "images" / f"{name}-{frame_idx:04d}.png"
        depth_path = scene.path / "depths" / f"{name}_{frame_idx:04d}.geometric.png"
        traj_path = scene.path / "trajectories" / f"{frame_idx:06d}.pth"
        return image_path, depth_path, traj_path

    def _frame_camera(self, scene: _Scene, frame_idx: int) -> _FrameCamera:
        key = (scene.name, frame_idx)
        cached = self.camera_cache.get(key)
        if cached is not None:
            return cached

        scene_cams = self.camera_index.get(scene.name, {})
        if frame_idx in scene_cams:
            cam = scene_cams[frame_idx]
        else:
            cam = _FrameCamera(
                k_src=np.full((3, 3), np.nan, dtype=np.float32),
                t_cw=np.full((4, 4), np.nan, dtype=np.float32),
                t_wc=np.full((4, 4), np.nan, dtype=np.float32),
                depth_scale_adjustment=1.0,
                src_h=scene.src_h,
                src_w=scene.src_w,
                valid=False,
            )
        self.camera_cache[key] = cam
        return cam

    def _sample_key(self, scene: _Scene, frame_ids: list[int]) -> str:
        frame_token = ",".join(str(int(fid)) for fid in frame_ids)
        return f"dynamic_replica_raw::{scene.name}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, frame_ids: list[int]) -> list[str]:
        out: list[str] = []
        for fid in frame_ids:
            img_path, depth_path, traj_path = self._frame_paths(scene, fid)
            out.extend([str(img_path), str(depth_path), str(traj_path)])
        return out

    def _build_sample(self, scene: _Scene, frame_ids: list[int], start: int) -> dict[str, Any]:
        videos: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        cameras: list[_FrameCamera] = []
        for fid in frame_ids:
            img_path, depth_path, _traj_path = self._frame_paths(scene, fid)
            cam = self._frame_camera(scene, fid)
            videos.append(_resize_rgb(img_path, self.w, self.h))
            depth_u16 = _resize_depth(depth_path, self.w, self.h)
            depth_raw = self._decode_depth(depth_u16)
            depth_scale = float(cam.depth_scale_adjustment)
            depths.append(depth_raw * depth_scale)
            cameras.append(cam)

        video = np.stack(videos, axis=0)
        video = np.transpose(video, (0, 3, 1, 2)).astype(np.float32) / 255.0
        depth = np.stack(depths, axis=0).astype(np.float32)
        depth_valid = np.isfinite(depth) & (depth > 0.0)

        t = self.cfg.clip_frames

        k_seq = np.full((t, 3, 3), np.nan, dtype=np.float32)
        t_wc_seq = np.full((t, 4, 4), np.nan, dtype=np.float32)
        camera_valid = np.zeros((t,), dtype=np.bool_)
        for i, cam in enumerate(cameras):
            if not cam.valid:
                continue
            k = cam.k_src.copy()
            sx = float(self.w) / float(max(1, cam.src_w))
            sy = float(self.h) / float(max(1, cam.src_h))
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy
            k_seq[i] = k
            t_wc_seq[i] = cam.t_wc
            camera_valid[i] = True

        aspect_ratio = np.array([scene.src_w / max(1.0, scene.src_h)], dtype=np.float32)
        _crop_info = {}
        if self.cfg.training:
            video = apply_photometric_augment(video, self.rng, self.augment)
            (video, depth, depth_valid, k_seq, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video,
                depth_t_hw=depth,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_seq,
                camera_valid_t=camera_valid,
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )

        # Load per-frame trajectory data for trajectory-based query building.
        # Fall back to depth reprojection if trajectory files are unavailable.
        traj_loaded = True
        traj_fallback_reason: str | None = None
        traj_3d_list: list[np.ndarray] = []
        traj_2d_list: list[np.ndarray] = []
        traj_vis_list: list[np.ndarray] = []
        for fid in frame_ids:
            _, _, traj_path = self._frame_paths(scene, fid)
            if not traj_path.exists():
                traj_loaded = False
                traj_fallback_reason = f"missing trajectory file {traj_path.name}"
                break
            try:
                td = _load_traj(traj_path)
                traj_2d_list.append(td["traj_2d"])
                traj_3d_list.append(td["traj_3d_world"])
                traj_vis_list.append(td["verts_inds_vis"])
            except Exception as exc:
                traj_loaded = False
                traj_fallback_reason = str(exc)
                break

        if traj_loaded and traj_3d_list:
            traj_2d = np.stack(traj_2d_list, axis=0)             # [T, N, 3]
            traj_3d_world = np.stack(traj_3d_list, axis=0)   # [T, N, 3]
            traj_visible = np.stack(traj_vis_list, axis=0)    # [T, N]
            traj_valid = np.isfinite(traj_3d_world).all(axis=-1) & traj_visible
            query, target, mask, query_stats = build_queries_from_trajectories(
                rng=self.rng,
                traj_3d_world=traj_3d_world,
                traj_visible=traj_visible,
                traj_valid=traj_valid,
                k_seq=k_seq,
                t_wc_seq=t_wc_seq,
                camera_valid=camera_valid,
                depth=depth,
                depth_valid=depth_valid,
                queries_per_clip=int(self.cfg.queries_per_clip),
                hard_query_ratio=float(self.cfg.hard_query_ratio),
                prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
                t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
                t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
            )
        else:
            self._warn_depth_fallback(
                scene,
                traj_fallback_reason or "trajectory data could not be loaded for the sampled clip",
            )
            query, target, mask, query_stats = build_queries_from_depth(
                rng=self.rng,
                depth=depth,
                depth_valid=depth_valid,
                k_seq=k_seq,
                t_wc_seq=t_wc_seq,
                camera_valid=camera_valid,
                queries_per_clip=int(self.cfg.queries_per_clip),
                hard_query_ratio=float(self.cfg.hard_query_ratio),
                prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
                t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
                t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
            )

        sample: dict[str, Any] = {}
        sample["video"] = torch.from_numpy(video)
        sample["aspect_ratio"] = torch.from_numpy(aspect_ratio.astype(np.float32))
        sample["depth_m"] = torch.from_numpy(depth)
        sample["depth_valid"] = torch.from_numpy(depth_valid)
        sample["camera"] = {
            "K": torch.from_numpy(k_seq).float(),
            "T_wc": torch.from_numpy(t_wc_seq).float(),
            "camera_valid": torch.from_numpy(camera_valid).bool(),
        }
        sample["augment_info"] = {
            k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()
        }
        sample["query"] = {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()}
        sample["query_stats"] = {k: torch.from_numpy(v).bool() for k, v in query_stats.items()}
        sample["target"] = {k: torch.from_numpy(v).float() for k, v in target.items()}
        sample["mask"] = {k: torch.from_numpy(v).bool() for k, v in mask.items()}
        sample["meta"] = {
            "dataset": "dynamic_replica_raw",
            "scene_id": scene.name,
            "clip_start": start,
            "source_mode": "dynamic_replica_raw",
            "camera_convention": str(self.cfg.camera_convention),
            "depth_decode_mode": str(self.cfg.depth_decode_mode),
            "depth_decode_mode_resolved": str(self._resolved_depth_decode_mode or self.cfg.depth_decode_mode),
            "depth_divisor": float(self.cfg.depth_divisor),
            "native_hw": (scene.src_h, scene.src_w),
        }
        if bool(self.cfg.benchmark_tracking_enabled) and traj_loaded and traj_3d_list:
            benchmark_tracking = self._build_benchmark_tracking(
                scene=scene,
                traj_2d=traj_2d,
                traj_3d_world=traj_3d_world,
                traj_visible=traj_visible,
                k_seq=k_seq,
                t_wc_seq=t_wc_seq,
                camera_valid=camera_valid,
                sample_key=f"{scene.name}::start={int(start)}",
            )
            if benchmark_tracking is not None:
                sample["benchmark_tracking"] = {
                    key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for key, value in benchmark_tracking.items()
                }
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene_for_index(query_index)
            frame_ids = self._frame_ids(scene, query_index)
            start = int(frame_ids[0]) if frame_ids else 0

            sample_key = self._sample_key(scene, frame_ids)
            sample_paths = self._sample_paths(scene, frame_ids)
            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue

            try:
                sample = self._build_sample(scene=scene, frame_ids=frame_ids, start=start)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="dynamic_replica_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"DynamicReplicaRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
