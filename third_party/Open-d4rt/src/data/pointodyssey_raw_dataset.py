"""PointOdyssey raw dataset adapter."""

from __future__ import annotations

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
from .depth_query_builder import build_queries_from_depth, build_queries_from_trajectories
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    sample_frame_indices_with_stride,
)
from .seeding import SeededDatasetMixin


def _read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_depth(path: Path, width: int, height: int) -> np.ndarray:
    try:
        dep = np.asarray(Image.open(path), dtype=np.uint16)
        dep_img = Image.fromarray(dep, mode="I;16").resize((width, height), resample=Image.Resampling.NEAREST)
        return np.asarray(dep_img, dtype=np.uint16)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read depth image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _world_to_cam(x_world: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    x_h = np.concatenate([x_world.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)
    x_cam = t_cw @ x_h
    return x_cam[:3]


def _is_nan_scalar_placeholder(arr: np.ndarray | None) -> bool:
    if arr is None or arr.ndim != 0 or arr.size != 1:
        return False
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    return bool(np.isnan(arr.item()))


@dataclass
class PointOdysseyRawConfig:
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
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64
    max_cached_scenes: int = 2
    val_clips_per_scene: int = 1


@dataclass
class _Scene:
    name: str
    path: Path
    rgb_paths: list[Path]
    depth_paths: list[Path]
    frame_count: int
    src_h: int
    src_w: int


@dataclass
class _SceneCache:
    traj_2d_tn2: np.ndarray
    traj_3d_tn3: np.ndarray | None
    traj_3d_fallback_reason: str | None
    valids_tn: np.ndarray
    visibs_tn: np.ndarray
    k_seq: np.ndarray
    t_cw_seq: np.ndarray
    t_wc_seq: np.ndarray
    frame_count: int
    n_points: int


class PointOdysseyRawDataset(SeededDatasetMixin, Dataset):
    """Loads PointOdyssey raw files and builds D4RT-compatible supervision."""

    def __init__(self, config: PointOdysseyRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="pointodyssey_raw", default_seed=20260320)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        self.max_cached_scenes = max(1, int(config.max_cached_scenes))
        if not config.training:
            self.augment = RawAugmentConfig()

        split_dir = config.root / self._split_dir_name()
        if not split_dir.exists():
            raise FileNotFoundError(f"PointOdyssey split dir not found: {split_dir}")

        self.scenes: list[_Scene] = []
        for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            anno_path = scene_dir / "anno.npz"
            if not anno_path.exists():
                continue
            rgb_paths = sorted((scene_dir / "rgbs").glob("*.jpg"))
            if not rgb_paths:
                rgb_paths = sorted((scene_dir / "rgbs").glob("*.png"))
            depth_paths = sorted((scene_dir / "depths").glob("*.png"))
            n = min(len(rgb_paths), len(depth_paths))
            if n < config.clip_frames:
                continue
            src_w, src_h = Image.open(rgb_paths[0]).size
            self.scenes.append(
                _Scene(
                    name=scene_dir.name,
                    path=scene_dir,
                    rgb_paths=rgb_paths[:n],
                    depth_paths=depth_paths[:n],
                    frame_count=n,
                    src_h=int(src_h),
                    src_w=int(src_w),
                )
            )
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break

        if not self.scenes:
            raise ValueError(f"No valid PointOdyssey scenes in {split_dir}")

        self.cache: dict[str, _SceneCache] = {}
        self.cache_order: list[str] = []

    def _split_dir_name(self) -> str:
        split = str(self.cfg.split).strip().lower()
        default_map = {"train": "train", "val": "test", "test": "test"}
        mapping = default_map
        if isinstance(self.cfg.split_map, dict):
            mapping = {k: str(v) for k, v in self.cfg.split_map.items()}
        return mapping.get(split, mapping.get("train", "train"))

    def _load_scene_cache(self, scene: _Scene) -> _SceneCache:
        cached = self.cache.get(scene.name)
        if cached is not None:
            if scene.name in self.cache_order:
                self.cache_order.remove(scene.name)
            self.cache_order.append(scene.name)
            return cached

        anno_path = scene.path / "anno.npz"
        try:
            anno = np.load(anno_path, allow_pickle=True)
        except Exception as exc:
            raise RetryableSampleError(f"Failed to read PointOdyssey anno: {anno_path}: {exc}", failed_paths=[str(anno_path)]) from exc

        traj_2d = np.asarray(anno.get("trajs_2d", np.empty((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        valids = np.asarray(anno.get("valids", np.empty((0, 0), dtype=np.bool_))).astype(np.bool_)
        visibs = np.asarray(anno.get("visibs", np.empty((0, 0), dtype=np.bool_))).astype(np.bool_)
        intr = np.asarray(anno.get("intrinsics", np.empty((0, 3, 3), dtype=np.float32)), dtype=np.float32)
        extr = np.asarray(anno.get("extrinsics", np.empty((0, 4, 4), dtype=np.float32)), dtype=np.float32)
        traj_3d_raw = anno.get("trajs_3d")
        traj_3d_fallback_reason: str | None = None
        traj_3d = None if traj_3d_raw is None else np.asarray(traj_3d_raw, dtype=np.float32)
        if traj_3d_raw is None:
            traj_3d_fallback_reason = "anno.npz is missing key 'trajs_3d'"
        elif _is_nan_scalar_placeholder(traj_3d):
            raise RetryableSampleError(
                f"PointOdyssey trajs_3d uses a NaN scalar placeholder in {anno_path}; "
                "blacklisting this scene instead of falling back to depth-based queries",
                failed_paths=[str(anno_path)],
            )

        if traj_2d.ndim != 3 or traj_2d.shape[-1] != 2:
            raise RetryableSampleError(f"Invalid PointOdyssey trajs_2d shape: {traj_2d.shape}", failed_paths=[str(anno_path)])
        if valids.shape != traj_2d.shape[:2] or visibs.shape != traj_2d.shape[:2]:
            raise RetryableSampleError(
                f"Invalid PointOdyssey valids/visibs shape: {valids.shape}, {visibs.shape} vs {traj_2d.shape[:2]}",
                failed_paths=[str(anno_path)],
            )
        if intr.ndim != 3 or intr.shape[1:] != (3, 3):
            raise RetryableSampleError(f"Invalid PointOdyssey intrinsics shape: {intr.shape}", failed_paths=[str(anno_path)])
        if extr.ndim != 3 or extr.shape[1:] != (4, 4):
            raise RetryableSampleError(f"Invalid PointOdyssey extrinsics shape: {extr.shape}", failed_paths=[str(anno_path)])

        n_frames = min(scene.frame_count, traj_2d.shape[0], valids.shape[0], visibs.shape[0], intr.shape[0], extr.shape[0])
        n_points = int(traj_2d.shape[1])
        if traj_3d is not None and traj_3d.ndim == 3 and traj_3d.shape[-1] == 3:
            n_frames = min(n_frames, traj_3d.shape[0])
            n_points = min(n_points, int(traj_3d.shape[1]))
        elif traj_3d is not None:
            traj_3d_fallback_reason = f"invalid trajs_3d shape {traj_3d.shape}"
            traj_3d = None
        if n_frames < self.cfg.clip_frames or n_points <= 0:
            raise RetryableSampleError(
                f"PointOdyssey scene too short/empty after alignment: {scene.name}: frames={n_frames}, points={n_points}",
                failed_paths=[str(anno_path)],
            )

        traj_2d = traj_2d[:n_frames, :n_points, :].astype(np.float32)
        valids = valids[:n_frames, :n_points].astype(np.bool_)
        visibs = visibs[:n_frames, :n_points].astype(np.bool_)
        intr = intr[:n_frames].astype(np.float32)
        t_cw_seq = extr[:n_frames].astype(np.float32)
        t_wc_seq = np.full_like(t_cw_seq, np.nan, dtype=np.float32)
        for i in range(n_frames):
            try:
                t_wc_seq[i] = np.linalg.inv(t_cw_seq[i]).astype(np.float32)
            except np.linalg.LinAlgError:
                continue

        if traj_3d is not None and traj_3d.ndim == 3 and traj_3d.shape[-1] == 3:
            traj_3d = traj_3d[:n_frames, :n_points, :].astype(np.float32)
        else:
            traj_3d = None

        out = _SceneCache(
            traj_2d_tn2=traj_2d,
            traj_3d_tn3=traj_3d,
            traj_3d_fallback_reason=traj_3d_fallback_reason,
            valids_tn=valids,
            visibs_tn=visibs,
            k_seq=intr,
            t_cw_seq=t_cw_seq,
            t_wc_seq=t_wc_seq,
            frame_count=n_frames,
            n_points=n_points,
        )

        self.cache[scene.name] = out
        if scene.name in self.cache_order:
            self.cache_order.remove(scene.name)
        self.cache_order.append(scene.name)
        while len(self.cache_order) > self.max_cached_scenes:
            old = self.cache_order.pop(0)
            self.cache.pop(old, None)
        return out

    def __len__(self) -> int:
        if self.cfg.training:
            base = len(self.scenes) * 40
        else:
            base = len(self.scenes) * max(1, int(self.cfg.val_clips_per_scene))
        return max(base, len(self.scenes))

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            sid = int(self.rng.integers(0, len(self.scenes)))
            return self.scenes[sid]
        return self.scenes[index % len(self.scenes)]

    def _eval_clip_slot(self, index: int) -> int:
        if self.cfg.training:
            return int(index)
        clips_per_scene = max(1, int(self.cfg.val_clips_per_scene))
        return int((index // len(self.scenes)) % clips_per_scene)

    def _frame_indices(self, scene_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng, scene_len=scene_len, clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment, training=bool(self.cfg.training), index=index,
        )

    def _sample_key(self, scene: _Scene, idxs: list[int]) -> str:
        token = ",".join(str(int(v)) for v in idxs)
        return f"pointodyssey_raw::{scene.name}::frames={token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out = [str(scene.path / "anno.npz")]
        for i in idxs:
            out.append(str(scene.rgb_paths[i]))
            out.append(str(scene.depth_paths[i]))
        return out

    def _build_sample(self, scene: _Scene, cache: _SceneCache, idxs: list[int], clip_start: int) -> dict[str, Any]:
        video_list: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        for i in idxs:
            video_list.append(_read_rgb(scene.rgb_paths[i], width=self.w, height=self.h))
            dep_u16 = _read_depth(scene.depth_paths[i], width=self.w, height=self.h)
            depth_list.append(dep_u16.astype(np.float32) / 65535.0 * 1000.0)

        video = np.stack(video_list, axis=0).astype(np.float32) / 255.0
        video = np.transpose(video, (0, 3, 1, 2))
        depth = np.stack(depth_list, axis=0).astype(np.float32)
        depth_valid = np.isfinite(depth) & (depth > 0.0)

        k_arr = cache.k_seq[idxs].copy()
        t_wc_arr = cache.t_wc_seq[idxs].copy()
        sx = float(self.w) / max(float(scene.src_w), 1.0)
        sy = float(self.h) / max(float(scene.src_h), 1.0)
        k_arr[:, 0, 0] *= sx
        k_arr[:, 0, 2] *= sx
        k_arr[:, 1, 1] *= sy
        k_arr[:, 1, 2] *= sy
        cam_valid = np.isfinite(k_arr).all(axis=(1, 2)) & np.isfinite(t_wc_arr).all(axis=(1, 2))

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

        # Use GT trajectories for query building when available (dynamic scenes).
        # Scenes whose trajs_3d is a NaN placeholder are blacklisted earlier in _load_scene_cache.
        # Remaining non-trajectory cases still fall back to depth reprojection.
        if cache.traj_3d_tn3 is not None:
            traj_3d = cache.traj_3d_tn3[idxs]       # [T_clip, N, 3]
            traj_vis = cache.visibs_tn[idxs]         # [T_clip, N]
            traj_val = cache.valids_tn[idxs]         # [T_clip, N]
            query, target, mask, query_stats = build_queries_from_trajectories(
                rng=self.rng,
                traj_3d_world=traj_3d,
                traj_visible=traj_vis,
                traj_valid=traj_val,
                k_seq=k_arr,
                t_wc_seq=t_wc_arr,
                camera_valid=cam_valid,
                depth=depth,
                depth_valid=depth_valid,
                queries_per_clip=int(self.cfg.queries_per_clip),
                hard_query_ratio=float(self.cfg.hard_query_ratio),
                prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
                t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
                t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
            )
        else:
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
                "dataset": "pointodyssey_raw",
                "scene_id": scene.name,
                "clip_start": int(clip_start),
                "source_mode": "pointodyssey_tracks_world",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            frame_index = query_index if self.cfg.training else self._eval_clip_slot(query_index)
            idxs = self._frame_indices(scene.frame_count, frame_index)
            clip_start = int(idxs[0]) if idxs else 0
            sample_key = self._sample_key(scene, idxs)
            sample_paths = self._sample_paths(scene, idxs)

            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue

            try:
                cache = self._load_scene_cache(scene)
                if cache.frame_count < self.cfg.clip_frames:
                    raise RetryableSampleError(f"PointOdyssey scene too short: {scene.name}", failed_paths=[str(scene.path)])
                idxs = self._frame_indices(cache.frame_count, frame_index)
                clip_start = int(idxs[0]) if idxs else 0
                sample = self._build_sample(scene=scene, cache=cache, idxs=idxs, clip_start=clip_start)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="pointodyssey_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"PointOdysseyRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
