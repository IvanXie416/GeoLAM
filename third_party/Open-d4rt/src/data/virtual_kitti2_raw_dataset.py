"""Virtual KITTI 2 raw dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from .bad_sample_registry import (
    BadSampleRegistry,
    RetryableSampleError,
    failed_paths_from_exception,
    is_retryable_data_error,
)
from .depth_query_builder import _compute_normal_map
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    depth_boundary_mask,
    sample_frame_indices_with_stride,
    sample_hard_query_flags,
)
from .seeding import SeededDatasetMixin


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


def _read_scene_flow_png(path: Path) -> np.ndarray | None:
    """Decode VKITTI2 forward scene flow from 16-bit RGB PNG to float32 [H,W,3] in meters.

    Encoding: dx = ((R * 2.0 / 65535.0) - 1.0) * 10.0  (range [-10, 10] m).
    Coordinate system: next-frame camera space (x-right, y-down, z-forward).
    """
    try:
        # PIL silently downcasts these RGB PNGs to 8-bit, which turns near-zero
        # flow values into a fake constant ~-10 m field after decoding. OpenCV
        # preserves the native uint16 payload; convert BGR -> RGB before decode.
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img.ndim != 3 or img.shape[2] < 3:
            return None
        img = img[:, :, ::-1].astype(np.float64, copy=False)
        sf = ((img[:, :, :3] * 2.0 / 65535.0) - 1.0) * 10.0
        return sf.astype(np.float32)
    except Exception:
        return None


def _build_vkitti2_queries_from_trajectories(
    rng: np.random.Generator,
    traj_3d_world: np.ndarray,
    traj_visible: np.ndarray,
    traj_valid: np.ndarray,
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    camera_valid: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    queries_per_clip: int,
    hard_query_ratio: float,
    prob_t_tgt_equals_t_cam: float,
    t_src_tgt_delta_choices: Sequence[int | None] | None = None,
    t_src_tgt_delta_probs: Sequence[float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """VKITTI2-specific trajectory builder with pid-aware target-time sampling.

    The shared builder samples ``t_tgt`` uniformly before choosing a source pid.
    VKITTI2 trajectories here are only partially observed forward tracks, so a
    global-uniform ``t_tgt`` wastes most queries. This builder instead samples:

    1. source frame ``t_src``
    2. source pid at that frame
    3. ``t_tgt`` uniformly from the valid time indices of that pid
    4. ``t_cam`` with the usual ``P(t_tgt == t_cam)`` bias
    """

    t_clip, n_pts, _ = traj_3d_world.shape
    m = int(queries_per_clip)
    h, w = depth.shape[1], depth.shape[2]

    q_u = np.zeros((m,), dtype=np.float32)
    q_v = np.zeros((m,), dtype=np.float32)
    q_t_src = np.zeros((m,), dtype=np.int64)
    q_t_tgt = np.zeros((m,), dtype=np.int64)
    q_t_cam = np.zeros((m,), dtype=np.int64)

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
    for i in range(t_clip):
        if not bool(camera_valid[i]):
            continue
        try:
            t_cw_seq[i] = np.linalg.inv(t_wc_seq[i]).astype(np.float32)
        except np.linalg.LinAlgError:
            continue

    eligible_ids: list[np.ndarray] = []
    eligible_uv: list[np.ndarray] = []
    source_frames: list[int] = []
    for fi in range(t_clip):
        if not bool(camera_valid[fi]):
            eligible_ids.append(np.zeros((0,), dtype=np.int64))
            eligible_uv.append(np.zeros((0, 2), dtype=np.float32))
            continue
        base_ok = (
            traj_valid[fi].astype(bool)
            & traj_visible[fi].astype(bool)
            & np.isfinite(traj_3d_world[fi]).all(axis=-1)
        )
        eids = np.where(base_ok)[0].astype(np.int64)
        if eids.size == 0:
            eligible_ids.append(eids)
            eligible_uv.append(np.zeros((0, 2), dtype=np.float32))
            continue
        pts_w = traj_3d_world[fi, eids]
        pts_h = np.concatenate([pts_w, np.ones((len(eids), 1), dtype=np.float32)], axis=1)
        pts_c = (t_cw_seq[fi] @ pts_h.T).T
        z_c = pts_c[:, 2]
        good = np.isfinite(z_c) & (z_c > 1e-6)
        proj = (k_seq[fi] @ pts_c[:, :3].T).T
        u_px = np.where(good, proj[:, 0] / np.maximum(z_c, 1e-8), -1.0)
        v_px = np.where(good, proj[:, 1] / np.maximum(z_c, 1e-8), -1.0)
        in_img = good & (u_px >= 0.0) & (u_px <= (w - 1)) & (v_px >= 0.0) & (v_px <= (h - 1))
        eids = eids[in_img]
        uvs = np.stack([u_px[in_img], v_px[in_img]], axis=-1).astype(np.float32)
        eligible_ids.append(eids)
        eligible_uv.append(uvs)
        if eids.size > 0:
            source_frames.append(fi)

    hard_slots: list[np.ndarray] = []
    for fi in range(t_clip):
        if eligible_ids[fi].size == 0 or not bool(camera_valid[fi]):
            hard_slots.append(np.zeros((0,), dtype=np.int64))
            continue
        bmask = depth_boundary_mask(depth[fi], depth_valid[fi], q=0.9)
        if not bmask.any():
            hard_slots.append(np.zeros((0,), dtype=np.int64))
            continue
        src_uv = eligible_uv[fi]
        u_int = np.clip(np.rint(src_uv[:, 0]).astype(np.int64), 0, w - 1)
        v_int = np.clip(np.rint(src_uv[:, 1]).astype(np.int64), 0, h - 1)
        on_boundary = bmask[v_int, u_int]
        hard_slots.append(np.where(on_boundary)[0].astype(np.int64))

    hard_target = int(sample_hard_query_flags(rng, m, float(hard_query_ratio)).sum())
    eq_flags = sample_hard_query_flags(rng, m, float(prob_t_tgt_equals_t_cam))
    use_hard = np.zeros((m,), dtype=np.bool_)
    hard_eligible_q = np.array([i for i in range(m) if source_frames], dtype=np.int64)
    if hard_target > 0 and hard_eligible_q.size > 0:
        picked = rng.choice(hard_eligible_q, size=min(hard_target, hard_eligible_q.size), replace=False)
        use_hard[picked.astype(np.int64)] = True

    normal_maps: list[np.ndarray] = []
    normal_valid_maps: list[np.ndarray] = []
    for fi in range(t_clip):
        if bool(camera_valid[fi]) and np.isfinite(k_seq[fi]).all():
            n_map, n_valid = _compute_normal_map(depth[fi], k_seq[fi], depth_valid[fi])
        else:
            n_map = np.zeros((h, w, 3), dtype=np.float32)
            n_valid = np.zeros((h, w), dtype=bool)
        normal_maps.append(n_map)
        normal_valid_maps.append(n_valid)

    valid_cam_frames = np.flatnonzero(camera_valid.astype(bool)).astype(np.int64)
    source_frames_arr = np.asarray(source_frames, dtype=np.int64)
    if source_frames_arr.size == 0 or valid_cam_frames.size == 0:
        query = {"u": q_u, "v": q_v, "t_src": q_t_src, "t_tgt": q_t_tgt, "t_cam": q_t_cam}
        query_stats = {"is_hard_query": is_hard_query}
        target = {"xyz_3d": y_xyz, "uv_2d": y_uv, "visibility": y_vis, "displacement": y_disp, "normal": y_normal}
        mask = {"xyz_3d": m_xyz, "uv_2d": m_uv, "visibility": m_vis, "displacement": m_disp, "normal": m_normal}
        return query, target, mask, query_stats

    w_norm = max(1.0, float(w - 1))
    h_norm = max(1.0, float(h - 1))

    def _sample_valid_target_time(fs: int, valid_tgt: np.ndarray) -> int:
        if t_src_tgt_delta_choices is None or t_src_tgt_delta_probs is None:
            return int(valid_tgt[int(rng.integers(0, valid_tgt.size))])

        choices = tuple(t_src_tgt_delta_choices)
        probs = np.asarray(t_src_tgt_delta_probs, dtype=np.float64)
        if len(choices) == 0:
            raise ValueError("t_src_tgt_delta_choices must be non-empty when q_t_src is provided")
        if probs.shape != (len(choices),):
            raise ValueError(
                "t_src_tgt_delta_probs must have the same length as t_src_tgt_delta_choices, "
                f"got {probs.shape[0]} vs {len(choices)}"
            )
        if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
            raise ValueError("t_src_tgt_delta_probs must be finite and sum to a positive value")

        probs = probs / float(probs.sum())
        max_delta_raw = choices[int(rng.choice(len(choices), p=probs))]
        candidates = valid_tgt
        if max_delta_raw is not None:
            max_delta = int(max_delta_raw)
            if 0 <= max_delta < t_clip - 1:
                local = valid_tgt[np.abs(valid_tgt - int(fs)) <= max_delta]
                if local.size > 0:
                    candidates = local
        return int(candidates[int(rng.integers(0, candidates.size))])

    for i in range(m):
        fs = int(source_frames_arr[int(rng.integers(0, source_frames_arr.size))])
        q_t_src[i] = fs

        if bool(use_hard[i]) and hard_slots[fs].size > 0:
            picked_slot = int(hard_slots[fs][int(rng.integers(0, hard_slots[fs].size))])
        else:
            if eligible_ids[fs].size == 0:
                continue
            picked_slot = int(rng.integers(0, eligible_ids[fs].size))
        if eligible_ids[fs].size == 0:
            continue

        pid = int(eligible_ids[fs][picked_slot])
        u_src = float(eligible_uv[fs][picked_slot, 0])
        v_src = float(eligible_uv[fs][picked_slot, 1])
        is_hard_query[i] = bool(use_hard[i]) and hard_slots[fs].size > 0

        valid_tgt = np.flatnonzero(
            camera_valid.astype(bool)
            & traj_valid[:, pid].astype(bool)
            & np.isfinite(traj_3d_world[:, pid]).all(axis=-1)
        ).astype(np.int64)
        if valid_tgt.size == 0:
            continue
        ft = _sample_valid_target_time(fs, valid_tgt)
        q_t_tgt[i] = ft

        if bool(eq_flags[i]) or valid_cam_frames.size <= 1:
            fc = ft
        else:
            cam_choices = valid_cam_frames[valid_cam_frames != ft]
            fc = int(cam_choices[int(rng.integers(0, cam_choices.size))]) if cam_choices.size > 0 else ft
        q_t_cam[i] = fc

        p_world_src = traj_3d_world[fs, pid]
        if not np.isfinite(p_world_src).all():
            continue
        p_world_tgt = traj_3d_world[ft, pid]
        if not np.isfinite(p_world_tgt).all():
            continue

        q_u[i] = u_src / w_norm
        q_v[i] = v_src / h_norm

        p_tgt_h = np.array([*p_world_tgt, 1.0], dtype=np.float32)
        xyz_cam = (t_cw_seq[fc] @ p_tgt_h)[:3]
        if not np.isfinite(xyz_cam).all():
            continue
        y_xyz[i] = xyz_cam.astype(np.float32)
        m_xyz[i] = True

        delta_world = p_world_tgt - p_world_src
        r_cw_fc = t_cw_seq[fc, :3, :3]
        disp_cam = r_cw_fc @ delta_world
        if np.isfinite(disp_cam).all():
            y_disp[i] = disp_cam.astype(np.float32)
            m_disp[i] = True

        p_cam_tgt = (t_cw_seq[ft] @ p_tgt_h)[:3]
        z_tgt = float(p_cam_tgt[2])
        target_in_frame = False
        u_tgt = 0.0
        v_tgt = 0.0
        if np.isfinite(z_tgt) and z_tgt > 1e-6:
            proj_tgt = k_seq[ft] @ p_cam_tgt
            u_tgt = float(proj_tgt[0] / z_tgt)
            v_tgt = float(proj_tgt[1] / z_tgt)
            target_in_frame = (
                np.isfinite(u_tgt)
                and np.isfinite(v_tgt)
                and 0.0 <= u_tgt <= (w - 1)
                and 0.0 <= v_tgt <= (h - 1)
            )

        target_visible = (
            bool(traj_valid[ft, pid])
            and bool(traj_visible[ft, pid])
            and target_in_frame
        )
        if bool(traj_valid[ft, pid]):
            y_vis[i] = 1.0 if target_visible else 0.0
            m_vis[i] = True

        if target_visible:
            y_uv[i, 0] = np.clip(u_tgt / w_norm, 0.0, 1.0)
            y_uv[i, 1] = np.clip(v_tgt / h_norm, 0.0, 1.0)
            m_uv[i] = True

        u_src_int = int(np.clip(round(u_src), 0, w - 1))
        v_src_int = int(np.clip(round(v_src), 0, h - 1))
        if bool(normal_valid_maps[fs][v_src_int, u_src_int]):
            n_src = normal_maps[fs][v_src_int, u_src_int]
            r_wc_fs = t_wc_seq[fs, :3, :3]
            n_cam = r_cw_fc @ (r_wc_fs @ n_src)
            n_len = float(np.linalg.norm(n_cam))
            if np.isfinite(n_cam).all() and n_len > 1e-6:
                y_normal[i] = (n_cam / n_len).astype(np.float32)
                m_normal[i] = True

    query = {"u": q_u, "v": q_v, "t_src": q_t_src, "t_tgt": q_t_tgt, "t_cam": q_t_cam}
    query_stats = {"is_hard_query": is_hard_query}
    target = {"xyz_3d": y_xyz, "uv_2d": y_uv, "visibility": y_vis, "displacement": y_disp, "normal": y_normal}
    mask = {"xyz_3d": m_xyz, "uv_2d": m_uv, "visibility": m_vis, "displacement": m_disp, "normal": m_normal}
    return query, target, mask, query_stats


def _frame_id_from_filename(path: Path) -> int | None:
    stem = path.stem
    tail = stem.split("_")[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def _parse_intrinsics(path: Path) -> dict[tuple[int, int], np.ndarray]:
    if not path.exists():
        return {}
    out: dict[tuple[int, int], np.ndarray] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            frame = int(parts[0])
            cam = int(parts[1])
            fx = float(parts[2])
            fy = float(parts[3])
            cx = float(parts[4])
            cy = float(parts[5])
        except ValueError:
            continue
        out[(frame, cam)] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return out


def _parse_extrinsics(path: Path) -> dict[tuple[int, int], np.ndarray]:
    if not path.exists():
        return {}
    out: dict[tuple[int, int], np.ndarray] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            frame = int(parts[0])
            cam = int(parts[1])
            vals = [float(v) for v in parts[2:18]]
        except ValueError:
            continue
        mat = np.asarray(vals, dtype=np.float32).reshape(4, 4)
        out[(frame, cam)] = mat
    return out


@dataclass
class VirtualKitti2RawConfig:
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
    split_scenes: dict[str, list[str]] | None = None
    variants: list[str] | None = None
    camera_ids: list[int] | None = None
    max_scenes: int | None = None
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _Scene:
    scene_id: str
    variant_dir: Path
    camera_id: int
    frame_ids: np.ndarray
    rgb_paths: list[Path]
    depth_paths: list[Path]
    scene_flow_paths: list[Path | None]  # forward scene flow; None for last frame
    k_src_seq: np.ndarray
    t_wc_seq: np.ndarray
    frame_count: int
    src_h: int
    src_w: int


class VirtualKitti2RawDataset(SeededDatasetMixin, Dataset):
    """Loads Virtual KITTI 2 RGBD + calibration and builds depth-projected supervision."""

    def __init__(self, config: VirtualKitti2RawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="virtual_kitti2_raw", default_seed=20260320)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        if not config.training:
            self.augment = RawAugmentConfig()

        if not config.root.exists():
            raise FileNotFoundError(f"Virtual KITTI 2 root not found: {config.root}")

        selected_scenes = self._selected_scene_names()
        selected_variants = set(config.variants or [])
        camera_ids = config.camera_ids or [0]

        self.scenes: list[_Scene] = []
        for scene_name in selected_scenes:
            scene_dir = config.root / scene_name
            if not scene_dir.exists():
                continue
            variant_dirs = sorted([p for p in scene_dir.iterdir() if p.is_dir()])
            for variant_dir in variant_dirs:
                if selected_variants and variant_dir.name not in selected_variants:
                    continue
                intr = _parse_intrinsics(variant_dir / "intrinsic.txt")
                extr = _parse_extrinsics(variant_dir / "extrinsic.txt")
                if not intr or not extr:
                    continue
                for cam_id in camera_ids:
                    rgb_dir = variant_dir / "frames" / "rgb" / f"Camera_{int(cam_id)}"
                    depth_dir = variant_dir / "frames" / "depth" / f"Camera_{int(cam_id)}"
                    if not (rgb_dir.exists() and depth_dir.exists()):
                        continue

                    rgb_by_fid: dict[int, Path] = {}
                    depth_by_fid: dict[int, Path] = {}
                    for p in sorted(rgb_dir.glob("*.jpg")):
                        fid = _frame_id_from_filename(p)
                        if fid is not None:
                            rgb_by_fid[fid] = p
                    for p in sorted(depth_dir.glob("*.png")):
                        fid = _frame_id_from_filename(p)
                        if fid is not None:
                            depth_by_fid[fid] = p

                    common = sorted(
                        [
                            fid
                            for fid in set(rgb_by_fid.keys()).intersection(depth_by_fid.keys())
                            if (fid, int(cam_id)) in intr and (fid, int(cam_id)) in extr
                        ]
                    )
                    if len(common) < config.clip_frames:
                        continue

                    rgb_paths = [rgb_by_fid[fid] for fid in common]
                    depth_paths = [depth_by_fid[fid] for fid in common]
                    k_seq = np.stack([intr[(fid, int(cam_id))] for fid in common], axis=0).astype(np.float32)

                    t_wc_seq = np.full((len(common), 4, 4), np.nan, dtype=np.float32)
                    for i, fid in enumerate(common):
                        t_cw = extr[(fid, int(cam_id))]
                        try:
                            t_wc_seq[i] = np.linalg.inv(t_cw).astype(np.float32)
                        except np.linalg.LinAlgError:
                            continue

                    # Collect forward scene flow paths (frame t → t+1).
                    sf_dir = variant_dir / "frames" / "forwardSceneFlow" / f"Camera_{int(cam_id)}"
                    sf_paths: list[Path | None] = []
                    for fid in common:
                        sf_path = sf_dir / f"sceneFlow_{fid:05d}.png"
                        sf_paths.append(sf_path if sf_path.exists() else None)

                    src_w, src_h = Image.open(rgb_paths[0]).size
                    self.scenes.append(
                        _Scene(
                            scene_id=f"{scene_name}/{variant_dir.name}/Camera_{int(cam_id)}",
                            variant_dir=variant_dir,
                            camera_id=int(cam_id),
                            frame_ids=np.asarray(common, dtype=np.int64),
                            rgb_paths=rgb_paths,
                            depth_paths=depth_paths,
                            scene_flow_paths=sf_paths,
                            k_src_seq=k_seq,
                            t_wc_seq=t_wc_seq,
                            frame_count=len(common),
                            src_h=int(src_h),
                            src_w=int(src_w),
                        )
                    )
                    if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                        break
                if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                    break
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break

        if not self.scenes:
            raise ValueError(f"No valid Virtual KITTI 2 scenes found for split={config.split} under {config.root}")

    def _selected_scene_names(self) -> list[str]:
        split = str(self.cfg.split).strip().lower()
        default_split_scenes = {
            "train": ["Scene01", "Scene02", "Scene06"],
            "val": ["Scene18"],
            "test": ["Scene20"],
        }
        mapping = default_split_scenes
        if isinstance(self.cfg.split_scenes, dict):
            normalized: dict[str, list[str]] = {}
            for key, value in self.cfg.split_scenes.items():
                if isinstance(value, (list, tuple)):
                    normalized[str(key)] = [str(v) for v in value]
            if normalized:
                mapping = normalized
        scenes = mapping.get(split)
        if scenes:
            return scenes
        return sorted([p.name for p in self.cfg.root.iterdir() if p.is_dir() and p.name.lower().startswith("scene")])

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
        frame_token = ",".join(str(int(scene.frame_ids[i])) for i in idxs)
        return f"virtual_kitti2_raw::{scene.scene_id}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out = [
            str(scene.variant_dir / "intrinsic.txt"),
            str(scene.variant_dir / "extrinsic.txt"),
        ]
        for i in idxs:
            out.append(str(scene.rgb_paths[i]))
            out.append(str(scene.depth_paths[i]))
        return out

    def _build_sample(self, scene: _Scene, idxs: list[int], clip_start: int) -> dict[str, Any]:
        video_list: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        depth_valid_list: list[np.ndarray] = []
        k_seq: list[np.ndarray] = []
        t_wc_seq: list[np.ndarray] = []
        camera_valid: list[bool] = []

        for i in idxs:
            rgb = _read_rgb(scene.rgb_paths[i], width=self.w, height=self.h)
            depth_u16 = _read_depth_png(scene.depth_paths[i], width=self.w, height=self.h)
            depth_m = depth_u16.astype(np.float32) / 100.0
            valid = np.isfinite(depth_m) & (depth_u16 > 0) & (depth_u16 < 65535)

            k = scene.k_src_seq[i].copy()
            sx = float(self.w) / max(float(scene.src_w), 1.0)
            sy = float(self.h) / max(float(scene.src_h), 1.0)
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy

            video_list.append(rgb)
            depth_list.append(depth_m.astype(np.float32))
            depth_valid_list.append(valid.astype(np.bool_))
            t_wc_seq.append(scene.t_wc_seq[i].astype(np.float32))
            k_seq.append(k.astype(np.float32))
            camera_valid.append(bool(np.isfinite(k).all() and np.isfinite(scene.t_wc_seq[i]).all()))

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

        # Build sparse world-coordinate trajectories by anchoring points at each
        # frame and propagating them forward with consecutive-frame scene flow.
        # Each pid therefore denotes one physical point trajectory, not a reused
        # image slot across time.
        t_clip = len(idxs)
        h_img, w_img = depth.shape[1], depth.shape[2]
        n_sample_pts = min(2000, max(int(t_clip), h_img * w_img))  # sparse sampling for efficiency

        # Pre-compute T_cw and K_inv for crop-adjusted intrinsics.
        t_cw_arr = np.full((t_clip, 4, 4), np.nan, dtype=np.float32)
        k_inv_arr = np.full((t_clip, 3, 3), np.nan, dtype=np.float32)
        for fi in range(t_clip):
            if not cam_valid[fi]:
                continue
            try:
                t_cw_arr[fi] = np.linalg.inv(t_wc_arr[fi])
                k_inv_arr[fi] = np.linalg.inv(k_arr[fi])
            except np.linalg.LinAlgError:
                pass

        def _project_points(frame_idx: int, points_world: np.ndarray, k_use: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            points_h = np.concatenate(
                [points_world.astype(np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
                axis=1,
            )
            points_cam = (t_cw_arr[frame_idx] @ points_h.T).T[:, :3]
            z_cam = points_cam[:, 2]
            good = np.isfinite(points_cam).all(axis=-1) & np.isfinite(z_cam) & (z_cam > 1e-6)
            proj = (k_use @ points_cam.T).T
            u_px = np.where(good, proj[:, 0] / np.maximum(z_cam, 1e-8), -1.0)
            v_px = np.where(good, proj[:, 1] / np.maximum(z_cam, 1e-8), -1.0)
            return points_cam, u_px.astype(np.float32), v_px.astype(np.float32)

        def _visibility_from_depth(frame_idx: int, points_world: np.ndarray) -> np.ndarray:
            if points_world.size == 0 or not bool(cam_valid[frame_idx]):
                return np.zeros((len(points_world),), dtype=bool)
            points_cam, u_px, v_px = _project_points(frame_idx, points_world, k_arr[frame_idx])
            z_cam = points_cam[:, 2]
            in_img = (
                np.isfinite(u_px)
                & np.isfinite(v_px)
                & np.isfinite(z_cam)
                & (z_cam > 1e-6)
                & (u_px >= 0.0)
                & (u_px <= (w_img - 1))
                & (v_px >= 0.0)
                & (v_px <= (h_img - 1))
            )
            if not in_img.any():
                return np.zeros((len(points_world),), dtype=bool)
            u_int = np.clip(np.rint(u_px).astype(np.int64), 0, w_img - 1)
            v_int = np.clip(np.rint(v_px).astype(np.int64), 0, h_img - 1)
            z_ref = depth[frame_idx, v_int, u_int]
            depth_ok = depth_valid[frame_idx, v_int, u_int] & np.isfinite(z_ref) & (z_ref > 0.0)
            thresh = np.maximum(0.05, 0.02 * np.maximum(z_ref, 0.0))
            consistent = depth_ok & (np.abs(z_cam - z_ref) <= thresh)
            return in_img & consistent

        pts_per_frame = np.full((t_clip,), n_sample_pts // max(t_clip, 1), dtype=np.int64)
        pts_per_frame[: n_sample_pts % max(t_clip, 1)] += 1

        traj_3d_world = np.full((t_clip, n_sample_pts, 3), np.nan, dtype=np.float32)
        traj_valid = np.zeros((t_clip, n_sample_pts), dtype=bool)
        next_pid = 0

        for fi in range(t_clip):
            n_anchor = int(pts_per_frame[fi])
            if n_anchor <= 0 or not bool(cam_valid[fi]) or not np.isfinite(k_inv_arr[fi]).all():
                continue
            valid_pixels = np.column_stack(np.where(depth_valid[fi] & np.isfinite(depth[fi]) & (depth[fi] > 0.0)))
            if valid_pixels.shape[0] == 0:
                continue
            take = min(n_anchor, int(valid_pixels.shape[0]))
            choice = self.rng.choice(valid_pixels.shape[0], size=take, replace=False)
            picked = valid_pixels[choice]
            v_anchor = picked[:, 0].astype(np.float32)
            u_anchor = picked[:, 1].astype(np.float32)
            z_anchor = depth[fi, picked[:, 0], picked[:, 1]].astype(np.float32)
            pix = np.stack([u_anchor, v_anchor, np.ones((take,), dtype=np.float32)], axis=-1)
            p_cam = (k_inv_arr[fi] @ pix.T).T * z_anchor[:, None]
            p_cam_h = np.concatenate([p_cam, np.ones((take, 1), dtype=np.float32)], axis=1)
            p_world = (t_wc_arr[fi] @ p_cam_h.T).T[:, :3]
            keep = np.isfinite(p_world).all(axis=-1)
            if not keep.any():
                continue
            count = int(keep.sum())
            pid_slice = slice(next_pid, next_pid + count)
            traj_3d_world[fi, pid_slice] = p_world[keep].astype(np.float32)
            traj_valid[fi, pid_slice] = True
            next_pid += count

        traj_3d_world = traj_3d_world[:, :next_pid]
        traj_valid = traj_valid[:, :next_pid]
        traj_visible = np.zeros((t_clip, next_pid), dtype=bool)

        for fi in range(t_clip):
            live_ids = np.where(traj_valid[fi])[0]
            if live_ids.size == 0:
                continue
            traj_visible[fi, live_ids] = _visibility_from_depth(fi, traj_3d_world[fi, live_ids])

        for fi in range(t_clip - 1):
            sf_path = scene.scene_flow_paths[idxs[fi]] if idxs[fi] < len(scene.scene_flow_paths) else None
            if sf_path is None or not bool(cam_valid[fi]) or not bool(cam_valid[fi + 1]):
                continue
            sf = _read_scene_flow_png(sf_path)
            if sf is None:
                continue
            live_ids = np.where(traj_valid[fi] & traj_visible[fi])[0]
            if live_ids.size == 0:
                continue
            points_world = traj_3d_world[fi, live_ids]
            points_cam, u_orig, v_orig = _project_points(fi, points_world, scene.k_src_seq[idxs[fi]])
            z_cam = points_cam[:, 2]
            sf_h, sf_w = sf.shape[:2]
            in_bounds = (
                np.isfinite(u_orig)
                & np.isfinite(v_orig)
                & np.isfinite(z_cam)
                & (z_cam > 1e-6)
                & (u_orig >= 0.0)
                & (u_orig < sf_w)
                & (v_orig >= 0.0)
                & (v_orig < sf_h)
            )
            if not in_bounds.any():
                continue
            u_int = np.clip(np.rint(u_orig).astype(np.int64), 0, sf_w - 1)
            v_int = np.clip(np.rint(v_orig).astype(np.int64), 0, sf_h - 1)
            sf_vec = sf[v_int, u_int]
            step_ok = in_bounds & np.isfinite(sf_vec).all(axis=-1)
            if not step_ok.any():
                continue
            step_ids = live_ids[step_ok]
            p_next_cam = points_cam[step_ok] + sf_vec[step_ok]
            p_next_cam_h = np.concatenate(
                [p_next_cam.astype(np.float32), np.ones((len(step_ids), 1), dtype=np.float32)],
                axis=1,
            )
            p_next_world = (t_wc_arr[fi + 1] @ p_next_cam_h.T).T[:, :3]
            finite_next = np.isfinite(p_next_world).all(axis=-1)
            if not finite_next.any():
                continue
            traj_3d_world[fi + 1, step_ids[finite_next]] = p_next_world[finite_next].astype(np.float32)
            traj_valid[fi + 1, step_ids[finite_next]] = True

        for fi in range(t_clip):
            live_ids = np.where(traj_valid[fi])[0]
            if live_ids.size == 0:
                continue
            traj_visible[fi, live_ids] = _visibility_from_depth(fi, traj_3d_world[fi, live_ids])

        query, target, mask, query_stats = _build_vkitti2_queries_from_trajectories(
            rng=self.rng,
            traj_3d_world=traj_3d_world,
            traj_visible=traj_visible,
            traj_valid=traj_valid,
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
                "dataset": "virtual_kitti2_raw",
                "scene_id": scene.scene_id,
                "clip_start": int(clip_start),
                "source_mode": "vkitti2_scene_flow_tracks",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            idxs = self._frame_indices(scene.frame_count, query_index)
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
                    dataset="virtual_kitti2_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"VirtualKitti2RawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
