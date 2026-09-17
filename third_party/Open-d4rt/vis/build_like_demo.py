#!/usr/bin/env python3
"""Export a demo package: video + animated RGB point cloud + sparse tracks."""

from __future__ import annotations

import json
import math
import subprocess
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import (
    _apply_sim3_to_xyz,
    _encode_model_memory,
    _grid_query_points,
    _infer_tracks,
    _load_video_rgb,
    _make_anchor_clip_indices,
    _make_sliding_window_clip_ranges,
    _model_clip_frames,
    _resolve_device,
    _resize_video,
    _run_clip_queries_for_target_indices,
    _unwrap_state_dict,
    _estimate_overlap_sim3,
)
from src.eval.tasks import _estimate_intrinsics_params_from_predictions, _run_model_for_queries, _solve_scale_only, _umeyama_rigid


def _jsonable_float_array(arr: np.ndarray, ndigits: int = 6) -> list[Any]:
    arr64 = np.asarray(arr, dtype=np.float64)
    arr64 = np.round(arr64, decimals=int(ndigits))
    return arr64.tolist()


def _sample_rgb_from_grid(video_rgb: np.ndarray, grid_points_px: np.ndarray) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    pts = np.asarray(grid_points_px, dtype=np.float32)
    t = int(video.shape[0])
    n = int(pts.shape[0])
    rgb = np.empty((t, n, 3), dtype=np.uint8)
    for ti in range(t):
        for qi in range(n):
            x = int(np.clip(np.rint(float(pts[qi, 0])), 0, max(video.shape[2] - 1, 0)))
            y = int(np.clip(np.rint(float(pts[qi, 1])), 0, max(video.shape[1] - 1, 0)))
            rgb[ti, qi] = video[ti, y, x]
    return rgb


def _sample_rgb_from_uv_sequence(video_rgb: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    uv = np.asarray(uv_px, dtype=np.float32)
    t = int(video.shape[0])
    n = int(uv.shape[1])
    rgb = np.zeros((t, n, 3), dtype=np.uint8)
    for ti in range(t):
        for qi in range(n):
            x = float(uv[ti, qi, 0])
            y = float(uv[ti, qi, 1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xi = int(np.clip(np.rint(x), 0, max(video.shape[2] - 1, 0)))
            yi = int(np.clip(np.rint(y), 0, max(video.shape[1] - 1, 0)))
            rgb[ti, qi] = video[ti, yi, xi]
    return rgb


def _detect_vertical_valid_crop(video_rgb: np.ndarray, black_bar_threshold: float) -> tuple[int, int]:
    video = np.asarray(video_rgb, dtype=np.uint8)
    if video.ndim != 4 or video.shape[0] <= 0:
        return 0, int(video.shape[1]) if video.ndim >= 2 else 0
    t, h = int(video.shape[0]), int(video.shape[1])
    sample_ids = np.linspace(0, t - 1, num=min(t, 8), dtype=np.int64)
    sampled = video[sample_ids]
    row_signal = np.median(np.max(sampled, axis=(0, 2, 3)), axis=0).astype(np.float32)
    valid = row_signal > float(black_bar_threshold)
    if int(np.count_nonzero(valid)) < max(16, h // 3):
        return 0, h
    top = int(np.argmax(valid))
    bottom = int(h - np.argmax(valid[::-1]))
    if bottom - top < max(16, h // 3):
        return 0, h
    return top, bottom


def _crop_video_vertical(video_rgb: np.ndarray, top: int, bottom: int) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    top_i = int(np.clip(top, 0, max(video.shape[1] - 1, 0)))
    bottom_i = int(np.clip(bottom, top_i + 1, int(video.shape[1])))
    return video[:, top_i:bottom_i].copy()


def _build_uv_grid(width: int, height: int, cols: int, rows: int, max_points: int) -> np.ndarray:
    pts = _grid_query_points(
        width=int(width),
        height=int(height),
        cols=int(cols),
        rows=int(rows),
        margin_ratio=0.02,
        max_points=int(max_points),
    )
    return pts.astype(np.float32)


def _make_normalized_uv_grid(grid_size: int) -> np.ndarray:
    size = max(2, int(grid_size))
    coords = np.linspace(0.0, 1.0, num=size, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, indexing="xy"), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32)


def _infer_regular_grid_shape(query_uv_px: np.ndarray) -> tuple[int, int] | None:
    pts = np.asarray(query_uv_px, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] <= 0:
        return None
    xs = np.unique(np.round(pts[:, 0], decimals=4))
    ys = np.unique(np.round(pts[:, 1], decimals=4))
    if int(xs.size) * int(ys.size) != int(pts.shape[0]):
        return None
    return int(ys.size), int(xs.size)


def _compute_non_boundary_candidate_mask(
    *,
    query_uv_px: np.ndarray,
    xyz_ref0_frame0: np.ndarray,
    visibility_frame0: np.ndarray,
    rel_thresh: float,
    abs_thresh: float,
    dilate_radius: int,
) -> np.ndarray:
    num_points = int(query_uv_px.shape[0])
    keep = np.ones((num_points,), dtype=bool)
    grid_shape = _infer_regular_grid_shape(query_uv_px)
    if grid_shape is None:
        return keep

    rows, cols = grid_shape
    xyz = np.asarray(xyz_ref0_frame0, dtype=np.float32).reshape(rows, cols, 3)
    vis = np.asarray(visibility_frame0, dtype=bool).reshape(rows, cols)
    finite = np.isfinite(xyz).all(axis=-1)
    valid = vis & finite
    z = xyz[..., 2]

    boundary = np.zeros((rows, cols), dtype=bool)

    def _mark(diff: np.ndarray, z_ref: np.ndarray, vmask: np.ndarray, sl_a: tuple[slice, slice], sl_b: tuple[slice, slice]) -> None:
        thresh = np.maximum(float(abs_thresh), float(rel_thresh) * np.maximum(np.abs(z_ref), 1e-6))
        edge = vmask & np.isfinite(diff) & (diff > thresh)
        boundary[sl_a] |= edge
        boundary[sl_b] |= edge

    if cols > 1:
        diff_x = np.abs(z[:, 1:] - z[:, :-1])
        vmask_x = valid[:, 1:] & valid[:, :-1]
        z_ref_x = np.minimum(np.abs(z[:, 1:]), np.abs(z[:, :-1]))
        _mark(diff_x, z_ref_x, vmask_x, (slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    if rows > 1:
        diff_y = np.abs(z[1:, :] - z[:-1, :])
        vmask_y = valid[1:, :] & valid[:-1, :]
        z_ref_y = np.minimum(np.abs(z[1:, :]), np.abs(z[:-1, :]))
        _mark(diff_y, z_ref_y, vmask_y, (slice(1, None), slice(None)), (slice(None, -1), slice(None)))

    if int(dilate_radius) > 0:
        try:
            import cv2

            k = int(2 * int(dilate_radius) + 1)
            kernel = np.ones((k, k), dtype=np.uint8)
            boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1) > 0
        except Exception:
            pass

    keep = (~boundary).reshape(-1)
    keep &= valid.reshape(-1)
    return keep


def _sigmoid_to_bool(logits_or_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(logits_or_scores, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-values))) > 0.5


def _compute_point_motion_scores(
    *,
    xyz_ref0: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_ref0, dtype=np.float32)
    vis = np.asarray(visibility, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float32)
    num_frames, num_points = xyz.shape[:2]
    finite = np.isfinite(xyz).all(axis=-1)
    valid = vis & finite
    motion_scores = np.zeros((num_points,), dtype=np.float32)
    visible_counts = valid.sum(axis=0).astype(np.int32)

    for qi in range(num_points):
        valid_idx = np.flatnonzero(valid[:, qi])
        if valid_idx.size < 2:
            continue
        pts = xyz[valid_idx, qi]
        ref = pts[0]
        displacement = np.linalg.norm(pts - ref[None, :], axis=-1)
        smooth_motion = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
        motion_scores[qi] = (
            float(np.nanpercentile(displacement, 90)) +
            0.35 * float(np.nanpercentile(smooth_motion, 75)) +
            0.02 * float(np.nanmean(conf[valid_idx, qi]))
        )

    return motion_scores, visible_counts


def _select_motion_tracks(
    *,
    query_uv_px: np.ndarray,
    xyz_ref0: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    max_tracks: int,
    min_visible_frames: int,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    num_points = int(query_uv_px.shape[0])
    if num_points == 0:
        return np.empty((0,), dtype=np.int64)

    motion_scores, visible_counts = _compute_point_motion_scores(
        xyz_ref0=xyz_ref0,
        visibility=visibility,
        confidence=confidence,
    )
    pts_uv = np.asarray(query_uv_px, dtype=np.float32)
    min_visible_frames = max(2, int(min_visible_frames))

    border_margin = 0.08
    uv_norm = pts_uv.copy()
    uv_norm[:, 0] /= float(max(float(np.max(pts_uv[:, 0])) if num_points > 0 else 1.0, 1.0))
    uv_norm[:, 1] /= float(max(float(np.max(pts_uv[:, 1])) if num_points > 0 else 1.0, 1.0))
    border_dist = np.minimum.reduce([
        uv_norm[:, 0],
        uv_norm[:, 1],
        1.0 - uv_norm[:, 0],
        1.0 - uv_norm[:, 1],
    ])
    border_bonus = np.clip((border_dist - border_margin) / max(1.0 - 2.0 * border_margin, 1e-6), 0.0, 1.0)

    high_motion_thresh = float(np.nanpercentile(motion_scores, 85)) if np.any(motion_scores > 0) else 0.0
    interior_mask = motion_scores >= high_motion_thresh
    density_bonus = np.zeros((num_points,), dtype=np.float32)
    if np.count_nonzero(interior_mask) >= 4:
        focus_pts = pts_uv[interior_mask]
        for qi in range(num_points):
            dist2 = np.sum((focus_pts - pts_uv[qi][None, :]) ** 2, axis=1)
            density_bonus[qi] = float(np.sum(dist2 < (28.0 ** 2)))

    scores = np.full((num_points,), -np.inf, dtype=np.float32)
    for qi in range(num_points):
        if allowed_mask is not None and not bool(allowed_mask[qi]):
            continue
        if visible_counts[qi] < min_visible_frames:
            continue
        scores[qi] = (
            motion_scores[qi] * (0.65 + 0.35 * border_bonus[qi]) +
            0.08 * density_bonus[qi]
        )

    ranked = np.argsort(scores)[::-1]
    ranked = ranked[np.isfinite(scores[ranked])]
    if ranked.size == 0:
        fallback = np.argsort(visible_counts)[::-1]
        return fallback[: max(1, min(int(max_tracks), fallback.size))].astype(np.int64)
    return ranked[: max(1, min(int(max_tracks), ranked.size))].astype(np.int64)


def _estimate_ref0_intrinsics(
    *,
    xyz_ref0_frame0: np.ndarray,
    uv_px_frame0: np.ndarray,
    visibility_frame0: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    xyz = np.asarray(xyz_ref0_frame0, dtype=np.float32)
    uv = np.asarray(uv_px_frame0, dtype=np.float32)
    vis = np.asarray(visibility_frame0, dtype=bool)
    valid = vis & np.isfinite(xyz).all(axis=-1) & np.isfinite(uv).all(axis=-1) & (xyz[:, 2] > 1e-5)
    if int(np.count_nonzero(valid)) < 8:
        fx = fy = 0.5 * float(max(image_width, image_height))
        cx = 0.5 * float(max(image_width - 1, 0))
        cy = 0.5 * float(max(image_height - 1, 0))
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    x_over_z = (xyz[valid, 0] / xyz[valid, 2]).astype(np.float64)
    y_over_z = (xyz[valid, 1] / xyz[valid, 2]).astype(np.float64)
    u = uv[valid, 0].astype(np.float64)
    v = uv[valid, 1].astype(np.float64)

    ax = np.stack([x_over_z, np.ones_like(x_over_z)], axis=1)
    ay = np.stack([y_over_z, np.ones_like(y_over_z)], axis=1)
    fx, cx = np.linalg.lstsq(ax, u, rcond=None)[0].tolist()
    fy, cy = np.linalg.lstsq(ay, v, rcond=None)[0].tolist()

    if not np.isfinite(fx) or abs(fx) < 1e-5:
        fx = 0.5 * float(max(image_width, image_height))
    if not np.isfinite(fy) or abs(fy) < 1e-5:
        fy = 0.5 * float(max(image_width, image_height))
    if not np.isfinite(cx):
        cx = 0.5 * float(max(image_width - 1, 0))
    if not np.isfinite(cy):
        cy = 0.5 * float(max(image_height - 1, 0))
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _enforce_shared_focal_intrinsics(intrinsics_params: np.ndarray) -> np.ndarray:
    intr = np.asarray(intrinsics_params, dtype=np.float64).reshape(-1)
    if intr.shape[0] != 4:
        raise ValueError(f"Expected 4 intrinsics params [fx, fy, cx, cy], got shape {intr.shape}.")
    fx, fy, cx, cy = [float(v) for v in intr.tolist()]
    valid_focals = np.asarray([fx, fy], dtype=np.float64)
    valid_focals = valid_focals[np.isfinite(valid_focals) & (valid_focals > 1e-6)]
    if valid_focals.size == 0:
        shared_f = 1.0
    else:
        shared_f = float(np.median(valid_focals))
    return np.asarray([shared_f, shared_f, cx, cy], dtype=np.float64)


def _quat_wxyz_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(v) for v in np.asarray(quat_wxyz, dtype=np.float64).reshape(4).tolist()]
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 1e-12:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    rot = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)
    return rot


def _apply_vertical_crop_to_camera_data(camera_data: dict[str, np.ndarray], top: int) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in camera_data.items()}
    if "K" in out:
        out["K"] = np.asarray(out["K"], dtype=np.float32).copy()
        out["K"][:, 1, 2] -= float(top)
    return out


def _load_kubric_camera_sequence(
    *,
    tfds_root: Path,
    sample_json_path: Path,
    expected_num_frames: int,
) -> dict[str, np.ndarray] | None:
    import tensorflow_datasets as tfds

    sample_meta = json.loads(sample_json_path.read_text(encoding="utf-8"))
    sample_index = int(sample_meta.get("sample_index", 0))
    split = str(sample_meta.get("split", "validation"))
    builder = tfds.builder_from_directory(str(tfds_root))
    ds = builder.as_dataset(split=split, shuffle_files=False)
    ex = next(iter(tfds.as_numpy(ds.skip(sample_index).take(1))))

    h = int(ex["metadata"]["height"])
    w = int(ex["metadata"]["width"])
    cam_pos = np.asarray(ex["camera"]["positions"], dtype=np.float32)
    cam_quat = np.asarray(ex["camera"]["quaternions"], dtype=np.float32)
    fov = float(ex["camera"]["field_of_view"])
    fx = 0.5 * float(w) / max(np.tan(0.5 * fov), 1e-6)
    fy = 0.5 * float(h) / max(np.tan(0.5 * fov), 1e-6)
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5
    s_blender_to_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    t_count = int(min(expected_num_frames, cam_pos.shape[0], cam_quat.shape[0]))
    k_seq = np.tile(np.eye(3, dtype=np.float32)[None], (t_count, 1, 1))
    k_seq[:, 0, 0] = fx
    k_seq[:, 1, 1] = fy
    k_seq[:, 0, 2] = cx
    k_seq[:, 1, 2] = cy
    t_wc_seq = np.tile(np.eye(4, dtype=np.float32)[None], (t_count, 1, 1))
    for i in range(t_count):
        r_bl = _quat_wxyz_to_rot(cam_quat[i])
        r_cv = r_bl @ s_blender_to_cv
        t_wc_seq[i, :3, :3] = r_cv
        t_wc_seq[i, :3, 3] = cam_pos[i]

    t_c0_w = np.linalg.inv(t_wc_seq[0]).astype(np.float32)
    t_ref0_cam = np.matmul(t_c0_w[None, ...], t_wc_seq).astype(np.float32)
    return {
        "K": k_seq.astype(np.float32),
        "T_ref0_cam": t_ref0_cam.astype(np.float32),
        "T_wc": t_wc_seq.astype(np.float32),
    }


def _load_kubric_sample(
    *,
    tfds_root: Path,
    sample_json_path: Path,
) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    sample_meta = json.loads(sample_json_path.read_text(encoding="utf-8"))
    sample_index = int(sample_meta.get("sample_index", 0))
    split = str(sample_meta.get("split", "validation"))
    builder = tfds.builder_from_directory(str(tfds_root))
    ds = builder.as_dataset(split=split, shuffle_files=False)
    return next(iter(tfds.as_numpy(ds.skip(sample_index).take(1))))


def _decode_depth_u16_to_metric(depth_u16: np.ndarray, depth_range: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_u16, dtype=np.float32)
    lo = float(depth_range[0])
    hi = float(depth_range[1])
    return lo + d / 65535.0 * (hi - lo)


def _decode_object_coordinates_u16(obj_u16: np.ndarray) -> np.ndarray:
    return np.asarray(obj_u16, dtype=np.float32) / 65535.0 - 0.5


def _build_object_local_to_world(bboxes_3d_ot83: np.ndarray) -> np.ndarray:
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
            bbox = np.asarray(bboxes_3d_ot83[oi, ti], dtype=np.float32)
            if bbox.shape != (8, 3) or not np.isfinite(bbox).all():
                continue
            bbox_h = np.concatenate([bbox, np.ones((8, 1), dtype=np.float32)], axis=-1)
            try:
                m, *_ = np.linalg.lstsq(local_box, bbox_h, rcond=None)
            except np.linalg.LinAlgError:
                continue
            out[oi, ti] = m.astype(np.float32)
    return out


def _project_point(k: np.ndarray, t_cw: np.ndarray, p_world: np.ndarray) -> tuple[float, float, float]:
    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32)
    p_cam = (t_cw @ p_h)[:3]
    z = float(p_cam[2])
    if not np.isfinite(z):
        return np.nan, np.nan, np.nan
    if z <= 1e-6:
        return np.nan, np.nan, z
    proj = k @ p_cam
    return float(proj[0] / z), float(proj[1] / z), z


def _depth_seg_occlusion(
    *,
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
    x0 = int(np.clip(np.floor(u), 0, w - 1))
    x1 = int(np.clip(x0 + 1, 0, w - 1))
    y0 = int(np.clip(np.floor(v), 0, h - 1))
    y1 = int(np.clip(y0 + 1, 0, h - 1))
    coords = ((y0, x0), (y1, x0), (y0, x1), (y1, x1))

    ds: list[float] = []
    seg_match: list[bool] = []
    for yy, xx in coords:
        if bool(depth_valid_hw[yy, xx]):
            ds.append(float(depth_z_hw[yy, xx]))
        seg_match.append(int(seg_hw[yy, xx]) == int(seg_id))
    if not ds:
        return True
    depth_nn = max(ds)
    depth_occluded = depth_nn < (float(z_proj) * 0.99)
    seg_occluded = int(seg_id) > 0 and not any(seg_match)
    return bool(depth_occluded or seg_occluded)


def _unproject_background_world(
    *,
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
    return (t_wc @ p_h)[:3].astype(np.float32)


def _transform_world_to_ref(points_world_tn3: np.ndarray, t_wc_ref: np.ndarray) -> np.ndarray:
    t_cw_ref = np.linalg.inv(np.asarray(t_wc_ref, dtype=np.float32))
    pts = np.asarray(points_world_tn3, dtype=np.float32)
    flat = pts.reshape(-1, 3)
    flat_h = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    out = (flat_h @ t_cw_ref.T)[:, :3].astype(np.float32)
    return out.reshape(pts.shape)


def _load_kubric_gt_tracks(
    *,
    tfds_root: Path,
    sample_json_path: Path,
    track_query_uv_px: np.ndarray,
    track_query_t_src: np.ndarray,
    expected_num_frames: int,
    crop_top: int,
    crop_bottom: int,
    output_video_hw: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    sample = _load_kubric_sample(tfds_root=tfds_root, sample_json_path=sample_json_path)
    raw_video = np.asarray(sample["video"], dtype=np.uint8)
    raw_h = int(raw_video.shape[1])
    raw_w = int(raw_video.shape[2])
    t_count = int(min(expected_num_frames, raw_video.shape[0]))
    if t_count <= 0:
        return None

    crop_top_i = int(np.clip(crop_top, 0, max(raw_h - 1, 0)))
    crop_bottom_i = int(np.clip(crop_bottom, crop_top_i + 1, raw_h))
    cropped_h = int(max(crop_bottom_i - crop_top_i, 1))
    out_h, out_w = int(output_video_hw[0]), int(output_video_hw[1])
    sx = float(raw_w - 1) / float(max(out_w - 1, 1))
    sy = float(cropped_h - 1) / float(max(out_h - 1, 1))

    depth_u16 = np.asarray(sample["depth"], dtype=np.uint16)[:t_count, ..., 0]
    seg = np.asarray(sample["segmentations"], dtype=np.int32)[:t_count, ..., 0]
    obj_coord_local = _decode_object_coordinates_u16(np.asarray(sample["object_coordinates"], dtype=np.uint16)[:t_count])
    depth_range = np.asarray(sample["metadata"]["depth_range"], dtype=np.float32)
    depth_range_m = _decode_depth_u16_to_metric(depth_u16, depth_range)
    camera_data = _load_kubric_camera_sequence(
        tfds_root=tfds_root,
        sample_json_path=sample_json_path,
        expected_num_frames=t_count,
    )
    if camera_data is None:
        return None
    k_seq = np.asarray(camera_data["K"], dtype=np.float32)
    t_wc_seq = np.asarray(camera_data["T_wc"], dtype=np.float32)
    cam_valid = np.isfinite(t_wc_seq).all(axis=(1, 2)) & np.isfinite(k_seq).all(axis=(1, 2))
    t_cw_seq = np.full((t_count, 4, 4), np.nan, dtype=np.float32)
    for ti in range(t_count):
        if not bool(cam_valid[ti]):
            continue
        try:
            t_cw_seq[ti] = np.linalg.inv(t_wc_seq[ti]).astype(np.float32)
        except np.linalg.LinAlgError:
            cam_valid[ti] = False

    cx = float(k_seq[0, 0, 2])
    cy = float(k_seq[0, 1, 2])
    fx = float(k_seq[0, 0, 0])
    fy = float(k_seq[0, 1, 1])
    uu = np.arange(raw_w, dtype=np.float32)[None, :]
    vv = np.arange(raw_h, dtype=np.float32)[:, None]
    x = (uu - cx) / max(fx, 1e-6)
    y = (vv - cy) / max(fy, 1e-6)
    ray_norm = np.sqrt(x * x + y * y + 1.0).astype(np.float32)
    depth_z = depth_range_m * (1.0 / np.maximum(ray_norm[None, :, :], 1e-6))
    depth_valid = np.isfinite(depth_z) & (depth_z > 0.0)

    bboxes_all = np.asarray(sample["instances"]["bboxes_3d"], dtype=np.float32)[:, :t_count]
    obj_l2w = _build_object_local_to_world(bboxes_all)
    num_obj = int(obj_l2w.shape[0])

    track_query_uv_px = np.asarray(track_query_uv_px, dtype=np.float32)
    track_query_t_src = np.asarray(track_query_t_src, dtype=np.int64).reshape(-1)
    num_tracks = int(min(track_query_uv_px.shape[0], track_query_t_src.shape[0]))
    if num_tracks <= 0:
        return None

    gt_xyz_ref0 = np.full((num_tracks, t_count, 3), np.nan, dtype=np.float32)
    gt_uv_px = np.full((num_tracks, t_count, 2), np.nan, dtype=np.float32)
    gt_vis = np.zeros((num_tracks, t_count), dtype=np.bool_)
    gt_conf = np.zeros((num_tracks, t_count), dtype=np.float32)

    for qi in range(num_tracks):
        fs = int(np.clip(track_query_t_src[qi], 0, max(t_count - 1, 0)))
        if not bool(cam_valid[fs]):
            continue
        u_src_raw = int(np.clip(np.rint(float(track_query_uv_px[qi, 0]) * sx), 0, raw_w - 1))
        v_src_raw = int(np.clip(np.rint(float(track_query_uv_px[qi, 1]) * sy + float(crop_top_i)), 0, raw_h - 1))
        seg_id = int(seg[fs, v_src_raw, u_src_raw])
        p_world_seq = np.full((t_count, 3), np.nan, dtype=np.float32)

        if seg_id > 0 and (seg_id - 1) < num_obj:
            obj_idx = seg_id - 1
            local = np.concatenate([obj_coord_local[fs, v_src_raw, u_src_raw], np.array([1.0], dtype=np.float32)], axis=0)
            for ti in range(t_count):
                m_tgt = obj_l2w[obj_idx, ti]
                if not np.isfinite(m_tgt).all():
                    continue
                world_h = local @ m_tgt
                if abs(float(world_h[3])) <= 1e-6:
                    continue
                p_world_seq[ti] = (world_h[:3] / world_h[3]).astype(np.float32)
        else:
            p_world = _unproject_background_world(
                u=u_src_raw,
                v=v_src_raw,
                depth_range_value=float(depth_range_m[fs, v_src_raw, u_src_raw]),
                k=k_seq[fs],
                t_wc=t_wc_seq[fs],
            )
            if np.isfinite(p_world).all():
                p_world_seq[:] = p_world.astype(np.float32)
                seg_id = 0

        if not np.isfinite(p_world_seq[fs]).all():
            continue
        gt_xyz_ref0[qi] = _transform_world_to_ref(p_world_seq, t_wc_seq[0])
        for ti in range(t_count):
            if not bool(cam_valid[ti]) or not np.isfinite(p_world_seq[ti]).all():
                continue
            u_tgt_raw, v_tgt_raw, z_tgt = _project_point(k_seq[ti], t_cw_seq[ti], p_world_seq[ti])
            in_img = (
                np.isfinite(u_tgt_raw)
                and np.isfinite(v_tgt_raw)
                and np.isfinite(z_tgt)
                and (z_tgt > 1e-6)
                and (0.0 <= u_tgt_raw <= float(raw_w - 1))
                and (float(crop_top_i) <= v_tgt_raw <= float(crop_bottom_i - 1))
            )
            if not in_img:
                continue
            gt_uv_px[qi, ti, 0] = float((u_tgt_raw) / max(sx, 1e-6))
            gt_uv_px[qi, ti, 1] = float((v_tgt_raw - float(crop_top_i)) / max(sy, 1e-6))
            gt_vis[qi, ti] = not _depth_seg_occlusion(
                depth_z_hw=depth_z[ti],
                depth_valid_hw=depth_valid[ti],
                seg_hw=seg[ti],
                u=float(u_tgt_raw),
                v=float(v_tgt_raw),
                z_proj=float(z_tgt),
                seg_id=int(seg_id),
            )
            if gt_vis[qi, ti]:
                gt_conf[qi, ti] = 1.0

    return {
        "query_uv_px": track_query_uv_px[:num_tracks].astype(np.float32),
        "query_t_src": track_query_t_src[:num_tracks].astype(np.int64),
        "xyz_ref0": gt_xyz_ref0.astype(np.float32),
        "uv_px": gt_uv_px.astype(np.float32),
        "visibility": gt_vis.astype(np.bool_),
        "confidence": gt_conf.astype(np.float32),
    }


def _align_pred_to_gt_scale_only(
    *,
    pred_xyz_ref0: np.ndarray,
    gt_xyz_ref0: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = np.asarray(pred_xyz_ref0, dtype=np.float32)
    gt = np.asarray(gt_xyz_ref0, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    info: dict[str, Any] = {"type": "global_scale", "applied": False}
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        info["reason"] = "shape_mismatch"
        return pred.copy(), info
    valid = valid & np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if int(np.count_nonzero(valid)) < 3:
        info["reason"] = "not_enough_points"
        return pred.copy(), info
    scale = _solve_scale_only(pred[valid].reshape(-1), gt[valid].reshape(-1))
    if not np.isfinite(scale) or scale <= 0.0:
        info["reason"] = "invalid_scale"
        return pred.copy(), info
    aligned = (pred.astype(np.float64) * float(scale)).astype(np.float32)
    info.update({"applied": True, "scale": float(scale)})
    return aligned, info


def _load_kubric_dynamic_masks(
    *,
    tfds_root: Path,
    sample_json_path: Path,
    expected_num_frames: int,
    crop_top: int,
    crop_bottom: int,
) -> np.ndarray | None:
    import tensorflow_datasets as tfds

    sample_meta = json.loads(sample_json_path.read_text(encoding="utf-8"))
    sample_index = int(sample_meta.get("sample_index", 0))
    split = str(sample_meta.get("split", "validation"))
    builder = tfds.builder_from_directory(str(tfds_root))
    ds = builder.as_dataset(split=split, shuffle_files=False)
    ex = next(iter(tfds.as_numpy(ds.skip(sample_index).take(1))))

    seg = np.asarray(ex["segmentations"], dtype=np.int32)
    if seg.ndim == 4 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    inst_dynamic = np.asarray(ex["instances"]["is_dynamic"], dtype=bool)
    t_count = int(min(expected_num_frames, seg.shape[0]))
    seg = seg[:t_count]
    dyn = np.zeros_like(seg, dtype=bool)
    valid_inst = seg > 0
    inst_ids = seg[valid_inst] - 1
    good = (inst_ids >= 0) & (inst_ids < inst_dynamic.shape[0])
    dyn_idx = np.flatnonzero(valid_inst)
    dyn.reshape(-1)[dyn_idx[good]] = inst_dynamic[inst_ids[good]]
    top = int(np.clip(crop_top, 0, max(seg.shape[1] - 1, 0)))
    bottom = int(np.clip(crop_bottom, top + 1, int(seg.shape[1])))
    return dyn[:, top:bottom].copy()


def _build_query_tensors(
    *,
    query_uv_norm: np.ndarray,
    t_src: np.ndarray,
    t_tgt: np.ndarray,
    t_cam: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "u": torch.from_numpy(np.asarray(query_uv_norm[:, 0], dtype=np.float32)).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(np.asarray(query_uv_norm[:, 1], dtype=np.float32)).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(np.asarray(t_src, dtype=np.int64)).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(np.asarray(t_tgt, dtype=np.int64)).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(np.asarray(t_cam, dtype=np.int64)).to(device=device, dtype=torch.long),
    }


def _run_point_cloud_queries_for_target_indices(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    local_target_indices: np.ndarray,
    query_chunk_size: int,
) -> dict[str, np.ndarray]:
    target_ids = np.asarray(local_target_indices, dtype=np.int64).reshape(-1)
    num_queries = int(query_uv_norm.shape[0])
    num_targets = int(target_ids.shape[0])
    if num_targets <= 0:
        return {}
    repeated_uv = np.tile(np.asarray(query_uv_norm, dtype=np.float32), (num_targets, 1))
    t_src = np.repeat(target_ids, num_queries)
    t_tgt = np.repeat(target_ids, num_queries)
    t_cam = np.zeros_like(t_tgt)
    query = _build_query_tensors(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_cam,
        device=video_clip.device,
    )
    pred = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    out: dict[str, np.ndarray] = {}
    for key, value in pred.items():
        arr = value.numpy()
        if arr.ndim == 1:
            out[key] = arr.reshape(num_targets, num_queries)
        else:
            out[key] = arr.reshape(num_targets, num_queries, *arr.shape[1:])
    return out


def _apply_sim3_to_pose_seq(t_ref_cam: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pose = np.asarray(t_ref_cam, dtype=np.float64)
    out = pose.copy()
    if pose.ndim != 3 or pose.shape[1:] != (4, 4):
        return pose.astype(np.float32)
    out[:, :3, :3] = np.asarray(rot, dtype=np.float64)[None, ...] @ pose[:, :3, :3]
    out[:, :3, 3] = (
        float(scale) * (np.asarray(rot, dtype=np.float64) @ pose[:, :3, 3].T)
    ).T + np.asarray(trans, dtype=np.float64)[None, :]
    return out.astype(np.float32)


def _infer_point_cloud_ref0(
    *,
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    point_query_uv_norm: np.ndarray,
    query_chunk_size: int,
    umeyama_slide_window: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float, np.ndarray, np.ndarray]]]:
    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    num_points = int(point_query_uv_norm.shape[0])
    clip_frames = _model_clip_frames(model)
    aspect_value = np.asarray([[float(video_model_rgb.shape[2]) / float(max(1, video_model_rgb.shape[1]))]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)
    video_tensor = torch.from_numpy(video_model_rgb).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0

    points_xyz_ref0 = np.full((num_frames, num_points, 3), np.nan, dtype=np.float32)
    points_vis = np.zeros((num_frames, num_points), dtype=bool)
    points_conf = np.full((num_frames, num_points), np.nan, dtype=np.float32)
    chunk_transforms: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        if num_frames <= clip_frames or not bool(umeyama_slide_window):
            clip_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
            for frame_idx in range(num_frames):
                clip_indices = _make_anchor_clip_indices(num_frames=num_frames, clip_frames=clip_frames, target_idx=frame_idx)
                local_tgt_idx = int(np.flatnonzero(clip_indices == frame_idx)[0])
                clip_groups.setdefault(tuple(int(v) for v in clip_indices.tolist()), []).append((frame_idx, local_tgt_idx))
            for clip_key, assignments in clip_groups.items():
                clip_indices = np.asarray(clip_key, dtype=np.int64)
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)
                global_frame_ids = np.asarray([item[0] for item in assignments], dtype=np.int64)
                local_target_ids = np.asarray([item[1] for item in assignments], dtype=np.int64)
                pred_point_ref = _run_point_cloud_queries_for_target_indices(
                    model=model,
                    video_clip=video_clip,
                    aspect_ratio=aspect_tensor,
                    memory=memory,
                    query_uv_norm=point_query_uv_norm,
                    local_target_indices=local_target_ids,
                    query_chunk_size=query_chunk_size,
                )
                points_xyz_ref0[global_frame_ids] = pred_point_ref["xyz_3d"].astype(np.float32)
                points_vis[global_frame_ids] = np.isfinite(points_xyz_ref0[global_frame_ids]).all(axis=-1)
                points_conf[global_frame_ids] = pred_point_ref.get("confidence", np.ones((local_target_ids.shape[0], num_points), dtype=np.float32)).astype(np.float32)
            chunk_transforms.append((0, num_frames, 1.0, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64)))
            return points_xyz_ref0, points_vis, points_conf, chunk_transforms

        window_ranges = _make_sliding_window_clip_ranges(num_frames=num_frames, clip_frames=clip_frames)
        for window_idx, (start, end) in enumerate(window_ranges):
            clip_indices = np.arange(start, end, dtype=np.int64)
            local_target_ids = np.arange(end - start, dtype=np.int64)
            video_clip = video_tensor[:, clip_indices]
            memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)
            pred_point_ref = _run_point_cloud_queries_for_target_indices(
                model=model,
                video_clip=video_clip,
                aspect_ratio=aspect_tensor,
                memory=memory,
                query_uv_norm=point_query_uv_norm,
                local_target_indices=local_target_ids,
                query_chunk_size=query_chunk_size,
            )
            chunk_xyz = pred_point_ref["xyz_3d"].astype(np.float32)
            chunk_conf = pred_point_ref.get("confidence", np.ones((local_target_ids.shape[0], num_points), dtype=np.float32)).astype(np.float32)
            chunk_vis = np.isfinite(chunk_xyz).all(axis=-1)
            scale = 1.0
            rot = np.eye(3, dtype=np.float64)
            trans = np.zeros((3,), dtype=np.float64)
            if window_idx > 0:
                prev_start, prev_end, *_ = chunk_transforms[-1]
                overlap_start = int(max(start, prev_start))
                overlap_end = int(min(end, prev_end))
                if overlap_end > overlap_start:
                    overlap_global = np.arange(overlap_start, overlap_end, dtype=np.int64)
                    overlap_local = overlap_global - int(start)
                    sim3 = _estimate_overlap_sim3(
                        prev_xyz_qt3=np.transpose(points_xyz_ref0[overlap_global], (1, 0, 2)),
                        curr_xyz_qt3=np.transpose(chunk_xyz[overlap_local], (1, 0, 2)),
                        prev_vis_qt=np.transpose(points_vis[overlap_global], (1, 0)),
                        curr_vis_qt=np.transpose(chunk_vis[overlap_local], (1, 0)),
                        prev_conf_qt=np.transpose(points_conf[overlap_global], (1, 0)),
                        curr_conf_qt=np.transpose(chunk_conf[overlap_local], (1, 0)),
                    )
                    if sim3 is not None:
                        scale, rot, trans = sim3
                        chunk_xyz = _apply_sim3_to_xyz(chunk_xyz, scale, rot, trans)
            chunk_transforms.append((int(start), int(end), float(scale), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)))
            for local_idx, global_idx in enumerate(clip_indices.tolist()):
                current_conf = chunk_conf[local_idx]
                existing_conf = points_conf[global_idx]
                current_ok = np.isfinite(current_conf)
                existing_ok = np.isfinite(existing_conf)
                better = (~existing_ok & current_ok) | (current_ok & existing_ok & (current_conf >= existing_conf))
                better |= (~existing_ok & ~current_ok & chunk_vis[local_idx])
                if not np.any(better):
                    continue
                points_xyz_ref0[global_idx, better] = chunk_xyz[local_idx, better]
                points_vis[global_idx, better] = chunk_vis[local_idx, better]
                points_conf[global_idx, better] = chunk_conf[local_idx, better]

    return points_xyz_ref0, points_vis, points_conf, chunk_transforms


def _sample_bool_mask_from_uv_sequence(mask_thw: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_thw, dtype=bool)
    uv = np.asarray(uv_px, dtype=np.float32)
    t = int(min(mask.shape[0], uv.shape[0]))
    n = int(uv.shape[1])
    out = np.zeros((t, n), dtype=bool)
    for ti in range(t):
        for qi in range(n):
            x = float(uv[ti, qi, 0])
            y = float(uv[ti, qi, 1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xi = int(np.clip(np.rint(x), 0, max(mask.shape[2] - 1, 0)))
            yi = int(np.clip(np.rint(y), 0, max(mask.shape[1] - 1, 0)))
            out[ti, qi] = bool(mask[ti, yi, xi])
    return out


def _select_dynamic_interior_track_queries(
    *,
    query_uv_px: np.ndarray,
    dynamic_mask_hw: np.ndarray,
    max_tracks: int,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    pts = np.asarray(query_uv_px, dtype=np.float32)
    mask = np.asarray(dynamic_mask_hw, dtype=bool)
    num_points = int(pts.shape[0])
    if num_points == 0:
        return np.empty((0,), dtype=np.int64)

    try:
        import cv2

        mask_u8 = mask.astype(np.uint8) * 255
        erode_kernel = np.ones((9, 9), dtype=np.uint8)
        interior = cv2.erode(mask_u8, erode_kernel, iterations=1) > 0
        dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
    except Exception:
        interior = mask
        dist_map = mask.astype(np.float32)

    scores = np.full((num_points,), -np.inf, dtype=np.float32)
    h, w = mask.shape[:2]
    for qi in range(num_points):
        if allowed_mask is not None and not bool(allowed_mask[qi]):
            continue
        x = int(np.clip(np.rint(float(pts[qi, 0])), 0, max(w - 1, 0)))
        y = int(np.clip(np.rint(float(pts[qi, 1])), 0, max(h - 1, 0)))
        if interior[y, x]:
            scores[qi] = float(dist_map[y, x]) + 1e3
        elif mask[y, x]:
            scores[qi] = float(dist_map[y, x])

    ranked = np.argsort(scores)[::-1]
    ranked = ranked[np.isfinite(scores[ranked])]
    if ranked.size <= 0:
        return np.empty((0,), dtype=np.int64)
    return ranked[: max(1, min(int(max_tracks), ranked.size))].astype(np.int64)


def _estimate_relative_pose_from_queries(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    frame_i_local: int,
    frame_j_local: int,
    query_chunk_size: int,
) -> np.ndarray | None:
    device = video_clip.device
    q = np.asarray(query_uv_norm, dtype=np.float32)
    m = int(q.shape[0])
    if m <= 0:
        return None
    u2 = np.concatenate([q[:, 0], q[:, 0]], axis=0)
    v2 = np.concatenate([q[:, 1], q[:, 1]], axis=0)
    query = {
        "u": torch.from_numpy(u2).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(v2).to(device=device, dtype=torch.float32),
        "t_src": torch.full((2 * m,), int(frame_i_local), dtype=torch.long, device=device),
        "t_tgt": torch.full((2 * m,), int(frame_i_local), dtype=torch.long, device=device),
        "t_cam": torch.cat(
            [
                torch.full((m,), int(frame_i_local), dtype=torch.long, device=device),
                torch.full((m,), int(frame_j_local), dtype=torch.long, device=device),
            ],
            dim=0,
        ),
    }
    pred = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    xyz = pred["xyz_3d"].numpy().astype(np.float64)
    p_i = xyz[:m]
    p_j = xyz[m:]
    finite = np.isfinite(p_i).all(axis=1) & np.isfinite(p_j).all(axis=1)
    if int(np.count_nonzero(finite)) < 16:
        return None
    rigid = _umeyama_rigid(p_i[finite], p_j[finite])
    if rigid is None:
        return None
    r, t = rigid
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.astype(np.float32)
    out[:3, 3] = t.astype(np.float32)
    return out


def _predict_camera_branches(
    *,
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    image_hw: tuple[int, int],
    camera_grid_size: int,
    camera_query_chunk_size: int,
    predict_intrinsics: bool,
    predict_extrinsics: bool,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
) -> dict[str, np.ndarray] | None:
    if not predict_intrinsics and not predict_extrinsics:
        return None

    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    clip_frames = _model_clip_frames(model)
    hm, wm = int(video_model_rgb.shape[1]), int(video_model_rgb.shape[2])
    aspect_value = np.asarray([[float(wm) / float(max(1, hm))]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)
    video_tensor = torch.from_numpy(video_model_rgb).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    coarse_uv = _make_normalized_uv_grid(camera_grid_size)

    k_seq = np.tile(np.eye(3, dtype=np.float32)[None], (num_frames, 1, 1))
    t_ref0_cam = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))
    intr_valid = np.zeros((num_frames,), dtype=np.bool_)
    pose_valid = np.zeros((num_frames,), dtype=np.bool_)
    pose_valid[0] = True

    with torch.no_grad():
        if num_frames <= clip_frames or not (bool(umeyama_slide_window) or bool(umeyama_slide_window_dense)):
            clip_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
            for frame_idx in range(num_frames):
                clip_indices = _make_anchor_clip_indices(num_frames=num_frames, clip_frames=clip_frames, target_idx=frame_idx)
                local_tgt_idx = int(np.flatnonzero(clip_indices == frame_idx)[0])
                clip_groups.setdefault(tuple(int(v) for v in clip_indices.tolist()), []).append((frame_idx, local_tgt_idx))

            for clip_key, assignments in tqdm(clip_groups.items(), total=len(clip_groups), desc="Camera branches", unit="clip", leave=False):
                clip_indices = np.asarray(clip_key, dtype=np.int64)
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)
                global_frame_ids = np.asarray([item[0] for item in assignments], dtype=np.int64)
                local_target_ids = np.asarray([item[1] for item in assignments], dtype=np.int64)

                if predict_intrinsics:
                    pred_point_ref = _run_point_cloud_queries_for_target_indices(
                        model=model,
                        video_clip=video_clip,
                        aspect_ratio=aspect_tensor,
                        memory=memory,
                        query_uv_norm=coarse_uv,
                        local_target_indices=local_target_ids,
                        query_chunk_size=camera_query_chunk_size,
                    )
                    xyz = pred_point_ref["xyz_3d"].astype(np.float32)
                    uv = np.tile(coarse_uv[None, :, :], (xyz.shape[0], 1, 1)).astype(np.float32)
                    for idx_in_clip, frame_global in enumerate(global_frame_ids.tolist()):
                        intr = _estimate_intrinsics_params_from_predictions(
                            pred_tracks=xyz[idx_in_clip][None, ...],
                            pred_uv_norm=uv[idx_in_clip][None, ...],
                            image_hw=image_hw,
                        )
                        intr = _enforce_shared_focal_intrinsics(intr)
                        k_seq[frame_global] = np.array(
                            [[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]],
                            dtype=np.float32,
                        )
                        intr_valid[frame_global] = True

                if predict_extrinsics and np.any(clip_indices == 0):
                    local_ref = int(np.flatnonzero(clip_indices == 0)[0])
                    for frame_global, local_tgt in assignments:
                        frame_global = int(frame_global)
                        if frame_global == 0:
                            continue
                        pose = _estimate_relative_pose_from_queries(
                            model=model,
                            video_clip=video_clip,
                            aspect_ratio=aspect_tensor,
                            memory=memory,
                            query_uv_norm=coarse_uv,
                            frame_i_local=local_ref,
                            frame_j_local=int(local_tgt),
                            query_chunk_size=camera_query_chunk_size,
                        )
                        if pose is not None:
                            try:
                                t_ref0_cam[frame_global] = np.linalg.inv(pose).astype(np.float32)
                                pose_valid[frame_global] = True
                            except np.linalg.LinAlgError:
                                pass
        else:
            window_ranges = _make_sliding_window_clip_ranges(num_frames=num_frames, clip_frames=clip_frames)
            chunk_transforms: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []
            for window_idx, (start, end) in enumerate(tqdm(window_ranges, total=len(window_ranges), desc="Camera branches", unit="clip", leave=False)):
                clip_indices = np.arange(start, end, dtype=np.int64)
                local_target_ids = np.arange(end - start, dtype=np.int64)
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)
                pred_point_ref = _run_point_cloud_queries_for_target_indices(
                    model=model,
                    video_clip=video_clip,
                    aspect_ratio=aspect_tensor,
                    memory=memory,
                    query_uv_norm=coarse_uv,
                    local_target_indices=local_target_ids,
                    query_chunk_size=camera_query_chunk_size,
                )
                xyz = pred_point_ref["xyz_3d"].astype(np.float32)
                conf = pred_point_ref.get("confidence", np.ones((local_target_ids.shape[0], coarse_uv.shape[0]), dtype=np.float32)).astype(np.float32)
                vis = np.isfinite(xyz).all(axis=-1)
                scale = 1.0
                rot = np.eye(3, dtype=np.float64)
                trans = np.zeros((3,), dtype=np.float64)
                if window_idx > 0:
                    prev_start, prev_end, *_ = chunk_transforms[-1]
                    overlap_start = int(max(start, prev_start))
                    overlap_end = int(min(end, prev_end))
                    if overlap_end > overlap_start:
                        overlap_global = np.arange(overlap_start, overlap_end, dtype=np.int64)
                        overlap_local = overlap_global - int(start)
                        sim3 = _estimate_overlap_sim3(
                            prev_xyz_qt3=np.transpose(xyz_global[overlap_global], (1, 0, 2)),
                            curr_xyz_qt3=np.transpose(xyz[overlap_local], (1, 0, 2)),
                            prev_vis_qt=np.transpose(vis_global[overlap_global], (1, 0)),
                            curr_vis_qt=np.transpose(vis[overlap_local], (1, 0)),
                            prev_conf_qt=np.transpose(conf_global[overlap_global], (1, 0)),
                            curr_conf_qt=np.transpose(conf[overlap_local], (1, 0)),
                        )
                        if sim3 is not None:
                            scale, rot, trans = sim3
                            xyz = _apply_sim3_to_xyz(xyz, scale, rot, trans)
                chunk_transforms.append((int(start), int(end), float(scale), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)))
                if window_idx == 0:
                    xyz_global = np.full((num_frames, xyz.shape[1], 3), np.nan, dtype=np.float32)
                    vis_global = np.zeros((num_frames, xyz.shape[1]), dtype=bool)
                    conf_global = np.full((num_frames, xyz.shape[1]), np.nan, dtype=np.float32)
                for local_idx, frame_global in enumerate(clip_indices.tolist()):
                    uv = coarse_uv[None, :, :].astype(np.float32)
                    if predict_intrinsics:
                        intr = _estimate_intrinsics_params_from_predictions(
                            pred_tracks=xyz[local_idx][None, ...],
                            pred_uv_norm=uv,
                            image_hw=image_hw,
                        )
                        intr = _enforce_shared_focal_intrinsics(intr)
                        k_seq[frame_global] = np.array(
                            [[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]],
                            dtype=np.float32,
                        )
                        intr_valid[frame_global] = True
                    xyz_global[frame_global] = xyz[local_idx]
                    vis_global[frame_global] = vis[local_idx]
                    conf_global[frame_global] = conf[local_idx]

                if predict_extrinsics:
                    local_ref = 0
                    t_chunk_cam = np.tile(np.eye(4, dtype=np.float32)[None], (local_target_ids.shape[0], 1, 1))
                    for local_tgt in local_target_ids.tolist():
                        if int(local_tgt) == 0:
                            continue
                        pose = _estimate_relative_pose_from_queries(
                            model=model,
                            video_clip=video_clip,
                            aspect_ratio=aspect_tensor,
                            memory=memory,
                            query_uv_norm=coarse_uv,
                            frame_i_local=local_ref,
                            frame_j_local=int(local_tgt),
                            query_chunk_size=camera_query_chunk_size,
                        )
                        if pose is not None:
                            try:
                                t_chunk_cam[int(local_tgt)] = np.linalg.inv(pose).astype(np.float32)
                            except np.linalg.LinAlgError:
                                pass
                    t_global_cam = _apply_sim3_to_pose_seq(t_chunk_cam, scale, rot, trans)
                    t_ref0_cam[start:end] = t_global_cam[: end - start]
                    pose_valid[start:end] = np.isfinite(t_global_cam[:, :3, :]).all(axis=(1, 2))
                    pose_valid[0] = True

    return {
        "K": k_seq.astype(np.float32),
        "T_ref0_cam": t_ref0_cam.astype(np.float32),
        "valid_intrinsics": intr_valid.astype(np.bool_),
        "valid_extrinsics": pose_valid.astype(np.bool_),
    }


def _export_video_from_frames(*, video_rgb: np.ndarray, fps: float, dst_video: Path) -> tuple[str, str]:
    poster_name = "video_poster.jpg"
    poster_path = dst_video.parent / poster_name
    import cv2

    first_frame_rgb = np.asarray(video_rgb[0], dtype=np.uint8)
    cv2.imwrite(str(poster_path), first_frame_rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    temp_video = dst_video.parent / "_tmp_input_video.mp4"
    h, w = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), float(max(fps, 1.0)), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {temp_video}")
    try:
        for frame_rgb in video_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(temp_video), "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst_video),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        shutil.copy2(temp_video, dst_video)
    finally:
        if temp_video.exists():
            temp_video.unlink()
    return dst_video.name, poster_name


def _export_demo_data(
    *,
    model: torch.nn.Module,
    video_rgb: np.ndarray,
    video_model_rgb: np.ndarray,
    point_query_uv_px: np.ndarray,
    point_query_chunk_size: int,
    track_query_chunk_size: int,
    track_selection: str,
    track_max_points: int,
    track_min_visible_frames: int,
    track_query_uv_px: np.ndarray | None = None,
    track_query_t_src: np.ndarray | None = None,
    camera_data: dict[str, np.ndarray] | None = None,
    predicted_camera_data: dict[str, np.ndarray] | None = None,
    point_dynamic_mask_thw: np.ndarray | None = None,
    suppress_depth_boundary_tracks: bool = True,
    depth_boundary_rel_thresh: float = 0.12,
    depth_boundary_abs_thresh: float = 0.20,
    depth_boundary_dilate: int = 1,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    clip_frames = _model_clip_frames(model)
    h0, w0 = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    hm, wm = int(video_model_rgb.shape[1]), int(video_model_rgb.shape[2])

    aspect_value = np.asarray([[float(wm) / float(max(1, hm))]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)
    video_tensor = torch.from_numpy(video_model_rgb).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0

    point_query_uv_norm = point_query_uv_px.copy()
    point_query_uv_norm[:, 0] /= float(max(w0 - 1, 1))
    point_query_uv_norm[:, 1] /= float(max(h0 - 1, 1))

    num_points = int(point_query_uv_px.shape[0])

    points_xyz_ref0 = np.full((num_frames, num_points, 3), np.nan, dtype=np.float32)
    points_vis = np.zeros((num_frames, num_points), dtype=bool)
    points_uv_px = np.tile(point_query_uv_px[None, :, :], (num_frames, 1, 1)).astype(np.float32)
    points_conf = np.full((num_frames, num_points), np.nan, dtype=np.float32)
    points_rgb = np.zeros((num_frames, num_points, 3), dtype=np.uint8)
    points_xyz_ref0, points_vis, points_conf, _ = _infer_point_cloud_ref0(
        model=model,
        video_model_rgb=video_model_rgb,
        point_query_uv_norm=point_query_uv_norm,
        query_chunk_size=point_query_chunk_size,
        umeyama_slide_window=bool(umeyama_slide_window),
    )

    points_rgb = _sample_rgb_from_uv_sequence(video_rgb=video_rgb, uv_px=points_uv_px)
    allowed_track_mask = np.ones((num_points,), dtype=bool)
    if bool(suppress_depth_boundary_tracks):
        allowed_track_mask = _compute_non_boundary_candidate_mask(
            query_uv_px=point_query_uv_px,
            xyz_ref0_frame0=points_xyz_ref0[0],
            visibility_frame0=points_vis[0],
            rel_thresh=float(depth_boundary_rel_thresh),
            abs_thresh=float(depth_boundary_abs_thresh),
            dilate_radius=int(depth_boundary_dilate),
        )
    if point_dynamic_mask_thw is not None:
        point_is_dynamic = _sample_bool_mask_from_uv_sequence(point_dynamic_mask_thw, points_uv_px)
        point_motion_scores = np.zeros((num_points,), dtype=np.float32)
    else:
        point_motion_scores, point_visible_counts = _compute_point_motion_scores(
            xyz_ref0=points_xyz_ref0,
            visibility=points_vis,
            confidence=points_conf,
        )
        dynamic_threshold = float(np.nanpercentile(point_motion_scores, 80)) if np.any(point_motion_scores > 0) else np.inf
        point_is_dynamic = (point_motion_scores >= dynamic_threshold) & (point_visible_counts >= max(2, int(track_min_visible_frames)))

    if track_selection == "motion":
        if point_dynamic_mask_thw is not None and int(point_dynamic_mask_thw.shape[0]) > 0:
            track_indices = _select_dynamic_interior_track_queries(
                query_uv_px=point_query_uv_px,
                dynamic_mask_hw=point_dynamic_mask_thw[0],
                max_tracks=int(track_max_points),
                allowed_mask=allowed_track_mask,
            )
            if track_indices.size <= 0:
                track_indices = _select_motion_tracks(
                    query_uv_px=point_query_uv_px,
                    xyz_ref0=points_xyz_ref0,
                    visibility=points_vis,
                    confidence=points_conf,
                    max_tracks=int(track_max_points),
                    min_visible_frames=int(track_min_visible_frames),
                    allowed_mask=allowed_track_mask,
                )
        else:
            track_indices = _select_motion_tracks(
                query_uv_px=point_query_uv_px,
                xyz_ref0=points_xyz_ref0,
                visibility=points_vis,
                confidence=points_conf,
                max_tracks=int(track_max_points),
                min_visible_frames=int(track_min_visible_frames),
                allowed_mask=allowed_track_mask,
            )
        track_query_uv_px = point_query_uv_px[track_indices]
    else:
        if track_query_uv_px is None:
            raise ValueError("track_query_uv_px is required when track_selection='grid'.")
        if track_query_t_src is not None:
            track_query_t_src = np.asarray(track_query_t_src, dtype=np.int64).reshape(-1)
            if track_query_t_src.shape[0] != int(track_query_uv_px.shape[0]):
                raise ValueError(
                    f"track_query_t_src must have shape [{int(track_query_uv_px.shape[0])}], got {track_query_t_src.shape}"
                )

    track_query_uv_norm = track_query_uv_px.copy()
    track_query_uv_norm[:, 0] /= float(max(w0 - 1, 1))
    track_query_uv_norm[:, 1] /= float(max(h0 - 1, 1))
    num_tracks = int(track_query_uv_px.shape[0])
    if track_query_t_src is None:
        track_query_t_src = np.zeros((num_tracks,), dtype=np.int64)
    else:
        track_query_t_src = np.asarray(track_query_t_src, dtype=np.int64).reshape(num_tracks)
    track_payload = _infer_tracks(
        model=model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=track_query_uv_norm.astype(np.float32),
        query_chunk_size=track_query_chunk_size,
        query_src_indices_global=track_query_t_src,
        umeyama_slide_window=bool(umeyama_slide_window),
        umeyama_slide_window_dense=bool(umeyama_slide_window_dense),
    )
    tracks_xyz_ref0 = np.asarray(track_payload["tracks_xyz_ref0"], dtype=np.float32)
    tracks_uv_px = np.asarray(track_payload["tracks_uv_norm"], dtype=np.float32)
    tracks_uv_px[..., 0] *= float(max(w0 - 1, 1))
    tracks_uv_px[..., 1] *= float(max(h0 - 1, 1))
    tracks_vis = np.asarray(track_payload["tracks_visibility"], dtype=bool)
    tracks_conf = np.asarray(track_payload["tracks_confidence"], dtype=np.float32)

    valid_xyz = np.isfinite(points_xyz_ref0).all(axis=-1) & points_vis
    if np.any(valid_xyz):
        flat = points_xyz_ref0[valid_xyz]
        xyz_min = flat.min(axis=0).astype(np.float32)
        xyz_max = flat.max(axis=0).astype(np.float32)
        xyz_center = ((xyz_min + xyz_max) * 0.5).astype(np.float32)
        xyz_radius = float(np.max(xyz_max - xyz_min) * 0.55)
    else:
        xyz_min = np.zeros((3,), dtype=np.float32)
        xyz_max = np.zeros((3,), dtype=np.float32)
        xyz_center = np.zeros((3,), dtype=np.float32)
        xyz_radius = 1.0

    if camera_data is not None and "K" in camera_data:
        ref0_k = np.asarray(camera_data["K"][0], dtype=np.float32)
    elif predicted_camera_data is not None and "K" in predicted_camera_data:
        ref0_k = np.asarray(predicted_camera_data["K"][0], dtype=np.float32)
    else:
        ref0_k = _estimate_ref0_intrinsics(
            xyz_ref0_frame0=points_xyz_ref0[0],
            uv_px_frame0=point_query_uv_px,
            visibility_frame0=points_vis[0],
            image_width=int(w0),
            image_height=int(h0),
        )

    return {
        "video_width": int(w0),
        "video_height": int(h0),
        "num_frames": int(num_frames),
        "clip_frames": int(clip_frames),
        "track_query_uv_px": track_query_uv_px.astype(np.float32),
        "track_query_t_src": track_query_t_src.astype(np.int64),
        "track_xyz_ref0": tracks_xyz_ref0.astype(np.float32),
        "track_uv_px": tracks_uv_px.astype(np.float32),
        "track_visibility": tracks_vis.astype(np.bool_),
        "track_confidence": tracks_conf.astype(np.float32),
        "track_stitch_diagnostics": track_payload.get("stitch_diagnostics", {}),
        "point_query_uv_px": point_query_uv_px.astype(np.float32),
        "point_xyz_ref0": points_xyz_ref0.astype(np.float32),
        "point_visibility": points_vis.astype(np.bool_),
        "point_uv_px": points_uv_px.astype(np.float32),
        "point_confidence": points_conf.astype(np.float32),
        "point_motion_score": point_motion_scores.astype(np.float32),
        "point_is_dynamic": np.asarray(point_is_dynamic, dtype=np.bool_),
        "point_rgb": points_rgb.astype(np.uint8),
        "bounds_min": xyz_min,
        "bounds_max": xyz_max,
        "bounds_center": xyz_center,
        "bounds_radius": np.asarray([xyz_radius], dtype=np.float32),
        "ref0_K": ref0_k.astype(np.float32),
        "camera_K_seq": None if camera_data is None else np.asarray(camera_data["K"], dtype=np.float32),
        "camera_T_ref0_cam": None if camera_data is None else np.asarray(camera_data["T_ref0_cam"], dtype=np.float32),
        "pred_camera_K_seq": None if predicted_camera_data is None else np.asarray(predicted_camera_data["K"], dtype=np.float32),
        "pred_camera_T_ref0_cam": None if predicted_camera_data is None else np.asarray(predicted_camera_data["T_ref0_cam"], dtype=np.float32),
        "pred_camera_valid_intrinsics": None if predicted_camera_data is None else np.asarray(predicted_camera_data["valid_intrinsics"], dtype=np.bool_),
        "pred_camera_valid_extrinsics": None if predicted_camera_data is None else np.asarray(predicted_camera_data["valid_extrinsics"], dtype=np.bool_),
    }
