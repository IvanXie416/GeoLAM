#!/usr/bin/env python3
"""Build a vis_like_demo package from one WorldTrack npz sample."""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _resolve_device, _resize_video, _unwrap_state_dict
from src.core import build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.model import build_model
from vis.build_like_demo import (
    _build_uv_grid,
    _compute_point_motion_scores,
    _export_demo_data,
    _export_video_from_frames,
    _jsonable_float_array,
    _predict_camera_branches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a vis_like_demo package from one WorldTrack npz.")
    parser.add_argument("--config", required=True, help="Model config yaml.")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path.")
    parser.add_argument("--worldtrack-npz", required=True, help="Path to one WorldTrack npz sample.")
    parser.add_argument("--output-dir", required=True, help="Output package directory.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--point-query-chunk-size", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--camera-grid-size", type=int, default=16)
    parser.add_argument("--camera-query-chunk-size", type=int, default=1024)
    parser.add_argument("--point-grid-cols", type=int, default=64)
    parser.add_argument("--point-grid-rows", type=int, default=64)
    parser.add_argument("--point-max-points", type=int, default=4096)
    parser.add_argument("--track-max-points", type=int, default=256, help="Max GT/pred tracks stored in viewer package. <=0 keeps all.")
    parser.add_argument("--track-min-visible-frames", type=int, default=6)
    parser.add_argument("--track-viz-max-points", type=int, default=300)
    parser.add_argument("--track-trace-frames", type=int, default=8)
    parser.add_argument("--render-track-videos", action="store_true", default=True)
    parser.add_argument("--no-render-track-videos", action="store_false", dest="render_track_videos")
    parser.add_argument("--export-depth-video", action="store_true", default=True)
    parser.add_argument("--no-export-depth-video", action="store_false", dest="export_depth_video")
    parser.add_argument("--suppress-depth-boundary-tracks", action="store_true", default=True)
    parser.add_argument("--no-suppress-depth-boundary-tracks", action="store_false", dest="suppress_depth_boundary_tracks")
    parser.add_argument("--depth-boundary-rel-thresh", type=float, default=0.12)
    parser.add_argument("--depth-boundary-abs-thresh", type=float, default=0.20)
    parser.add_argument("--depth-boundary-dilate", type=int, default=1)
    parser.add_argument(
        "--umeyama-slide-window",
        "--umeyama_slide_window",
        action="store_true",
        dest="umeyama_slide_window",
        help="Use paper-style overlapped sliding windows with Umeyama Sim(3) stitching for long sequences.",
    )
    parser.add_argument(
        "--umeyama-slide-window-dense",
        "--umeyama_slide_window_dense",
        action="store_true",
        dest="umeyama_slide_window_dense",
        help="Use dense high-confidence overlap point clouds to estimate chunk Sim(3) stitching.",
    )
    return parser.parse_args()


def _decode_jpeg_rgb(frame_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(frame_bytes, np.uint8)
    image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise RuntimeError("Failed to decode JPEG frame bytes.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _project_points_to_video_frame(camera_pov_points3d: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
    pts = np.asarray(camera_pov_points3d, dtype=np.float64)
    intr = np.asarray(camera_intrinsics, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = intr[:4]
    z = pts[..., 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = (pts[..., 0] / safe_z) * fx + cx
    v = (pts[..., 1] / safe_z) * fy + cy
    return np.stack([u, v], axis=-1)


def _portable_path(path: str | Path, base_dir: Path = REPO_ROOT) -> str:
    item = Path(path)
    if not item.is_absolute():
        return item.as_posix()
    try:
        return item.relative_to(base_dir).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.relpath(item, start=base_dir)).as_posix()
    except ValueError:
        return item.name


def load_worldtrack_sequence(npz_path: Path, num_frames: int) -> dict[str, Any]:
    pack = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
    tracks_xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64)
    visibility = np.asarray(pack["visibility"], dtype=bool)
    extrinsics_w2c_raw = pack["extrinsics_w2c"] if "extrinsics_w2c" in pack.files else None

    frame_count = min(int(num_frames), int(images_jpeg_bytes.shape[0]), int(tracks_xyz_cam.shape[0]), int(visibility.shape[0]))
    if frame_count <= 0:
        raise RuntimeError(f"No usable frames in {npz_path}")

    images_jpeg_bytes = images_jpeg_bytes[:frame_count]
    tracks_xyz_cam = tracks_xyz_cam[:frame_count]
    visibility = visibility[:frame_count]
    video_rgb = np.stack([_decode_jpeg_rgb(frame_bytes) for frame_bytes in images_jpeg_bytes], axis=0)
    tracks_uv = _project_points_to_video_frame(tracks_xyz_cam, intrinsics)

    if extrinsics_w2c_raw is not None:
        extrinsics_w2c = np.asarray(extrinsics_w2c_raw, dtype=np.float64)[:frame_count]
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        extrinsics_w2c = np.asarray([extr @ first_inv for extr in extrinsics_w2c], dtype=np.float64)
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
        tracks_xyz_world = np.empty_like(tracks_xyz_cam, dtype=np.float64)
        for frame_idx in range(frame_count):
            r = extrinsics_c2w[frame_idx, :3, :3]
            t = extrinsics_c2w[frame_idx, :3, 3]
            tracks_xyz_world[frame_idx] = (r @ tracks_xyz_cam[frame_idx].T).T + t
    else:
        extrinsics_w2c = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
        tracks_xyz_world = tracks_xyz_cam.copy()

    return {
        "video_rgb": video_rgb,
        "tracks_xyz_cam": tracks_xyz_cam,
        "tracks_xyz_world": tracks_xyz_world,
        "tracks_uv": tracks_uv,
        "visibility": visibility,
        "intrinsics": intrinsics,
        "extrinsics_w2c": extrinsics_w2c,
        "video_name": npz_path.stem,
        "sequence_path": _portable_path(npz_path),
    }


def _worldtrack_gt_camera_data(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    num_frames = int(sample["video_rgb"].shape[0])
    fx, fy, cx, cy = [float(v) for v in np.asarray(sample["intrinsics"], dtype=np.float32).reshape(-1).tolist()[:4]]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    k_seq = np.tile(k[None, :, :], (num_frames, 1, 1))
    t_ref0_cam = np.linalg.inv(np.asarray(sample["extrinsics_w2c"], dtype=np.float32))
    return {"K": k_seq, "T_ref0_cam": t_ref0_cam}


def _compute_scale_factor_global(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
    gt_flat = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    pred_flat = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    gt_norm = np.linalg.norm(gt_flat, axis=-1)
    pred_norm = np.linalg.norm(pred_flat, axis=-1)
    eps = 1e-12
    return float(np.median(np.maximum(gt_norm, eps)) / max(float(np.median(np.maximum(pred_norm, eps))), eps))


def align_tracks_global(gt_tracks_world: np.ndarray, pred_tracks_ref0: np.ndarray) -> tuple[np.ndarray, float]:
    scale = _compute_scale_factor_global(gt_tracks_world, pred_tracks_ref0)
    return np.asarray(pred_tracks_ref0, dtype=np.float64) * float(scale), float(scale)


def _select_worldtrack_track_indices(
    *,
    gt_tracks_world_tq3: np.ndarray,
    visibility_tq: np.ndarray,
    max_tracks: int,
    min_visible_frames: int,
) -> np.ndarray:
    gt = np.asarray(gt_tracks_world_tq3, dtype=np.float32)
    vis = np.asarray(visibility_tq, dtype=bool)
    motion_scores, visible_counts = _compute_point_motion_scores(
        xyz_ref0=gt,
        visibility=vis,
        confidence=np.ones_like(vis, dtype=np.float32),
    )
    scores = np.full((gt.shape[1],), -np.inf, dtype=np.float32)
    for qi in range(gt.shape[1]):
        if int(visible_counts[qi]) < int(min_visible_frames):
            continue
        scores[qi] = float(motion_scores[qi]) + 0.01 * float(visible_counts[qi])
    ranked = np.argsort(scores)[::-1]
    ranked = ranked[np.isfinite(scores[ranked])]
    if int(max_tracks) > 0:
        ranked = ranked[: max(1, min(int(max_tracks), ranked.size))]
    return ranked.astype(np.int64)


def _track_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cols = []
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 1.0)
        cols.append([int(round(c * 255.0)) for c in rgb])
    return np.asarray(cols, dtype=np.uint8)


def _sample_track_subset(
    gt_tracks_world: np.ndarray,
    pred_tracks_world: np.ndarray,
    visibility_tq: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_points = int(gt_tracks_world.shape[1])
    if num_points <= int(max_points) or int(max_points) <= 0:
        idx = np.arange(num_points, dtype=np.int64)
    else:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(num_points, size=int(max_points), replace=False).astype(np.int64))
    return gt_tracks_world[:, idx], pred_tracks_world[:, idx], visibility_tq[:, idx]


def _project_world_tracks_to_uv(points_world_tq3: np.ndarray, extrinsics_w2c: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    num_frames = int(points_world_tq3.shape[0])
    fx, fy, cx, cy = [float(v) for v in np.asarray(intrinsics, dtype=np.float64).reshape(-1).tolist()[:4]]
    out = np.full((num_frames, points_world_tq3.shape[1], 2), np.nan, dtype=np.float32)
    for t in range(num_frames):
        rot = extrinsics_w2c[t, :3, :3]
        trans = extrinsics_w2c[t, :3, 3]
        cam = (rot @ points_world_tq3[t].T).T + trans
        z = cam[:, 2]
        ok = np.isfinite(cam).all(axis=-1) & (z > 1e-6)
        if not np.any(ok):
            continue
        out[t, ok, 0] = fx * (cam[ok, 0] / z[ok]) + cx
        out[t, ok, 1] = fy * (cam[ok, 1] / z[ok]) + cy
    return out


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    return np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()


def _disable_encoder_pretrain(cfg: Any) -> None:
    encoder_cfg = cfg.get_path("model.encoder", {})
    if isinstance(encoder_cfg, dict):
        pretrained_cfg = encoder_cfg.setdefault("pretrained", {})
        if isinstance(pretrained_cfg, dict):
            pretrained_cfg["enabled"] = False


def render_track_comparison_videos(
    *,
    video_rgb: np.ndarray,
    gt_tracks_world: np.ndarray,
    pred_tracks_world: np.ndarray,
    visibility_tq: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
    max_points: int = 300,
    trace_frames: int = 8,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_subset, pred_subset, vis_subset = _sample_track_subset(
        gt_tracks_world=gt_tracks_world,
        pred_tracks_world=pred_tracks_world,
        visibility_tq=visibility_tq,
        max_points=int(max_points),
    )
    gt_uv = _project_world_tracks_to_uv(gt_subset, extrinsics_w2c, intrinsics)
    pred_uv = _project_world_tracks_to_uv(pred_subset, extrinsics_w2c, intrinsics)
    colors = _track_colors(int(gt_subset.shape[1]))

    frames_2d: list[np.ndarray] = []
    for frame_idx in range(int(video_rgb.shape[0])):
        frame = np.asarray(video_rgb[frame_idx], dtype=np.uint8).copy()
        for qi in range(int(gt_subset.shape[1])):
            color = tuple(int(v) for v in colors[qi].tolist())
            for hist_idx in range(max(0, frame_idx - int(trace_frames)), frame_idx):
                if bool(vis_subset[hist_idx, qi]) and bool(vis_subset[hist_idx + 1, qi]):
                    p0 = gt_uv[hist_idx, qi]
                    p1 = gt_uv[hist_idx + 1, qi]
                    if np.isfinite(p0).all() and np.isfinite(p1).all():
                        cv2.line(frame, tuple(np.rint(p0).astype(np.int32)), tuple(np.rint(p1).astype(np.int32)), color, 1, cv2.LINE_AA)
                    p0p = pred_uv[hist_idx, qi]
                    p1p = pred_uv[hist_idx + 1, qi]
                    if np.isfinite(p0p).all() and np.isfinite(p1p).all():
                        cv2.line(frame, tuple(np.rint(p0p).astype(np.int32)), tuple(np.rint(p1p).astype(np.int32)), color, 1, cv2.LINE_AA)
            if bool(vis_subset[frame_idx, qi]):
                gt_p = gt_uv[frame_idx, qi]
                pred_p = pred_uv[frame_idx, qi]
                if np.isfinite(gt_p).all():
                    cv2.circle(frame, tuple(np.rint(gt_p).astype(np.int32)), 3, color, -1)
                if np.isfinite(pred_p).all():
                    cv2.drawMarker(frame, tuple(np.rint(pred_p).astype(np.int32)), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        frames_2d.append(frame)
    frames_2d_np = np.stack(frames_2d, axis=0)
    video_2d_name, poster_2d_name = _export_video_from_frames(
        video_rgb=frames_2d_np,
        fps=15.0,
        dst_video=output_dir / "tracks_2d_overlay.mp4",
    )

    valid = np.isfinite(gt_subset).all(axis=-1) | np.isfinite(pred_subset).all(axis=-1)
    flat = np.concatenate([gt_subset[valid], pred_subset[valid]], axis=0) if np.any(valid) else np.zeros((1, 3), dtype=np.float32)
    xyz_min = np.nanmin(flat, axis=0)
    xyz_max = np.nanmax(flat, axis=0)
    center = (xyz_min + xyz_max) * 0.5
    half_extent = max(float(np.nanmax(xyz_max - xyz_min) * 0.55), 0.5)

    frames_3d: list[np.ndarray] = []
    for frame_idx in range(int(video_rgb.shape[0])):
        fig = plt.figure(figsize=(10, 5), dpi=140)
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        for ax, title in ((ax_gt, "GT Tracks"), (ax_pred, "Pred Tracks (Global Aligned)")):
            ax.set_title(title)
            ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
            ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
            ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=24.0, azim=45.0)
        for qi in range(int(gt_subset.shape[1])):
            rgb = colors[qi].astype(np.float32) / 255.0
            t0 = max(0, frame_idx - int(trace_frames))
            gt_hist = gt_subset[t0 : frame_idx + 1, qi]
            pred_hist = pred_subset[t0 : frame_idx + 1, qi]
            gt_ok = np.isfinite(gt_hist).all(axis=-1) & vis_subset[t0 : frame_idx + 1, qi]
            pred_ok = np.isfinite(pred_hist).all(axis=-1) & vis_subset[t0 : frame_idx + 1, qi]
            if int(gt_ok.sum()) >= 2:
                pts = gt_hist[gt_ok]
                ax_gt.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=rgb, linewidth=1.2)
            if int(pred_ok.sum()) >= 2:
                pts = pred_hist[pred_ok]
                ax_pred.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=rgb, linewidth=1.2)
            if bool(vis_subset[frame_idx, qi]):
                gt_now = gt_subset[frame_idx, qi]
                pred_now = pred_subset[frame_idx, qi]
                if np.isfinite(gt_now).all():
                    ax_gt.scatter(gt_now[0], gt_now[1], gt_now[2], color=rgb, s=18)
                if np.isfinite(pred_now).all():
                    ax_pred.scatter(pred_now[0], pred_now[1], pred_now[2], color=rgb, s=18, marker="x")
        fig.tight_layout()
        frames_3d.append(_figure_to_rgb(fig))
        plt.close(fig)
    frames_3d_np = np.stack(frames_3d, axis=0)
    video_3d_name, poster_3d_name = _export_video_from_frames(
        video_rgb=frames_3d_np,
        fps=15.0,
        dst_video=output_dir / "tracks_3d.mp4",
    )
    return {
        "tracks_2d_overlay": video_2d_name,
        "tracks_2d_overlay_poster": poster_2d_name,
        "tracks_3d": video_3d_name,
        "tracks_3d_poster": poster_3d_name,
    }


def _colorize_depth_map(depth_hw: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    depth = np.asarray(depth_hw, dtype=np.float32)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return out
    norm = np.clip((depth - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap(np.rint(norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid] = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)[valid]
    return out


def export_predicted_depth_video(
    *,
    package: dict[str, Any],
    camera_data: dict[str, np.ndarray],
    output_dir: Path,
    grid_rows: int,
    grid_cols: int,
) -> dict[str, Any] | None:
    point_xyz_ref0 = np.asarray(package["point_xyz_ref0"], dtype=np.float32)
    point_vis = np.asarray(package["point_visibility"], dtype=bool)
    point_uv = np.asarray(package["point_uv_px"], dtype=np.float32)
    t_ref0_cam = np.asarray(camera_data["T_ref0_cam"], dtype=np.float32)
    if point_xyz_ref0.shape[1] <= 0:
        return None

    num_frames = int(point_xyz_ref0.shape[0])
    image_h = int(package["video_height"])
    image_w = int(package["video_width"])
    depth_raw = np.full((num_frames, image_h, image_w), np.nan, dtype=np.float32)
    coarse_depth = np.full((num_frames, int(grid_rows), int(grid_cols)), np.nan, dtype=np.float32)

    use_grid = int(grid_rows) * int(grid_cols) == int(point_xyz_ref0.shape[1])
    for frame_idx in range(num_frames):
        pose = np.linalg.inv(t_ref0_cam[frame_idx]).astype(np.float32)
        xyz = point_xyz_ref0[frame_idx]
        vis = point_vis[frame_idx] & np.isfinite(xyz).all(axis=-1)
        if not np.any(vis):
            continue
        pts = xyz[vis]
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        cam = (pose @ pts_h.T).T[:, :3]
        z = cam[:, 2]
        z = np.where(np.isfinite(z) & (z > 1e-6), z, np.nan)
        if use_grid:
            depth_grid = np.full((int(grid_rows) * int(grid_cols),), np.nan, dtype=np.float32)
            depth_grid[np.flatnonzero(vis)] = z.astype(np.float32)
            coarse_depth[frame_idx] = depth_grid.reshape(int(grid_rows), int(grid_cols))
            mask = np.isfinite(coarse_depth[frame_idx]).astype(np.float32)
            depth_up = cv2.resize(coarse_depth[frame_idx], (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            mask_up = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            depth_raw[frame_idx] = np.where(mask_up > 1e-3, depth_up, np.nan).astype(np.float32)
        else:
            uv = point_uv[frame_idx, vis]
            xy = np.rint(uv).astype(np.int32)
            inside = (
                np.isfinite(uv).all(axis=-1)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < image_w)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < image_h)
                & np.isfinite(z)
            )
            for p_xy, p_z in zip(xy[inside], z[inside], strict=False):
                x, y = int(p_xy[0]), int(p_xy[1])
                prev = depth_raw[frame_idx, y, x]
                depth_raw[frame_idx, y, x] = float(p_z) if not np.isfinite(prev) else float(min(prev, p_z))

    valid = np.isfinite(depth_raw) & (depth_raw > 0.0)
    if not np.any(valid):
        return None
    depth_vals = depth_raw[valid]
    vmin = float(np.nanpercentile(depth_vals, 5.0))
    vmax = float(np.nanpercentile(depth_vals, 95.0))
    depth_rgb = np.stack([_colorize_depth_map(depth_raw[t], vmin=vmin, vmax=vmax) for t in range(num_frames)], axis=0)
    video_name, poster_name = _export_video_from_frames(
        video_rgb=depth_rgb,
        fps=15.0,
        dst_video=output_dir / "depth_pred.mp4",
    )
    np.savez_compressed(output_dir / "depth_pred_raw.npz", depth=depth_raw)
    return {
        "video": video_name,
        "poster": poster_name,
        "raw": "assets/depth_pred_raw.npz",
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def build_worldtrack_demo_package(
    *,
    model: torch.nn.Module,
    cfg: Any,
    npz_path: Path,
    output_dir: Path,
    num_frames: int,
    fps: float,
    point_query_chunk_size: int,
    track_query_chunk_size: int,
    camera_grid_size: int,
    camera_query_chunk_size: int,
    point_grid_cols: int,
    point_grid_rows: int,
    point_max_points: int,
    track_max_points: int,
    track_min_visible_frames: int,
    render_track_videos: bool,
    export_depth_video: bool,
    track_viz_max_points: int,
    track_trace_frames: int,
    suppress_depth_boundary_tracks: bool,
    depth_boundary_rel_thresh: float,
    depth_boundary_abs_thresh: float,
    depth_boundary_dilate: int,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("vis_like_demo_worldtrack", output_dir)

    sample = load_worldtrack_sequence(npz_path=npz_path, num_frames=int(num_frames))
    video_rgb = np.asarray(sample["video_rgb"], dtype=np.uint8)
    image_size = cfg.get_path("model.input.image_size", [int(video_rgb.shape[1]), int(video_rgb.shape[2])])
    video_model_rgb = _resize_video(video_rgb, image_hw=(int(image_size[0]), int(image_size[1])))

    gt_camera_data = _worldtrack_gt_camera_data(sample)
    predicted_camera_data = _predict_camera_branches(
        model=model,
        video_model_rgb=video_model_rgb,
        image_hw=(int(video_rgb.shape[1]), int(video_rgb.shape[2])),
        camera_grid_size=int(camera_grid_size),
        camera_query_chunk_size=int(camera_query_chunk_size),
        predict_intrinsics=True,
        predict_extrinsics=True,
        umeyama_slide_window=bool(umeyama_slide_window),
        umeyama_slide_window_dense=bool(umeyama_slide_window_dense),
    )

    point_query_uv_px = _build_uv_grid(
        width=int(video_rgb.shape[2]),
        height=int(video_rgb.shape[1]),
        cols=int(point_grid_cols),
        rows=int(point_grid_rows),
        max_points=int(point_max_points),
    )

    visible_mask = np.asarray(sample["visibility"][0], dtype=bool)
    query_uv = np.asarray(sample["tracks_uv"][0, visible_mask], dtype=np.float32)
    depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible_mask, 2], dtype=np.float32)
    keep = np.isfinite(query_uv).all(axis=-1) & np.isfinite(depth0) & (depth0 > 1e-6)
    if not np.any(keep):
        raise RuntimeError(f"No valid frame-0 visible queries in {npz_path}")
    gt_tracks_world_all = np.asarray(sample["tracks_xyz_world"][:, visible_mask], dtype=np.float32)[:, keep]
    track_visibility_all = np.asarray(sample["visibility"][:, visible_mask], dtype=bool)[:, keep]
    track_query_uv_all = np.asarray(sample["tracks_uv"][0, visible_mask], dtype=np.float32)[keep]
    selected = _select_worldtrack_track_indices(
        gt_tracks_world_tq3=gt_tracks_world_all,
        visibility_tq=track_visibility_all,
        max_tracks=int(track_max_points),
        min_visible_frames=int(track_min_visible_frames),
    )
    if selected.size <= 0:
        raise RuntimeError(f"No WorldTrack queries remained after selection in {npz_path}")

    track_query_uv_px = track_query_uv_all[selected]
    track_query_t_src = np.zeros((track_query_uv_px.shape[0],), dtype=np.int64)
    gt_tracks_world = gt_tracks_world_all[:, selected]
    track_visibility = track_visibility_all[:, selected]
    track_uv_gt = np.asarray(sample["tracks_uv"][:, visible_mask], dtype=np.float32)[:, keep][:, selected]

    package = _export_demo_data(
        model=model,
        video_rgb=video_rgb,
        video_model_rgb=video_model_rgb,
        point_query_uv_px=point_query_uv_px,
        point_query_chunk_size=int(point_query_chunk_size),
        track_query_chunk_size=int(track_query_chunk_size),
        track_selection="grid",
        track_max_points=int(track_query_uv_px.shape[0]),
        track_min_visible_frames=int(track_min_visible_frames),
        track_query_uv_px=track_query_uv_px,
        track_query_t_src=track_query_t_src,
        camera_data=gt_camera_data,
        predicted_camera_data=predicted_camera_data,
        point_dynamic_mask_thw=None,
        suppress_depth_boundary_tracks=bool(suppress_depth_boundary_tracks),
        depth_boundary_rel_thresh=float(depth_boundary_rel_thresh),
        depth_boundary_abs_thresh=float(depth_boundary_abs_thresh),
        depth_boundary_dilate=int(depth_boundary_dilate),
        umeyama_slide_window=bool(umeyama_slide_window),
        umeyama_slide_window_dense=bool(umeyama_slide_window_dense),
    )

    pred_tracks_raw = np.transpose(np.asarray(package["track_xyz_ref0"], dtype=np.float64), (1, 0, 2))
    pred_tracks_aligned, scale_global = align_tracks_global(gt_tracks_world, pred_tracks_raw)
    pred_tracks_aligned_qt3 = np.transpose(pred_tracks_aligned, (1, 0, 2)).astype(np.float32)
    point_xyz_aligned = np.asarray(package["point_xyz_ref0"], dtype=np.float32) * float(scale_global)

    pred_camera_k_seq = None if package["pred_camera_K_seq"] is None else np.asarray(package["pred_camera_K_seq"], dtype=np.float32)
    pred_camera_t_ref0_cam = None
    if package["pred_camera_T_ref0_cam"] is not None:
        pred_camera_t_ref0_cam = np.asarray(package["pred_camera_T_ref0_cam"], dtype=np.float32).copy()
        pred_camera_t_ref0_cam[:, :3, 3] *= float(scale_global)

    depth_manifest = None
    if bool(export_depth_video):
        depth_manifest = export_predicted_depth_video(
            package=package,
            camera_data=gt_camera_data,
            output_dir=assets_dir,
            grid_rows=int(point_grid_rows),
            grid_cols=int(point_grid_cols),
        )

    track_video_manifest = None
    if bool(render_track_videos):
        track_video_manifest = render_track_comparison_videos(
            video_rgb=video_rgb,
            gt_tracks_world=gt_tracks_world,
            pred_tracks_world=pred_tracks_aligned,
            visibility_tq=track_visibility,
            extrinsics_w2c=np.asarray(sample["extrinsics_w2c"], dtype=np.float32),
            intrinsics=np.asarray(sample["intrinsics"], dtype=np.float32),
            output_dir=assets_dir,
            max_points=int(track_viz_max_points),
            trace_frames=int(track_trace_frames),
        )

    pred_track_vis = np.asarray(package["track_visibility"], dtype=bool)
    pred_track_conf = np.asarray(package["track_confidence"], dtype=np.float32)
    if pred_track_vis.shape != (track_query_uv_px.shape[0], video_rgb.shape[0]):
        pred_track_vis = np.transpose(pred_track_vis, (1, 0))
        pred_track_conf = np.transpose(pred_track_conf, (1, 0))

    gt_track_xyz_qt3 = np.transpose(gt_tracks_world, (1, 0, 2)).astype(np.float32)
    gt_track_uv_qt2 = np.transpose(track_uv_gt, (1, 0, 2)).astype(np.float32)
    gt_track_vis_qt = np.transpose(track_visibility, (1, 0)).astype(bool)
    gt_track_conf_qt = gt_track_vis_qt.astype(np.float32)

    meta = {
        "fps": float(fps if fps > 0.0 else 15.0),
        "numFrames": int(package["num_frames"]),
        "videoWidth": int(package["video_width"]),
        "videoHeight": int(package["video_height"]),
        "crop": {"top": 0, "bottom": int(package["video_height"])},
        "clipFrames": int(package["clip_frames"]),
        "umeyamaSlideWindow": bool(umeyama_slide_window),
        "umeyamaSlideWindowDense": bool(umeyama_slide_window_dense),
        "trackStitchDiagnostics": package.get("track_stitch_diagnostics", {}),
        "trackCount": int(package["track_query_uv_px"].shape[0]),
        "trackCountPred": int(package["track_query_uv_px"].shape[0]),
        "trackCountGt": int(track_query_uv_px.shape[0]),
        "pointCountPerFrame": int(package["point_query_uv_px"].shape[0]),
        "bounds": {
            "min": _jsonable_float_array(package["bounds_min"]),
            "max": _jsonable_float_array(package["bounds_max"]),
            "center": _jsonable_float_array(package["bounds_center"]),
            "radius": float(package["bounds_radius"][0]),
        },
        "ref0K": _jsonable_float_array(package["ref0_K"], ndigits=5),
        "camera": {
            "K": _jsonable_float_array(package["camera_K_seq"], ndigits=5),
            "TRef0Cam": _jsonable_float_array(package["camera_T_ref0_cam"], ndigits=6),
            "source": "worldtrack_gt",
        },
        "cameraPred": None if package["pred_camera_K_seq"] is None else {
            "K": _jsonable_float_array(package["pred_camera_K_seq"], ndigits=5),
            "TRef0Cam": _jsonable_float_array(package["pred_camera_T_ref0_cam"], ndigits=6),
            "validIntrinsics": package["pred_camera_valid_intrinsics"].astype(np.int32).tolist(),
            "validExtrinsics": package["pred_camera_valid_extrinsics"].astype(np.int32).tolist(),
            "source": "d4rt_queries",
        },
        "depthPred": None if depth_manifest is None else {
            "video": f"assets/{depth_manifest['video']}",
            "poster": f"assets/{depth_manifest['poster']}",
            "raw": depth_manifest["raw"],
            "vmin": float(depth_manifest["vmin"]),
            "vmax": float(depth_manifest["vmax"]),
            "source": "d4rt_dense_point_queries",
        },
        "worldtrack": {
            "npz": sample["sequence_path"],
            "sequencePath": sample["sequence_path"],
            "trackQuerySource": "frame0_visible_queries",
            "trackAlignment": {"type": "global_median_scale", "scale": float(scale_global)},
            "pointAlignment": {"type": "global_median_scale", "scale": float(scale_global)},
            "predCameraAlignment": {"type": "global_median_scale_translation", "scale": float(scale_global)},
            "gtTrackSource": "worldtrack_ref0_world_tracks",
            "predTrackSource": "d4rt_query_tracks_global_aligned",
        },
    }

    data_json = {
        "meta": meta,
        "tracks": {
            "queryUvPx": _jsonable_float_array(package["track_query_uv_px"], ndigits=3),
            "queryTSrc": package["track_query_t_src"].astype(np.int32).tolist(),
            "xyzRef0": _jsonable_float_array(pred_tracks_aligned_qt3, ndigits=5),
            "uvPx": _jsonable_float_array(package["track_uv_px"], ndigits=3),
            "visibility": pred_track_vis.astype(np.int32).tolist(),
            "confidence": _jsonable_float_array(pred_track_conf, ndigits=4),
        },
        "tracksGt": {
            "queryUvPx": _jsonable_float_array(track_query_uv_px, ndigits=3),
            "queryTSrc": track_query_t_src.astype(np.int32).tolist(),
            "xyzRef0": _jsonable_float_array(gt_track_xyz_qt3, ndigits=5),
            "uvPx": _jsonable_float_array(gt_track_uv_qt2, ndigits=3),
            "visibility": gt_track_vis_qt.astype(np.int32).tolist(),
            "confidence": _jsonable_float_array(gt_track_conf_qt, ndigits=4),
        },
        "tracksRaw": {
            "xyzRef0": _jsonable_float_array(package["track_xyz_ref0"], ndigits=5),
            "uvPx": _jsonable_float_array(package["track_uv_px"], ndigits=3),
            "visibility": package["track_visibility"].astype(np.int32).tolist(),
            "confidence": _jsonable_float_array(package["track_confidence"], ndigits=4),
        },
        "points": {
            "queryUvPx": _jsonable_float_array(package["point_query_uv_px"], ndigits=3),
            "xyzRef0": _jsonable_float_array(point_xyz_aligned, ndigits=5),
            "visibility": package["point_visibility"].astype(np.int32).tolist(),
            "rgb": package["point_rgb"].astype(np.int32).tolist(),
            "uvPx": _jsonable_float_array(package["point_uv_px"], ndigits=3),
            "confidence": _jsonable_float_array(package["point_confidence"], ndigits=4),
            "motionScore": _jsonable_float_array(package["point_motion_score"], ndigits=5),
            "isDynamic": package["point_is_dynamic"].astype(np.int32).tolist(),
        },
        "pointsRaw": {
            "xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5),
        },
    }

    if pred_camera_k_seq is not None and pred_camera_t_ref0_cam is not None:
        meta["cameraPred"] = {
            "K": _jsonable_float_array(pred_camera_k_seq, ndigits=5),
            "TRef0Cam": _jsonable_float_array(pred_camera_t_ref0_cam, ndigits=6),
            "validIntrinsics": package["pred_camera_valid_intrinsics"].astype(np.int32).tolist(),
            "validExtrinsics": package["pred_camera_valid_extrinsics"].astype(np.int32).tolist(),
            "source": "d4rt_queries",
        }

    (assets_dir / "demo_data.json").write_text(json.dumps(data_json, ensure_ascii=False), encoding="utf-8")
    video_copy_name, poster_name = _export_video_from_frames(
        video_rgb=video_rgb,
        fps=float(fps if fps > 0.0 else 15.0),
        dst_video=assets_dir / "input_video.mp4",
    )

    manifest = {
        "worldtrack_npz": sample["sequence_path"],
        "video_copy": f"assets/{video_copy_name}",
        "video_poster": f"assets/{poster_name}",
        "data_json": "assets/demo_data.json",
        "viewer": "viser",
    }
    if depth_manifest is not None:
        manifest["depth_pred_video"] = f"assets/{depth_manifest['video']}"
        manifest["depth_pred_poster"] = f"assets/{depth_manifest['poster']}"
        manifest["depth_pred_raw"] = depth_manifest["raw"]
    if track_video_manifest is not None:
        manifest["tracks_2d_overlay"] = f"assets/{track_video_manifest['tracks_2d_overlay']}"
        manifest["tracks_2d_overlay_poster"] = f"assets/{track_video_manifest['tracks_2d_overlay_poster']}"
        manifest["tracks_3d"] = f"assets/{track_video_manifest['tracks_3d']}"
        manifest["tracks_3d_poster"] = f"assets/{track_video_manifest['tracks_3d_poster']}"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved WorldTrack demo package to %s", output_dir)
    return {
        "sample": sample,
        "package": package,
        "manifest": manifest,
        "pred_tracks_aligned_tq3": pred_tracks_aligned.astype(np.float32),
        "gt_tracks_world_tq3": gt_tracks_world.astype(np.float32),
        "track_visibility_tq": track_visibility.astype(bool),
        "scale_global": float(scale_global),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_config(args.config)
    _disable_encoder_pretrain(cfg)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    device = _resolve_device(args.device)

    npz_path = Path(args.worldtrack_npz)
    ckpt_path = Path(args.ckpt_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    model = build_model(cfg["model"]).eval().to(device)
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    build_worldtrack_demo_package(
        model=model,
        cfg=cfg,
        npz_path=npz_path,
        output_dir=output_dir,
        num_frames=int(args.num_frames),
        fps=float(args.fps),
        point_query_chunk_size=int(args.point_query_chunk_size),
        track_query_chunk_size=int(args.query_chunk_size),
        camera_grid_size=int(args.camera_grid_size),
        camera_query_chunk_size=int(args.camera_query_chunk_size),
        point_grid_cols=int(args.point_grid_cols),
        point_grid_rows=int(args.point_grid_rows),
        point_max_points=int(args.point_max_points),
        track_max_points=int(args.track_max_points),
        track_min_visible_frames=int(args.track_min_visible_frames),
        render_track_videos=bool(args.render_track_videos),
        export_depth_video=bool(args.export_depth_video),
        track_viz_max_points=int(args.track_viz_max_points),
        track_trace_frames=int(args.track_trace_frames),
        suppress_depth_boundary_tracks=bool(args.suppress_depth_boundary_tracks),
        depth_boundary_rel_thresh=float(args.depth_boundary_rel_thresh),
        depth_boundary_abs_thresh=float(args.depth_boundary_abs_thresh),
        depth_boundary_dilate=int(args.depth_boundary_dilate),
        umeyama_slide_window=bool(args.umeyama_slide_window),
        umeyama_slide_window_dense=bool(args.umeyama_slide_window_dense),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
