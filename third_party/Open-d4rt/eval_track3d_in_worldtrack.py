#!/usr/bin/env python3
"""Evaluate D4RT checkpoints on the WorldTrack 3D tracking benchmark.

Protocol alignment target:
- First-frame visible queries only.
- Predict query trajectories in the frame-0 reference coordinate system.
- WorldTrack metrics following St4RTrack's actual evaluator:
  - APD = avg_pts_global
  - EPE = epe_global
  - alignment = global median scale alignment
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from infer_track_3d import _infer_tracks, _resolve_device, _resize_video, _unwrap_state_dict
from src.core import build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.model import build_model


PIXEL_TO_FIXED_METRIC_THRESH: dict[int, float] = {
    1: 0.1,
    2: 0.3,
    4: 0.5,
    8: 1.0,
}


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


def load_worldtrack_sequence(npz_path: Path, num_frames: int) -> dict[str, Any]:
    pack = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
    tracks_xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64)
    visibility = np.asarray(pack["visibility"], dtype=bool)
    extrinsics_w2c_raw = pack["extrinsics_w2c"] if "extrinsics_w2c" in pack.files else None

    frame_count = min(
        int(num_frames),
        int(images_jpeg_bytes.shape[0]),
        int(tracks_xyz_cam.shape[0]),
        int(visibility.shape[0]),
    )
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
            rot = extrinsics_c2w[frame_idx, :3, :3]
            trans = extrinsics_c2w[frame_idx, :3, 3]
            tracks_xyz_world[frame_idx] = (rot @ tracks_xyz_cam[frame_idx].T).T + trans
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
        "sequence_path": str(npz_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate D4RT on WorldTrack using the St4RTrack tracking protocol.")
    parser.add_argument("--model-config", required=True, help="Model config yaml.")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path.")
    parser.add_argument("--data-root", default="data/worldtrack_release", help="WorldTrack root directory.")
    parser.add_argument(
        "--subsets",
        default="adt_mini,po_mini,pstudio_mini,ds_mini",
        help="Comma-separated WorldTrack subsets.",
    )
    parser.add_argument("--output-dir", default="tmp/eval/worldtrack_d4rt")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1000000,
        help="Frames per sequence to evaluate. Default is a large cap, so full WorldTrack cases are used.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--limit-seqs", type=int, default=0, help="Optional cap per subset. <=0 disables.")
    parser.add_argument("--save-per-sequence", action="store_true", help="Write per-sequence metric JSON files.")
    return parser.parse_args()


def _compute_scale_factor_global(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
    gt_flat = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    pred_flat = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pred_flat).all(axis=-1)
    if not np.any(finite):
        return 1.0
    gt_norm = np.linalg.norm(gt_flat[finite], axis=-1)
    pred_norm = np.linalg.norm(pred_flat[finite], axis=-1)
    eps = 1e-12
    if gt_norm.size <= 0 or pred_norm.size <= 0:
        return 1.0
    gt_norm = np.maximum(gt_norm, eps)
    pred_norm = np.maximum(pred_norm, eps)
    return float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))


def _scale_per_trajectory(gt_points: np.ndarray, pred_points: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    out = pred.copy()
    eps = 1e-12
    for idx in range(gt.shape[1]):
        finite = np.isfinite(gt[:, idx]).all(axis=-1) & np.isfinite(pred[:, idx]).all(axis=-1)
        if not np.any(finite):
            continue
        gt_norm = np.linalg.norm(gt[finite, idx], axis=-1)
        pred_norm = np.linalg.norm(pred[finite, idx], axis=-1)
        if gt_norm.size <= 0 or pred_norm.size <= 0:
            continue
        gt_norm = np.maximum(gt_norm, eps)
        pred_norm = np.maximum(pred_norm, eps)
        scale = float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))
        out[:, idx] = pred[:, idx] * scale
    return out


def _estimate_sim3_closed_form(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    centroid_src = src.mean(axis=0, keepdims=True)
    centroid_dst = dst.mean(axis=0, keepdims=True)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    h = src_centered.T @ dst_centered
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    var_src = float((src_centered**2).sum())
    scale = float(np.sum(s) / max(var_src, 1e-12))
    t = centroid_dst[0] - scale * (r @ centroid_src[0])
    return scale, r, t


def _estimate_sim3_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 1000,
    inlier_threshold: float = 0.05,
) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for Sim3 estimation.")
    best_count = -1
    best_model: tuple[float, np.ndarray, np.ndarray] | None = None
    best_mask: np.ndarray | None = None
    rng = np.random.default_rng(0)
    for _ in range(int(iterations)):
        subset_idx = rng.choice(src.shape[0], size=3, replace=False)
        try:
            scale, rot, trans = _estimate_sim3_closed_form(src[subset_idx], dst[subset_idx])
        except np.linalg.LinAlgError:
            continue
        transformed = scale * (rot @ src.T).T + trans
        dists = np.linalg.norm(transformed - dst, axis=1)
        mask = dists < float(inlier_threshold)
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_model = (scale, rot, trans)
            best_mask = mask
    if best_model is None:
        return _estimate_sim3_closed_form(src, dst)
    if best_mask is not None and int(best_mask.sum()) >= 3:
        return _estimate_sim3_closed_form(src[best_mask], dst[best_mask])
    return best_model


def _finite_correspondence_pairs(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src_flat = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_flat = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src_flat).all(axis=-1) & np.isfinite(dst_flat).all(axis=-1)
    return src_flat[finite], dst_flat[finite]


def _compute_average_pts_within_thresh(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    scaling: str = "global",
    compute_epe: bool = True,
) -> tuple[float, np.ndarray, dict[int, float], tuple[float | None, np.ndarray | None, np.ndarray | None], float]:
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    params: tuple[float | None, np.ndarray | None, np.ndarray | None]
    if scaling == "global":
        scale = _compute_scale_factor_global(gt, pred)
        pred_aligned = pred * scale
        params = (scale, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64))
    elif scaling == "per_traj":
        pred_aligned = _scale_per_trajectory(gt, pred)
        params = (1.0, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64))
    elif scaling == "sim3_closed":
        src_fit, dst_fit = _finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
            params = (None, None, None)
        else:
            scale, rot, trans = _estimate_sim3_closed_form(src_fit, dst_fit)
            src = pred.reshape(-1, 3)
            finite = np.isfinite(src).all(axis=-1)
            pred_aligned = np.full_like(src, np.nan, dtype=np.float64)
            pred_aligned[finite] = (scale * (rot @ src[finite].T)).T + trans
            pred_aligned = pred_aligned.reshape(gt.shape)
            params = (scale, rot, trans)
    elif scaling == "sim3":
        src_fit, dst_fit = _finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
            params = (None, None, None)
        else:
            if src_fit.shape[0] > 16384:
                rng = np.random.default_rng(0)
                pick = rng.choice(src_fit.shape[0], size=16384, replace=False)
                src_sample = src_fit[pick]
                dst_sample = dst_fit[pick]
            else:
                src_sample = src_fit
                dst_sample = dst_fit
            scale, rot, trans = _estimate_sim3_ransac(src_sample, dst_sample)
            src = pred.reshape(-1, 3)
            finite = np.isfinite(src).all(axis=-1)
            pred_aligned = np.full_like(src, np.nan, dtype=np.float64)
            pred_aligned[finite] = (scale * (rot @ src[finite].T)).T + trans
            pred_aligned = pred_aligned.reshape(gt.shape)
            params = (scale, rot, trans)
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    dists = np.linalg.norm(pred_aligned - gt, axis=-1)
    total_points = int(np.isfinite(dists).sum())
    fractions: dict[int, float] = {}
    for thr_key, fixed_threshold in PIXEL_TO_FIXED_METRIC_THRESH.items():
        within_dist = np.isfinite(dists) & (dists <= float(fixed_threshold))
        fractions[int(thr_key)] = float(np.sum(within_dist) / max(total_points, 1))
    avg_pts = float(np.mean(list(fractions.values()))) if fractions else float("nan")
    epe = float(np.mean(dists[np.isfinite(dists)])) if compute_epe and np.any(np.isfinite(dists)) else float("inf")
    return avg_pts, pred_aligned, fractions, params, epe


def _metrics_for_sequence(
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    compute_dyn: bool = True,
) -> dict[str, Any]:
    avg_pts_global, _, fractions_global, _, epe_global = _compute_average_pts_within_thresh(
        gt_tracks_world,
        pred_tracks_ref0,
        scaling="global",
        compute_epe=True,
    )
    avg_pts_pertraj, _, fractions_pertraj, _, epe_pertraj = _compute_average_pts_within_thresh(
        gt_tracks_world,
        pred_tracks_ref0,
        scaling="per_traj",
        compute_epe=True,
    )
    avg_pts_sim3, _, fractions_sim3, _, epe_sim3 = _compute_average_pts_within_thresh(
        gt_tracks_world,
        pred_tracks_ref0,
        scaling="sim3",
        compute_epe=True,
    )
    avg_pts_sim3_closed, _, fractions_sim3_closed, _, epe_sim3_closed = _compute_average_pts_within_thresh(
        gt_tracks_world,
        pred_tracks_ref0,
        scaling="sim3_closed",
        compute_epe=True,
    )

    avg_pts_global_dyn = float("nan")
    epe_global_dyn = float("nan")
    avg_pts_sim3_closed_dyn = float("nan")
    epe_sim3_closed_dyn = float("nan")
    fractions_global_dyn: dict[int, float] = {}
    fractions_sim3_closed_dyn: dict[int, float] = {}
    dyn_fraction = float("nan")
    dyn_count = 0

    if compute_dyn and gt_tracks_world.shape[0] >= 2:
        total_motion = gt_tracks_world[1:] - gt_tracks_world[:-1]
        total_motion_norm = np.linalg.norm(total_motion, axis=-1).sum(axis=0)
        dyn_mask = total_motion_norm > 0.01
        dyn_count = int(dyn_mask.sum())
        dyn_fraction = float(dyn_mask.mean()) if dyn_mask.size > 0 else float("nan")
        if dyn_count > 0:
            avg_pts_global_dyn, _, fractions_global_dyn, _, epe_global_dyn = _compute_average_pts_within_thresh(
                gt_tracks_world[:, dyn_mask],
                pred_tracks_ref0[:, dyn_mask],
                scaling="global",
                compute_epe=True,
            )
            avg_pts_sim3_closed_dyn, _, fractions_sim3_closed_dyn, _, epe_sim3_closed_dyn = _compute_average_pts_within_thresh(
                gt_tracks_world[:, dyn_mask],
                pred_tracks_ref0[:, dyn_mask],
                scaling="sim3_closed",
                compute_epe=True,
            )

    return {
        "avg_pts_global": float(avg_pts_global),
        "avg_pts_pertraj": float(avg_pts_pertraj),
        "avg_pts_sim3": float(avg_pts_sim3),
        "avg_pts_sim3_closed": float(avg_pts_sim3_closed),
        "epe_global": float(epe_global),
        "epe_pertraj": float(epe_pertraj),
        "epe_sim3": float(epe_sim3),
        "epe_sim3_closed": float(epe_sim3_closed),
        "avg_pts_global_dyn": float(avg_pts_global_dyn),
        "epe_global_dyn": float(epe_global_dyn),
        "avg_pts_sim3_closed_dyn": float(avg_pts_sim3_closed_dyn),
        "epe_sim3_closed_dyn": float(epe_sim3_closed_dyn),
        "fractions_global": fractions_global,
        "fractions_pertraj": fractions_pertraj,
        "fractions_sim3": fractions_sim3,
        "fractions_sim3_closed": fractions_sim3_closed,
        "fractions_global_dyn": fractions_global_dyn,
        "fractions_sim3_closed_dyn": fractions_sim3_closed_dyn,
        "dyn_fraction": float(dyn_fraction),
        "dyn_count": int(dyn_count),
        "num_queries": int(gt_tracks_world.shape[1]),
    }


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "avg_pts_global",
        "avg_pts_pertraj",
        "avg_pts_sim3",
        "avg_pts_sim3_closed",
        "epe_global",
        "epe_pertraj",
        "epe_sim3",
        "epe_sim3_closed",
        "avg_pts_global_dyn",
        "epe_global_dyn",
        "avg_pts_sim3_closed_dyn",
        "epe_sim3_closed_dyn",
        "dyn_fraction",
    ]
    fraction_keys = [
        "fractions_global",
        "fractions_pertraj",
        "fractions_sim3",
        "fractions_sim3_closed",
        "fractions_global_dyn",
        "fractions_sim3_closed_dyn",
    ]
    summary: dict[str, Any] = {"num_sequences": int(len(results))}
    for key in scalar_keys:
        values = [float(item[key]) for item in results if np.isfinite(float(item.get(key, float("nan"))))]
        summary[key] = float(np.mean(values)) if values else float("nan")
    summary["total_queries"] = int(sum(int(item.get("num_queries", 0)) for item in results))
    summary["total_dynamic_queries"] = int(sum(int(item.get("dyn_count", 0)) for item in results))

    for frac_key in fraction_keys:
        agg: dict[int, list[float]] = defaultdict(list)
        for item in results:
            payload = item.get(frac_key, {})
            if not isinstance(payload, dict):
                continue
            for thr, value in payload.items():
                if np.isfinite(float(value)):
                    agg[int(thr)].append(float(value))
        summary[frac_key] = {int(thr): float(np.mean(vals)) for thr, vals in sorted(agg.items())}
    return summary


def _format_subset_summary(subset: str, summary: dict[str, Any]) -> str:
    return (
        f"{subset}: "
        f"APD(global)={summary.get('avg_pts_global', float('nan')):.4f} "
        f"EPE(global)={summary.get('epe_global', float('nan')):.4f} "
        f"APD(global,dyn)={summary.get('avg_pts_global_dyn', float('nan')):.4f} "
        f"EPE(global,dyn)={summary.get('epe_global_dyn', float('nan')):.4f} "
        f"queries={int(summary.get('total_queries', 0))}"
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("eval_track3d_in_worldtrack", output_dir)

    cfg = load_yaml_config(args.model_config)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(cfg["model"]).eval()
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint %s", ckpt_path)
    logger.info("Missing keys: %d  Unexpected keys: %d", len(load_result.missing_keys), len(load_result.unexpected_keys))
    device = _resolve_device(args.device)
    model.to(device).eval()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"WorldTrack root not found: {data_root}")

    subsets = [item.strip() for item in str(args.subsets).split(",") if item.strip()]
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    model_h = int(image_size[0])
    model_w = int(image_size[1])

    all_summary: dict[str, Any] = {
        "inputs": {
            "model_config": str(args.model_config),
            "ckpt_path": str(ckpt_path),
            "data_root": str(data_root),
            "subsets": subsets,
            "num_frames": int(args.num_frames),
            "query_chunk_size": int(args.query_chunk_size),
        },
        "subsets": {},
    }

    for subset in subsets:
        subset_dir = data_root / subset
        if not subset_dir.exists():
            logger.warning("Skipping missing subset directory: %s", subset_dir)
            continue
        seq_paths = sorted(subset_dir.glob("*.npz"))
        if int(args.limit_seqs) > 0:
            seq_paths = seq_paths[: int(args.limit_seqs)]
        if not seq_paths:
            logger.warning("No sequences found in %s", subset_dir)
            continue

        logger.info("Evaluating subset=%s sequences=%d", subset, len(seq_paths))
        subset_results: list[dict[str, Any]] = []
        subset_out_dir = output_dir / subset
        subset_out_dir.mkdir(parents=True, exist_ok=True)

        for seq_path in seq_paths:
            sample = load_worldtrack_sequence(seq_path, num_frames=int(args.num_frames))
            video_rgb = sample["video_rgb"]
            original_h = int(video_rgb.shape[1])
            original_w = int(video_rgb.shape[2])
            video_model_rgb = _resize_video(video_rgb, image_hw=(model_h, model_w))

            visible_mask = np.asarray(sample["visibility"][0], dtype=bool)
            if not np.any(visible_mask):
                logger.warning("Skipping %s because frame-0 has no visible queries", seq_path.name)
                continue

            query_uv = np.asarray(sample["tracks_uv"][0, visible_mask], dtype=np.float64)
            finite_mask = np.isfinite(query_uv).all(axis=-1)
            depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible_mask, 2], dtype=np.float64)
            finite_mask &= np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
            if not np.any(finite_mask):
                logger.warning("Skipping %s because no finite frame-0 visible query UVs remain", seq_path.name)
                continue

            query_uv = query_uv[finite_mask]
            gt_tracks_world = np.asarray(sample["tracks_xyz_world"][:, visible_mask], dtype=np.float64)[:, finite_mask]
            query_uv_norm = query_uv.astype(np.float32)
            query_uv_norm[:, 0] /= float(max(original_w - 1, 1))
            query_uv_norm[:, 1] /= float(max(original_h - 1, 1))
            query_uv_norm = np.clip(query_uv_norm, 0.0, 1.0)

            pred_payload = _infer_tracks(
                model=model,
                video_model_rgb=video_model_rgb,
                query_uv_norm=query_uv_norm,
                query_chunk_size=int(args.query_chunk_size),
            )
            pred_tracks_ref0 = np.asarray(pred_payload["tracks_xyz_ref0"], dtype=np.float64).transpose(1, 0, 2)
            metrics = _metrics_for_sequence(gt_tracks_world=gt_tracks_world, pred_tracks_ref0=pred_tracks_ref0, compute_dyn=True)
            _, pred_tracks_aligned_global, _, _, _ = _compute_average_pts_within_thresh(
                gt_tracks_world,
                pred_tracks_ref0,
                scaling="global",
                compute_epe=True,
            )
            metrics.update(
                {
                    "video_name": sample["video_name"],
                    "sequence_path": str(seq_path),
                    "clip_frames": int(pred_payload["clip_frames"]),
                    "model_image_size": [int(model_h), int(model_w)],
                    "original_image_size": [int(original_h), int(original_w)],
                }
            )
            stitch_diagnostics = pred_payload.get("stitch_diagnostics", {})
            if isinstance(stitch_diagnostics, dict):
                chunks = stitch_diagnostics.get("chunks", [])
                if isinstance(chunks, list):
                    success_chunks = [
                        item for item in chunks
                        if isinstance(item, dict) and int(item.get("window_idx", 0)) > 0 and bool(item.get("sim3_success", False))
                    ]
                    failed_chunks = [
                        item for item in chunks
                        if isinstance(item, dict) and int(item.get("window_idx", 0)) > 0 and not bool(item.get("sim3_success", False))
                    ]
                else:
                    chunks = []
                    success_chunks = []
                    failed_chunks = []
                metrics.update(
                    {
                        "stitch_mode": str(stitch_diagnostics.get("mode", "")),
                        "stitch_num_chunks": int(len(chunks)),
                        "stitch_success_chunks": int(len(success_chunks)),
                        "stitch_failed_chunks": int(len(failed_chunks)),
                        "stitch_diagnostics": stitch_diagnostics,
                    }
            )
            subset_results.append(metrics)
            logger.info(
                "subset=%s seq=%s APD(global)=%.4f EPE(global)=%.4f queries=%d clip_frames=%d",
                subset,
                sample["video_name"],
                metrics["avg_pts_global"],
                metrics["epe_global"],
                metrics["num_queries"],
                metrics["clip_frames"],
            )
            if bool(args.save_per_sequence):
                per_seq_path = subset_out_dir / f"{sample['video_name']}.json"
                per_seq_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        subset_summary = _aggregate_results(subset_results)
        subset_summary["sequences"] = subset_results
        all_summary["subsets"][subset] = subset_summary
        subset_summary_path = subset_out_dir / "summary.json"
        subset_summary_path.write_text(json.dumps(subset_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(_format_subset_summary(subset, subset_summary))

    overall_path = output_dir / "summary.json"
    overall_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["WorldTrack D4RT Summary"]
    for subset in subsets:
        if subset not in all_summary["subsets"]:
            continue
        lines.append(_format_subset_summary(subset, all_summary["subsets"][subset]))
    summary_txt = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")
    logger.info("Saved summary to %s", overall_path)
    print(summary_txt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
