"""Minimal model-query helpers used by WorldTrack evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 3 or src.shape[0] < 3:
        return None

    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = (src_centered.T @ dst_centered) / float(n)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones((3,), dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1] = -1.0
    rot = vt.T @ np.diag(d) @ u.T
    var_src = float((src_centered**2).sum() / float(n))
    if var_src <= 1e-12:
        return None
    scale = float((s * d).sum() / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    if not (np.isfinite(scale) and np.isfinite(rot).all() and np.isfinite(trans).all()):
        return None
    return scale, rot, trans


def _solve_scale_only(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt_flat = np.asarray(gt, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred_flat) & np.isfinite(gt_flat)
    if not np.any(valid):
        return 1.0
    denom = float(np.dot(pred_flat[valid], pred_flat[valid]))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(pred_flat[valid], gt_flat[valid]) / denom)


def _umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 3 or src.shape[0] < 3:
        return None

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = (src_centered.T @ dst_centered) / float(src.shape[0])
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = mu_dst - rot @ mu_src
    if not (np.isfinite(rot).all() and np.isfinite(trans).all()):
        return None
    return rot, trans


def _estimate_intrinsics_params_from_predictions(
    pred_tracks: np.ndarray,
    pred_uv_norm: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    height, width = image_hw
    cx = 0.5 * float(max(width - 1, 1))
    cy = 0.5 * float(max(height - 1, 1))
    u_px = np.asarray(pred_uv_norm, dtype=np.float64)[..., 0] * float(max(width - 1, 1))
    v_px = np.asarray(pred_uv_norm, dtype=np.float64)[..., 1] * float(max(height - 1, 1))

    pred = np.asarray(pred_tracks, dtype=np.float64)
    x = pred[..., 0]
    y = pred[..., 1]
    z = pred[..., 2]

    fx_vals = z * np.abs(u_px - cx) / np.maximum(np.abs(x), 1e-6)
    fy_vals = z * np.abs(v_px - cy) / np.maximum(np.abs(y), 1e-6)
    fx_vals = fx_vals[np.isfinite(fx_vals) & (fx_vals > 1e-6)]
    fy_vals = fy_vals[np.isfinite(fy_vals) & (fy_vals > 1e-6)]

    fx = float(np.median(fx_vals)) if fx_vals.size > 0 else float(max(width, 1))
    fy = float(np.median(fy_vals)) if fy_vals.size > 0 else float(max(height, 1))
    return np.asarray([fx, fy, cx, cy], dtype=np.float64)


def _run_model_for_queries(
    model: torch.nn.Module,
    video_b: torch.Tensor,
    aspect_b: torch.Tensor | None,
    query: dict[str, torch.Tensor],
    chunk_size: int,
    memory_b: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    num_queries = int(query["u"].numel())
    if num_queries == 0:
        return {
            "xyz_3d": torch.empty((0, 3), dtype=video_b.dtype),
            "uv_2d": torch.empty((0, 2), dtype=video_b.dtype),
            "visibility": torch.empty((0,), dtype=video_b.dtype),
            "displacement": torch.empty((0, 3), dtype=video_b.dtype),
            "normal": torch.empty((0, 3), dtype=video_b.dtype),
            "confidence": torch.empty((0,), dtype=video_b.dtype),
        }

    out_chunks: dict[str, list[torch.Tensor]] = {}
    step = max(1, int(chunk_size))
    cached_model = getattr(model, "module", model)
    decode_queries = getattr(cached_model, "decode_queries", None)

    for start in range(0, num_queries, step):
        end = min(num_queries, start + step)
        query_chunk = {
            "u": query["u"][start:end].view(1, -1),
            "v": query["v"][start:end].view(1, -1),
            "t_src": query["t_src"][start:end].view(1, -1),
            "t_tgt": query["t_tgt"][start:end].view(1, -1),
            "t_cam": query["t_cam"][start:end].view(1, -1),
        }
        if memory_b is not None and callable(decode_queries):
            pred = decode_queries(video=video_b, query=query_chunk, memory=memory_b)
        else:
            batch: dict[str, Any] = {"video": video_b, "query": query_chunk}
            if aspect_b is not None:
                batch["aspect_ratio"] = aspect_b
            pred = model(batch)

        for key, value in pred.items():
            chunk_value = value[0].detach()
            if torch.is_floating_point(chunk_value):
                chunk_value = chunk_value.to(dtype=torch.float32)
            out_chunks.setdefault(key, []).append(chunk_value.cpu())

    return {key: torch.cat(chunks, dim=0) for key, chunks in out_chunks.items()}


def _encode_model_memory(
    model: torch.nn.Module,
    video_b: torch.Tensor,
    aspect_b: torch.Tensor | None,
) -> torch.Tensor | None:
    cached_model = getattr(model, "module", model)
    encode_video = getattr(cached_model, "encode_video", None)
    if not callable(encode_video):
        return None
    return encode_video(video=video_b, aspect_ratio=aspect_b)


def _model_clip_frames(model: torch.nn.Module | None, default: int = 48) -> int:
    if model is None:
        return int(default)
    cached_model = getattr(model, "module", model)
    query_embedder = getattr(cached_model, "query_embedder", None)
    max_frames = getattr(query_embedder, "max_frames", None)
    try:
        return max(1, int(max_frames))
    except (TypeError, ValueError):
        return int(default)
