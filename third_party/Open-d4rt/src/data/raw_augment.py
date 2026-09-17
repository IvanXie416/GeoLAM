"""Shared raw-dataloader augmentations and hard-boundary helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES: tuple[int | None, ...] = (8, 16, None)
STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS: tuple[float, ...] = (0.65, 0.25, 0.10)


@dataclass
class RawAugmentConfig:
    color_jitter_enabled: bool = False
    jitter_brightness_enabled: bool = True
    jitter_saturation_enabled: bool = True
    jitter_contrast_enabled: bool = True
    jitter_hue_enabled: bool = True
    color_drop_prob: float = 0.0
    blur_prob: float = 0.0
    random_crop_enabled: bool = False
    crop_scale_min: float = 1.0
    crop_scale_max: float = 1.0
    crop_aspect_min: float = 0.5
    crop_aspect_max: float = 2.0
    crop_aspect_sampling: str = "log_uniform"
    crop_sampling_domain: str = "original"
    crop_boundary_mode: str = "rejection"
    random_zoom_prob: float = 0.0
    temporal_subsample_enabled: bool = False
    temporal_subsample_stride_sampling: str = "random"
    temporal_subsample_stride_min: int = 1
    temporal_subsample_stride_max: int = 16


def sanitize_depth_map(
    depth_m: np.ndarray,
    *,
    max_depth_m: float,
    depth_clip_percentile: float = 0.0,
    min_valid_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a filtered depth map and its validity mask.

    Invalid values are represented by the returned validity mask; the depth array
    is set to NaN at invalid pixels so accidental unmasked use fails loudly.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)

    max_depth = float(max_depth_m)
    if np.isfinite(max_depth) and max_depth > 0.0:
        valid &= depth < max_depth

    percentile = float(depth_clip_percentile)
    if 0.0 < percentile < 100.0 and bool(valid.any()):
        cutoff = float(np.percentile(depth[valid], percentile))
        if np.isfinite(cutoff) and cutoff > 0.0:
            valid &= depth <= cutoff

    min_ratio = max(0.0, min(1.0, float(min_valid_ratio)))
    if min_ratio > 0.0 and float(valid.mean()) < min_ratio:
        valid &= False

    filtered = np.where(valid, depth, np.nan).astype(np.float32, copy=False)
    return filtered, valid.astype(np.bool_, copy=False)


def count_valid_depth_frames(depth_valid_t_hw: np.ndarray, *, min_valid_ratio: float) -> int:
    valid = np.asarray(depth_valid_t_hw, dtype=bool)
    if valid.ndim != 3 or valid.shape[0] == 0:
        return 0
    ratio_threshold = max(0.0, min(1.0, float(min_valid_ratio)))
    frame_ratios = valid.reshape(valid.shape[0], -1).mean(axis=1)
    if ratio_threshold <= 0.0:
        return int((frame_ratios > 0.0).sum())
    return int((frame_ratios >= ratio_threshold).sum())


def build_augment_info(
    crop_info: dict[str, Any] | None,
    *,
    image_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Build a batch-stable augmentation info payload for debug/visualization.

    This must be present for every sample in a batch so that ``default_collate``
    can stack it safely. When no crop is applied, ``crop_hw`` falls back to the
    output image size and ``crop_applied`` is ``False``.
    """

    out_h = int(image_hw[0])
    out_w = int(image_hw[1])
    info = crop_info or {}
    crop_hw_raw = info.get("crop_hw", (out_h, out_w))
    crop_xy_raw = info.get("crop_xy", (0, 0))
    crop_h = int(crop_hw_raw[0]) if len(crop_hw_raw) >= 2 else out_h
    crop_w = int(crop_hw_raw[1]) if len(crop_hw_raw) >= 2 else out_w
    crop_x = int(crop_xy_raw[0]) if len(crop_xy_raw) >= 2 else 0
    crop_y = int(crop_xy_raw[1]) if len(crop_xy_raw) >= 2 else 0
    crop_applied = bool("crop_hw" in info)
    return {
        "crop_hw": np.asarray([crop_h, crop_w], dtype=np.int32),
        "crop_xy": np.asarray([crop_x, crop_y], dtype=np.int32),
        "crop_applied": np.asarray([crop_applied], dtype=np.bool_),
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _probability_if_enabled(cfg: Any, enabled_path: str, probability_path: str) -> float:
    prob = float(cfg.get_path(probability_path, 0.0))
    enabled_raw = cfg.get_path(enabled_path, None)
    if enabled_raw is None:
        return prob if prob > 0.0 else 0.0
    return prob if _truthy(enabled_raw) else 0.0


def _normalize_supported_mode(value: Any, *, path: str, supported: set[str]) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in supported:
        supported_msg = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported {path}={value!r}; supported values: {supported_msg}")
    return mode


def _min_max_pair(value: Any, default: tuple[float, float], *, floor: float) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    lo = float(value[0])
    hi = float(value[1])
    if lo > hi:
        lo, hi = hi, lo
    lo = max(float(floor), lo)
    hi = max(lo, hi)
    return lo, hi


def augment_cfg_from_train_config(cfg: Any) -> RawAugmentConfig:
    scale_min, scale_max = _min_max_pair(
        cfg.get_path("augmentation.random_crop.scale_min_max", [1.0, 1.0]),
        (1.0, 1.0),
        floor=0.05,
    )
    aspect_min, aspect_max = _min_max_pair(
        cfg.get_path("augmentation.random_crop.aspect_ratio_min_max", [0.5, 2.0]),
        (0.5, 2.0),
        floor=1e-3,
    )

    crop_aspect_sampling = _normalize_supported_mode(
        cfg.get_path("augmentation.random_crop.aspect_ratio_sampling", "log_uniform"),
        path="augmentation.random_crop.aspect_ratio_sampling",
        supported={"log_uniform"},
    )
    crop_sampling_domain = _normalize_supported_mode(
        cfg.get_path("augmentation.random_crop.sampling_domain", "original"),
        path="augmentation.random_crop.sampling_domain",
        supported={"original", "resized"},
    )
    crop_boundary_mode = _normalize_supported_mode(
        cfg.get_path("augmentation.random_crop.boundary_mode", "rejection"),
        path="augmentation.random_crop.boundary_mode",
        supported={"rejection"},
    )
    stride_sampling = _normalize_supported_mode(
        cfg.get_path("augmentation.temporal_subsample.stride_sampling", "random"),
        path="augmentation.temporal_subsample.stride_sampling",
        supported={"random"},
    )

    return RawAugmentConfig(
        color_jitter_enabled=_truthy(cfg.get_path("augmentation.color_jitter.enabled", False)),
        jitter_brightness_enabled=_truthy(cfg.get_path("augmentation.color_jitter.brightness", True)),
        jitter_saturation_enabled=_truthy(cfg.get_path("augmentation.color_jitter.saturation", True)),
        jitter_contrast_enabled=_truthy(cfg.get_path("augmentation.color_jitter.contrast", True)),
        jitter_hue_enabled=_truthy(cfg.get_path("augmentation.color_jitter.hue", True)),
        color_drop_prob=_probability_if_enabled(
            cfg,
            "augmentation.color_drop.enabled",
            "augmentation.color_drop.probability",
        ),
        blur_prob=_probability_if_enabled(
            cfg,
            "augmentation.gaussian_blur.enabled",
            "augmentation.gaussian_blur.probability",
        ),
        random_crop_enabled=_truthy(cfg.get_path("augmentation.random_crop.enabled", False)),
        crop_scale_min=max(0.05, scale_min),
        crop_scale_max=max(0.05, scale_max),
        crop_aspect_min=aspect_min,
        crop_aspect_max=aspect_max,
        crop_aspect_sampling=crop_aspect_sampling,
        crop_sampling_domain=crop_sampling_domain,
        crop_boundary_mode=crop_boundary_mode,
        random_zoom_prob=float(cfg.get_path("augmentation.random_crop.random_zoom_in_probability", 0.0)),
        temporal_subsample_enabled=_truthy(cfg.get_path("augmentation.temporal_subsample.enabled", False)),
        temporal_subsample_stride_sampling=stride_sampling,
        temporal_subsample_stride_min=int(cfg.get_path("augmentation.temporal_subsample.stride_min", 1)),
        temporal_subsample_stride_max=int(cfg.get_path("augmentation.temporal_subsample.stride_max", 16)),
    )


def sample_frame_indices_with_stride(
    rng: np.random.Generator,
    scene_len: int,
    clip_frames: int,
    cfg: RawAugmentConfig,
    training: bool,
    index: int = 0,
) -> list[int]:
    """Select frame indices with optional temporal stride subsampling.

    When ``cfg.temporal_subsample_enabled`` is True and ``training`` is True,
    stride is uniformly sampled from ``[stride_min, min(stride_max, max_stride)]``
    where ``max_stride = (scene_len - 1) // (clip_frames - 1)`` ensures the
    selected frames fit within the scene.

    Config keys (in training YAML under ``augmentation.temporal_subsample``):
        enabled: bool          — whether to apply temporal subsampling.
        stride_sampling: str   — currently only "random" is supported.
        stride_min: int        — minimum stride (default 1).
        stride_max: int        — maximum stride (default 16).
    """
    def _pad_to_clip(indices: list[int], *, target_len: int, valid_len: int) -> list[int]:
        if target_len <= 0:
            return []
        if valid_len <= 0:
            return [0] * target_len
        if not indices:
            indices = [0]
        max_valid = max(0, valid_len - 1)
        out = [int(np.clip(v, 0, max_valid)) for v in indices]
        if len(out) >= target_len:
            return out[:target_len]
        tail = out[-1]
        out.extend([tail] * (target_len - len(out)))
        return out

    t = int(clip_frames)
    if t <= 1:
        return [0]
    if scene_len <= 0:
        return [0] * t
    if training and bool(cfg.temporal_subsample_enabled):
        if cfg.temporal_subsample_stride_sampling != "random":
            raise ValueError(
                f"Unsupported temporal_subsample_stride_sampling={cfg.temporal_subsample_stride_sampling!r}"
            )
        max_stride = max(1, (scene_len - 1) // (t - 1))
        s_lo = max(1, int(cfg.temporal_subsample_stride_min))
        s_hi = min(max_stride, max(s_lo, int(cfg.temporal_subsample_stride_max)))
        stride = int(rng.integers(s_lo, s_hi + 1))
        max_start = max(0, scene_len - 1 - (t - 1) * stride)
        start = int(rng.integers(0, max_start + 1))
        return _pad_to_clip([start + i * stride for i in range(t)], target_len=t, valid_len=scene_len)
    max_start = scene_len - t
    if max_start <= 0:
        return _pad_to_clip(list(range(scene_len)), target_len=t, valid_len=scene_len)
    if training:
        start = int(rng.integers(0, max_start + 1))
    else:
        start = int((index * t) % (max_start + 1))
    return _pad_to_clip(list(range(start, start + t)), target_len=t, valid_len=scene_len)


def depth_boundary_mask(depth_m: np.ndarray, valid_mask: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Approximate hard-boundary mask from depth gradients."""

    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if d.ndim != 2 or valid.ndim != 2:
        return np.zeros_like(d, dtype=bool)
    if valid.sum() < 8:
        return np.zeros_like(valid, dtype=bool)

    # Smooth depth before the gradient pass so textured surfaces and local
    # sensor noise do not light up the boundary mask as aggressively.
    filled = np.where(valid, d, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    blur_ksize = 5
    blur_sigma = 1.2
    d_blur = cv2.GaussianBlur(filled, (blur_ksize, blur_ksize), blur_sigma)
    w_blur = cv2.GaussianBlur(weight, (blur_ksize, blur_ksize), blur_sigma)
    smooth = np.zeros_like(d_blur, dtype=np.float32)
    np.divide(d_blur, w_blur, out=smooth, where=(w_blur > 1e-6))

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
    grad[~valid] = 0.0
    vals = grad[valid]
    if vals.size < 8:
        return np.zeros_like(valid, dtype=bool)
    thr = float(np.quantile(vals, float(np.clip(q, 0.0, 0.99))))
    hard = (grad >= thr) & valid

    # Slight dilation to include neighborhood around boundaries.
    hard_u8 = hard.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    hard = cv2.dilate(hard_u8, kernel, iterations=1).astype(bool) & valid
    return hard


def sample_hard_query_flags(rng: np.random.Generator, queries_per_clip: int, hard_query_ratio: float) -> np.ndarray:
    """Sample per-query hard/easy flags with exact clip-level hard ratio."""

    m = max(0, int(queries_per_clip))
    flags = np.zeros((m,), dtype=np.bool_)
    if m == 0:
        return flags
    ratio = float(np.clip(hard_query_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return flags
    if ratio >= 1.0:
        flags[:] = True
        return flags
    n_hard = int(round(float(m) * ratio))
    n_hard = int(np.clip(n_hard, 0, m))
    if n_hard == 0:
        return flags
    picked = rng.choice(m, size=(n_hard,), replace=False)
    flags[picked] = True
    return flags


def _sample_t_tgt_from_source(
    rng: np.random.Generator,
    q_t_src: np.ndarray,
    clip_frames: int,
    delta_choices: Sequence[int | None],
    delta_probs: Sequence[float],
) -> np.ndarray:
    """Sample target timesteps from a local/global mixture around source time."""

    t = max(1, int(clip_frames))
    src = np.asarray(q_t_src, dtype=np.int64).reshape(-1)
    m = int(src.shape[0])
    if m <= 0:
        return np.zeros((0,), dtype=np.int64)

    choices = tuple(delta_choices)
    probs = np.asarray(delta_probs, dtype=np.float64)
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
    bucket_ids = rng.choice(len(choices), size=(m,), replace=True, p=probs)
    q_t_tgt = np.empty((m,), dtype=np.int64)
    for i in range(m):
        fs = int(np.clip(src[i], 0, t - 1))
        max_delta_raw = choices[int(bucket_ids[i])]
        if max_delta_raw is None:
            q_t_tgt[i] = int(rng.integers(0, t))
            continue

        max_delta = int(max_delta_raw)
        if max_delta < 0 or max_delta >= t - 1:
            q_t_tgt[i] = int(rng.integers(0, t))
            continue

        lo = max(0, fs - max_delta)
        hi = min(t - 1, fs + max_delta)
        q_t_tgt[i] = int(rng.integers(lo, hi + 1))
    return q_t_tgt


def sample_t_tgt_t_cam(
    rng: np.random.Generator,
    queries_per_clip: int,
    clip_frames: int,
    prob_t_tgt_equals_t_cam: float,
    q_t_src: np.ndarray | None = None,
    t_src_tgt_delta_choices: Sequence[int | None] | None = None,
    t_src_tgt_delta_probs: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (t_tgt, t_cam) with exact clip-level P(t_tgt=t_cam) when feasible.

    When ``q_t_src`` and a delta-mixture are provided, ``t_tgt`` is sampled
    relative to ``t_src``. This keeps static depth-reprojection queries local
    enough to retain valid 2D/displacement supervision while still preserving a
    small full-range bucket for long-baseline consistency.
    """

    m = max(0, int(queries_per_clip))
    t = max(1, int(clip_frames))
    if q_t_src is not None and t_src_tgt_delta_choices is not None and t_src_tgt_delta_probs is not None:
        q_t_tgt = _sample_t_tgt_from_source(
            rng=rng,
            q_t_src=np.asarray(q_t_src, dtype=np.int64).reshape(-1)[:m],
            clip_frames=t,
            delta_choices=t_src_tgt_delta_choices,
            delta_probs=t_src_tgt_delta_probs,
        )
        if q_t_tgt.shape[0] != m:
            raise ValueError(f"q_t_src must provide at least {m} timesteps, got {q_t_tgt.shape[0]}")
    else:
        q_t_tgt = rng.integers(0, t, size=(m,), dtype=np.int64)
    q_t_cam = rng.integers(0, t, size=(m,), dtype=np.int64)
    eq_flags = sample_hard_query_flags(rng, m, prob_t_tgt_equals_t_cam)

    if t <= 1:
        q_t_cam = q_t_tgt.copy()
        eq_flags[:] = True
        return q_t_tgt, q_t_cam, eq_flags

    if m <= 0:
        return q_t_tgt, q_t_cam, eq_flags

    q_t_cam[eq_flags] = q_t_tgt[eq_flags]
    neq_idx = np.flatnonzero(~eq_flags)
    if neq_idx.size > 0:
        # Uniformly sample one of (t-1) different timesteps.
        delta = rng.integers(1, t, size=(neq_idx.size,), dtype=np.int64)
        q_t_cam[neq_idx] = (q_t_tgt[neq_idx] + delta) % t
    return q_t_tgt, q_t_cam, eq_flags


def apply_photometric_augment(video_t_chw: np.ndarray, rng: np.random.Generator, cfg: RawAugmentConfig) -> np.ndarray:
    """Apply temporally consistent color jitter/drop/blur to video in [0,1]."""

    video = np.asarray(video_t_chw, dtype=np.float32)
    out = np.clip(video.copy(), 0.0, 1.0)
    if out.ndim != 4 or out.shape[1] != 3:
        return out

    if cfg.color_jitter_enabled:
        if cfg.jitter_brightness_enabled:
            b = float(rng.uniform(0.8, 1.2))
            out = out * b
        if cfg.jitter_contrast_enabled:
            c = float(rng.uniform(0.8, 1.2))
            mean = out.mean(axis=(2, 3), keepdims=True)
            out = (out - mean) * c + mean
        if cfg.jitter_saturation_enabled:
            s = float(rng.uniform(0.8, 1.2))
            gray = out.mean(axis=1, keepdims=True)
            out = gray + s * (out - gray)
        out = np.clip(out, 0.0, 1.0)

        if cfg.jitter_hue_enabled:
            hue_delta = float(rng.uniform(-0.05, 0.05))
            if abs(hue_delta) > 1e-6:
                t, _, h, w = out.shape
                for i in range(t):
                    img = np.transpose((out[i] * 255.0).astype(np.uint8), (1, 2, 0))
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    shift = int(round(hue_delta * 180.0))
                    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
                    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    out[i] = np.transpose(img2.astype(np.float32) / 255.0, (2, 0, 1))

    if cfg.color_drop_prob > 0.0 and float(rng.random()) < float(cfg.color_drop_prob):
        gray = out.mean(axis=1, keepdims=True)
        out = np.repeat(gray, repeats=3, axis=1)

    if cfg.blur_prob > 0.0 and float(rng.random()) < float(cfg.blur_prob):
        sigma = float(rng.uniform(0.1, 1.6))
        t = out.shape[0]
        for i in range(t):
            img = np.transpose(out[i], (1, 2, 0))
            img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=sigma, sigmaY=sigma)
            out[i] = np.transpose(img, (2, 0, 1))

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _sample_crop_window(
    h: int,
    w: int,
    rng: np.random.Generator,
    cfg: RawAugmentConfig,
    native_aspect_ratio: float,
) -> tuple[int, int, int, int]:
    if h <= 2 or w <= 2:
        return 0, 0, h, w

    scale_min = float(np.clip(cfg.crop_scale_min, 0.05, 1.0))
    scale_max = float(np.clip(cfg.crop_scale_max, scale_min, 1.0))
    asp_min = max(1e-3, float(cfg.crop_aspect_min))
    asp_max = max(asp_min, float(cfg.crop_aspect_max))
    if cfg.crop_aspect_sampling != "log_uniform":
        raise ValueError(f"Unsupported crop_aspect_sampling={cfg.crop_aspect_sampling!r}")
    if cfg.crop_boundary_mode != "rejection":
        raise ValueError(f"Unsupported crop_boundary_mode={cfg.crop_boundary_mode!r}")
    if cfg.crop_sampling_domain not in {"original", "resized"}:
        raise ValueError(f"Unsupported crop_sampling_domain={cfg.crop_sampling_domain!r}")

    native_ar = float(native_aspect_ratio)
    if not np.isfinite(native_ar) or native_ar <= 0.0:
        native_ar = float(w) / max(float(h), 1.0)

    def _sample_scale() -> float:
        scale = float(rng.uniform(scale_min, scale_max))
        if cfg.random_zoom_prob > 0.0 and float(rng.random()) < float(cfg.random_zoom_prob):
            scale = min(scale, float(rng.uniform(0.3, 0.7)))
        return float(np.clip(scale, 0.05, 1.0))

    def _sample_log_uniform(lo: float, hi: float) -> float:
        return math.exp(float(rng.uniform(math.log(lo), math.log(hi))))

    def _maybe_accept(crop_h: int, crop_w: int) -> tuple[int, int, int, int] | None:
        if crop_w < 2 or crop_h < 2 or crop_w > w or crop_h > h:
            return None
        if crop_w >= w and crop_h >= h:
            return 0, 0, h, w
        x0 = int(rng.integers(0, w - crop_w + 1))
        y0 = int(rng.integers(0, h - crop_h + 1))
        return x0, y0, crop_h, crop_w

    max_attempts = 32
    for _ in range(max_attempts):
        scale = _sample_scale()
        if cfg.crop_sampling_domain == "resized":
            area = float(h * w)
            aspect = _sample_log_uniform(asp_min, asp_max)
            crop_w = int(round(math.sqrt(area * scale * aspect)))
            crop_h = int(round(math.sqrt(area * scale / max(aspect, 1e-6))))
        else:
            feasible_min = max(asp_min, scale * native_ar)
            feasible_max = min(asp_max, native_ar / max(scale, 1e-6))
            if feasible_min > feasible_max:
                continue
            aspect = _sample_log_uniform(feasible_min, feasible_max)
            rel_w = math.sqrt(scale * aspect / native_ar)
            rel_h = math.sqrt(scale * native_ar / max(aspect, 1e-6))
            crop_w = int(round(rel_w * float(w)))
            crop_h = int(round(rel_h * float(h)))
        accepted = _maybe_accept(crop_h, crop_w)
        if accepted is not None:
            return accepted

    if cfg.crop_sampling_domain == "resized":
        fallback_scale = min(scale_max, 1.0)
        fallback_aspect = float(np.clip(float(w) / max(float(h), 1.0), asp_min, asp_max))
        area = float(h * w)
        crop_w = int(round(math.sqrt(area * fallback_scale * fallback_aspect)))
        crop_h = int(round(math.sqrt(area * fallback_scale / max(fallback_aspect, 1e-6))))
    else:
        max_feasible_scale = min(1.0, asp_max / native_ar, native_ar / asp_min)
        fallback_scale = float(np.clip(min(scale_max, max_feasible_scale), 4.0 / max(float(h * w), 1.0), 1.0))
        feasible_min = max(asp_min, fallback_scale * native_ar)
        feasible_max = min(asp_max, native_ar / max(fallback_scale, 1e-6))
        if feasible_min <= feasible_max:
            fallback_aspect = float(np.clip(native_ar, feasible_min, feasible_max))
            rel_w = math.sqrt(fallback_scale * fallback_aspect / native_ar)
            rel_h = math.sqrt(fallback_scale * native_ar / max(fallback_aspect, 1e-6))
            crop_w = int(round(rel_w * float(w)))
            crop_h = int(round(rel_h * float(h)))
        else:
            return 0, 0, h, w

    accepted = _maybe_accept(crop_h, crop_w)
    if accepted is not None:
        return accepted

    return 0, 0, h, w


def _transform_uv(
    uv_norm: np.ndarray,
    x0: int,
    y0: int,
    crop_h: int,
    crop_w: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv_norm, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        return uv, np.zeros((0,), dtype=bool)
    u_px = uv[:, 0] * max(float(w - 1), 1.0)
    v_px = uv[:, 1] * max(float(h - 1), 1.0)
    u_new = (u_px - float(x0)) / max(float(crop_w - 1), 1.0)
    v_new = (v_px - float(y0)) / max(float(crop_h - 1), 1.0)
    inside = (u_new >= 0.0) & (u_new <= 1.0) & (v_new >= 0.0) & (v_new <= 1.0)
    out = np.stack([np.clip(u_new, 0.0, 1.0), np.clip(v_new, 0.0, 1.0)], axis=-1).astype(np.float32)
    return out, inside


def _effective_crop_aspect_ratio(
    native_aspect_ratio: np.ndarray,
    *,
    crop_h: int,
    crop_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    native = np.asarray(native_aspect_ratio, dtype=np.float32).reshape(-1)
    scale_x = float(crop_w) / max(float(out_w), 1.0)
    scale_y = float(crop_h) / max(float(out_h), 1.0)
    effective = native * (scale_x / max(scale_y, 1e-6))
    return effective.astype(np.float32, copy=False)


def apply_spatial_crop_images_only(
    video_t_chw: np.ndarray,
    depth_t_hw: np.ndarray,
    depth_valid_t_hw: np.ndarray,
    k_t_33: np.ndarray,
    camera_valid_t: np.ndarray,
    rng: np.random.Generator,
    cfg: RawAugmentConfig,
    native_aspect_ratio: np.ndarray | None = None,
    out_info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random crop + resize-back on images/depth/K only (no query handling).

    Use this BEFORE query sampling so that queries are built in the cropped
    coordinate space and no queries are wasted.

    Returns (video, depth, depth_valid, k_seq, aspect_ratio).
    """

    video = np.asarray(video_t_chw, dtype=np.float32)
    depth = np.asarray(depth_t_hw, dtype=np.float32)
    depth_valid = np.asarray(depth_valid_t_hw, dtype=bool)
    k_seq = np.asarray(k_t_33, dtype=np.float32).copy()
    cam_valid = np.asarray(camera_valid_t, dtype=bool)

    t, _, h, w = video.shape
    if native_aspect_ratio is not None:
        _passthrough_ar = np.asarray(native_aspect_ratio, dtype=np.float32).reshape(-1)
    else:
        _passthrough_ar = np.array([float(w) / max(float(h), 1.0)], dtype=np.float32)
    if not cfg.random_crop_enabled:
        return video, depth, depth_valid, k_seq, _passthrough_ar

    native_ar_scalar = float(_passthrough_ar[0]) if _passthrough_ar.size > 0 else float(w) / max(float(h), 1.0)
    x0, y0, crop_h, crop_w = _sample_crop_window(
        h=h,
        w=w,
        rng=rng,
        cfg=cfg,
        native_aspect_ratio=native_ar_scalar,
    )
    if crop_h == h and crop_w == w:
        return video, depth, depth_valid, k_seq, _passthrough_ar

    out_video = np.empty_like(video)
    out_depth = np.empty_like(depth)
    out_valid = np.zeros_like(depth_valid)
    for i in range(t):
        img = np.transpose(video[i], (1, 2, 0))[y0 : y0 + crop_h, x0 : x0 + crop_w]
        dep = depth[i, y0 : y0 + crop_h, x0 : x0 + crop_w]
        val = depth_valid[i, y0 : y0 + crop_h, x0 : x0 + crop_w].astype(np.uint8)

        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        dep_r = cv2.resize(dep, (w, h), interpolation=cv2.INTER_NEAREST)
        val_r = cv2.resize(val, (w, h), interpolation=cv2.INTER_NEAREST) > 0

        out_video[i] = np.transpose(img_r, (2, 0, 1))
        out_depth[i] = dep_r.astype(np.float32)
        out_valid[i] = val_r

    sx = float(w) / float(crop_w)
    sy = float(h) / float(crop_h)
    for i in range(k_seq.shape[0]):
        if i >= cam_valid.shape[0] or not cam_valid[i]:
            continue
        k = k_seq[i]
        if not np.isfinite(k).all():
            continue
        k[0, 2] -= float(x0)
        k[1, 2] -= float(y0)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        k_seq[i] = k

    if out_info is not None:
        out_info["crop_hw"] = (int(crop_h), int(crop_w))
        out_info["crop_xy"] = (int(x0), int(y0))
        out_info["image_hw"] = (int(h), int(w))

    return (
        out_video.astype(np.float32, copy=False),
        out_depth.astype(np.float32, copy=False),
        out_valid.astype(bool, copy=False),
        k_seq.astype(np.float32, copy=False),
        _effective_crop_aspect_ratio(
            _passthrough_ar,
            crop_h=crop_h,
            crop_w=crop_w,
            out_h=h,
            out_w=w,
        ),
    )


def apply_spatial_crop_augment(
    video_t_chw: np.ndarray,
    depth_t_hw: np.ndarray,
    depth_valid_t_hw: np.ndarray,
    k_t_33: np.ndarray,
    camera_valid_t: np.ndarray,
    query: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    mask: dict[str, np.ndarray],
    rng: np.random.Generator,
    cfg: RawAugmentConfig,
    native_aspect_ratio: np.ndarray | None = None,
    out_info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Random crop + resize-back, with K/query/target mask updates."""

    video = np.asarray(video_t_chw, dtype=np.float32)
    depth = np.asarray(depth_t_hw, dtype=np.float32)
    depth_valid = np.asarray(depth_valid_t_hw, dtype=bool)
    k_seq = np.asarray(k_t_33, dtype=np.float32).copy()
    cam_valid = np.asarray(camera_valid_t, dtype=bool)

    t, _, h, w = video.shape
    # Fallback: use native_aspect_ratio if provided, else tensor w/h.
    if native_aspect_ratio is not None:
        _passthrough_ar = np.asarray(native_aspect_ratio, dtype=np.float32).reshape(-1)
    else:
        _passthrough_ar = np.array([float(w) / max(float(h), 1.0)], dtype=np.float32)
    if not cfg.random_crop_enabled:
        return (
            video,
            depth,
            depth_valid,
            k_seq,
            query,
            target,
            mask,
            _passthrough_ar,
        )

    native_ar_scalar = float(_passthrough_ar[0]) if _passthrough_ar.size > 0 else float(w) / max(float(h), 1.0)
    x0, y0, crop_h, crop_w = _sample_crop_window(
        h=h,
        w=w,
        rng=rng,
        cfg=cfg,
        native_aspect_ratio=native_ar_scalar,
    )
    if crop_h == h and crop_w == w:
        return (
            video,
            depth,
            depth_valid,
            k_seq,
            query,
            target,
            mask,
            _passthrough_ar,
        )

    out_video = np.empty_like(video)
    out_depth = np.empty_like(depth)
    out_valid = np.zeros_like(depth_valid)
    for i in range(t):
        img = np.transpose(video[i], (1, 2, 0))[y0 : y0 + crop_h, x0 : x0 + crop_w]
        dep = depth[i, y0 : y0 + crop_h, x0 : x0 + crop_w]
        val = depth_valid[i, y0 : y0 + crop_h, x0 : x0 + crop_w].astype(np.uint8)

        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        dep_r = cv2.resize(dep, (w, h), interpolation=cv2.INTER_NEAREST)
        val_r = cv2.resize(val, (w, h), interpolation=cv2.INTER_NEAREST) > 0

        out_video[i] = np.transpose(img_r, (2, 0, 1))
        out_depth[i] = dep_r.astype(np.float32)
        out_valid[i] = val_r

    sx = float(w) / float(crop_w)
    sy = float(h) / float(crop_h)
    for i in range(k_seq.shape[0]):
        if i >= cam_valid.shape[0] or not cam_valid[i]:
            continue
        k = k_seq[i]
        if not np.isfinite(k).all():
            continue
        k[0, 2] -= float(x0)
        k[1, 2] -= float(y0)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        k_seq[i] = k

    q_uv = np.stack([query["u"].astype(np.float32), query["v"].astype(np.float32)], axis=-1)
    q_uv_new, src_inside = _transform_uv(q_uv, x0=x0, y0=y0, crop_h=crop_h, crop_w=crop_w, h=h, w=w)
    query["u"] = q_uv_new[:, 0]
    query["v"] = q_uv_new[:, 1]

    tgt_uv = target.get("uv_2d")
    tgt_inside = np.ones((q_uv.shape[0],), dtype=bool)
    if tgt_uv is not None:
        tgt_uv_new, tgt_inside = _transform_uv(
            tgt_uv.astype(np.float32),
            x0=x0,
            y0=y0,
            crop_h=crop_h,
            crop_w=crop_w,
            h=h,
            w=w,
        )
        target["uv_2d"] = tgt_uv_new

    for key, arr in mask.items():
        if arr.shape[0] != src_inside.shape[0]:
            continue
        mask[key] = arr.astype(bool) & src_inside
    if "uv_2d" in mask:
        mask["uv_2d"] = mask["uv_2d"] & tgt_inside
    if "visibility" in mask:
        mask["visibility"] = mask["visibility"] & tgt_inside

    if out_info is not None:
        out_info["crop_hw"] = (int(crop_h), int(crop_w))
        out_info["crop_xy"] = (int(x0), int(y0))
        out_info["image_hw"] = (int(h), int(w))

    return (
        out_video.astype(np.float32, copy=False),
        out_depth.astype(np.float32, copy=False),
        out_valid.astype(bool, copy=False),
        k_seq.astype(np.float32, copy=False),
        query,
        target,
        mask,
        _effective_crop_aspect_ratio(
            _passthrough_ar,
            crop_h=crop_h,
            crop_w=crop_w,
            out_h=h,
            out_w=w,
        ),
    )
