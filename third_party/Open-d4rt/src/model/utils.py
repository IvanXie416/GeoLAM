"""Model helpers."""

from __future__ import annotations

import math
from typing import Any

import torch


def resolve_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sinusoidal_position_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("sinusoidal_position_embedding requires even dim")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(eps)
    return (values * mask_f).sum() / denom


def masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.ndim != mask.ndim:
        raise ValueError(f"values/mask ndim mismatch: {values.shape} vs {mask.shape}")
    if values.ndim == 1:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
    if values.ndim < 2:
        raise ValueError(f"Expected at least [B, N], got {values.shape}")
    mask_f = mask.to(dtype=values.dtype)
    reduce_dims = tuple(range(1, values.ndim))
    denom = mask_f.sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return (values * mask_f).sum(dim=reduce_dims, keepdim=True) / denom
