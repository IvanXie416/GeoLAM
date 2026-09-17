"""Dataset composition utilities for weighted multi-source training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

# Canonical keys that every raw dataset must provide in sample["meta"].
# Dataset-specific diagnostic keys (e.g. camera_convention, depth_decode_mode
# from DynamicReplica) are stripped during mixture sampling so that PyTorch's
# default_collate can batch heterogeneous sources without KeyError.
_CANONICAL_META_KEYS = frozenset({
    "dataset",
    "scene_id",
    "clip_start",
    "source_mode",
    "sample_key",
})


def _normalize_sample_meta(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep only canonical keys in ``sample["meta"]``."""
    meta = sample.get("meta")
    if meta is not None and isinstance(meta, dict):
        sample["meta"] = {k: v for k, v in meta.items() if k in _CANONICAL_META_KEYS}
    return sample


@dataclass
class MixtureDatasetConfig:
    datasets: Sequence[Dataset]
    weights: Sequence[float] | None = None
    pattern_scale: int = 100


class MixtureDataset(Dataset):
    """Round-robin weighted mixture over multiple datasets."""

    def __init__(self, config: MixtureDatasetConfig) -> None:
        if not config.datasets:
            raise ValueError("MixtureDataset requires at least one dataset")
        self.datasets = list(config.datasets)
        raw_weights = np.array(config.weights if config.weights is not None else [1.0] * len(self.datasets), dtype=np.float64)
        if raw_weights.shape[0] != len(self.datasets):
            raise ValueError("weights length must match number of datasets")
        if np.any(raw_weights < 0):
            raise ValueError("weights must be non-negative")
        if float(raw_weights.sum()) <= 0:
            raw_weights = np.ones_like(raw_weights)
        weights = raw_weights / raw_weights.sum()

        pattern: list[int] = []
        for i, w in enumerate(weights):
            n = max(1, int(round(float(w) * config.pattern_scale)))
            pattern.extend([i] * n)
        self.pattern = np.array(pattern, dtype=np.int64)
        self.max_len = max(len(ds) for ds in self.datasets)

    def __len__(self) -> int:
        return int(self.max_len * len(self.pattern))

    def __getitem__(self, index: int):
        did = int(self.pattern[index % len(self.pattern)])
        ds = self.datasets[did]
        sub_index = index % len(ds)
        sample = ds[sub_index]
        return _normalize_sample_meta(sample)

