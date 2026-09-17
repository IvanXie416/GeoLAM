"""Shared dataset seeding helpers for multi-worker / DDP-safe sampling."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterable
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional in non-OpenCV environments
    cv2 = None

try:
    import torch
    from torch.utils.data import ConcatDataset, get_worker_info
except Exception:  # pragma: no cover - torch is available in normal training
    torch = None
    ConcatDataset = None

    def get_worker_info():  # type: ignore[override]
        return None


def _stable_namespace_seed(namespace: str) -> int:
    digest = hashlib.blake2b(str(namespace).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def stable_split_bucket(name: str, modulo: int = 20) -> int:
    """Stable hash bucket for dataset splits across processes and runs."""

    digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return value % max(3, int(modulo))


def _dataset_children(dataset: Any) -> list[Any]:
    if ConcatDataset is not None and isinstance(dataset, ConcatDataset):
        return list(dataset.datasets)
    children = getattr(dataset, "datasets", None)
    if isinstance(children, Iterable) and not isinstance(children, (str, bytes, dict)):
        return list(children)
    return []


class SeededDatasetMixin:
    """Utility mixin for deterministic, worker-aware per-sample RNG."""

    def _init_dataset_seeding(self, namespace: str, default_seed: int = 0) -> None:
        self._seed_namespace = str(namespace)
        self._seed_default = int(default_seed)
        self._seed_base = int(default_seed)
        self._seed_epoch = 0
        self._seed_worker_id = 0
        self._seed_rank = int(os.environ.get("RANK", "0"))
        self.rng = np.random.default_rng(self._seed_material(index=0, attempt=0, stream=0))

    def configure_dataset_seed(self, base_seed: int) -> None:
        seed = (int(base_seed) + _stable_namespace_seed(getattr(self, "_seed_namespace", type(self).__name__))) % (2**63 - 1)
        self._seed_base = int(seed)
        self.rng = np.random.default_rng(self._seed_material(index=0, attempt=0, stream=0))

    def set_dataset_epoch(self, epoch: int) -> None:
        self._seed_epoch = int(epoch)

    def set_dataset_worker(self, worker_id: int) -> None:
        self._seed_worker_id = int(worker_id)

    def _seed_material(self, index: int, attempt: int, stream: int) -> np.random.SeedSequence:
        worker_id = getattr(self, "_seed_worker_id", 0)
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = int(worker_info.id)
        rank = int(getattr(self, "_seed_rank", int(os.environ.get("RANK", "0"))))
        epoch = int(getattr(self, "_seed_epoch", 0))
        base = int(getattr(self, "_seed_base", getattr(self, "_seed_default", 0)))
        return np.random.SeedSequence([base, rank, worker_id, epoch, int(index), int(attempt), int(stream)])

    def _sample_rng(self, index: int, attempt: int = 0, stream: int = 0) -> np.random.Generator:
        return np.random.default_rng(self._seed_material(index=index, attempt=attempt, stream=stream))

    def _bind_sample_rng(self, index: int, attempt: int = 0, stream: int = 0) -> np.random.Generator:
        rng = self._sample_rng(index=index, attempt=attempt, stream=stream)
        self.rng = rng
        return rng

    def _prepare_sample_rng(self, index: int, total: int, attempt: int) -> tuple[int, np.random.Generator]:
        total_i = max(1, int(total))
        if attempt <= 0:
            query_index = int(index)
        else:
            retry_rng = self._sample_rng(index=int(index), attempt=int(attempt), stream=1)
            query_index = int(retry_rng.integers(0, total_i))
        rng = self._bind_sample_rng(index=query_index, attempt=int(attempt), stream=0)
        return query_index, rng


def configure_dataset_seeding(dataset: Any, base_seed: int) -> None:
    if hasattr(dataset, "configure_dataset_seed"):
        dataset.configure_dataset_seed(int(base_seed))
    for child in _dataset_children(dataset):
        configure_dataset_seeding(child, base_seed)


def set_dataset_epoch_recursive(dataset: Any, epoch: int) -> None:
    if hasattr(dataset, "set_dataset_epoch"):
        dataset.set_dataset_epoch(int(epoch))
    for child in _dataset_children(dataset):
        set_dataset_epoch_recursive(child, epoch)


def set_dataset_worker_recursive(dataset: Any, worker_id: int) -> None:
    if hasattr(dataset, "set_dataset_worker"):
        dataset.set_dataset_worker(int(worker_id))
    for child in _dataset_children(dataset):
        set_dataset_worker_recursive(child, worker_id)


def seed_dataloader_worker(worker_id: int) -> None:
    worker_info = get_worker_info()
    if worker_info is None:
        return
    seed = int(getattr(worker_info, "seed", 0)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    # DataLoader already parallelizes across worker processes. Letting OpenCV or
    # torch spawn their own CPU thread pools inside every worker can oversubscribe
    # the node and create long-tail stalls on image/depth-heavy raw datasets.
    if cv2 is not None:
        try:
            cv2.setNumThreads(int(os.environ.get("D4RT_CV2_WORKER_THREADS", "0")))
        except Exception:
            pass
    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.set_num_threads(max(1, int(os.environ.get("D4RT_TORCH_WORKER_THREADS", "1"))))
        except Exception:
            pass
    set_dataset_worker_recursive(worker_info.dataset, worker_id)
