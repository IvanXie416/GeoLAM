"""Data package for datasets, samplers and dataloader builders."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

__all__ = ["build_dataloader"]


def build_dataloader(*args: Any, **kwargs: Any):
    from .builder import build_dataloader as _build_dataloader

    return _build_dataloader(*args, **kwargs)
