"""Loss builder registry."""

from __future__ import annotations

from typing import Any

from src.core.registry import Registry

from .d4rt_loss import D4RTLoss

LOSS_REGISTRY = Registry("loss")


@LOSS_REGISTRY.register("d4rt")
def _build_d4rt_loss(train_cfg: Any):
    return D4RTLoss(train_cfg)


def build_loss(model_name: str, train_cfg: Any):
    builder = LOSS_REGISTRY.get(model_name)
    return builder(train_cfg)
