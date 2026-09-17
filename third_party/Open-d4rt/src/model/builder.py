"""Model builder registry."""

from __future__ import annotations

from typing import Any

from src.core.registry import Registry

from .d4rt import D4RTModel

MODEL_REGISTRY = Registry("model")


@MODEL_REGISTRY.register("d4rt")
def _build_d4rt(model_cfg: Any):
    return D4RTModel(model_cfg)


def build_model(model_section_cfg: Any):
    model_name = model_section_cfg.get("name", "d4rt")
    builder = MODEL_REGISTRY.get(model_name)
    return builder(model_section_cfg)
