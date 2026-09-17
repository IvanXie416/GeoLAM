"""Configuration helpers with recursive node access and override support."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


@dataclass
class ConfigNode(dict):
    """Dictionary with attribute-style access."""

    def __init__(self, source: dict[str, Any] | None = None) -> None:
        super().__init__()
        source = source or {}
        for key, value in source.items():
            super().__setitem__(key, _wrap(value))

    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def clone(self) -> "ConfigNode":
        return ConfigNode(copy.deepcopy(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, ConfigNode) else v for v in value]
            else:
                out[key] = value
        return out

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, (ConfigNode, dict)) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: ConfigNode = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], ConfigNode):
                node[part] = ConfigNode()
            node = node[part]
        node[parts[-1]] = _wrap(value)


def load_yaml_config(path: str | Path) -> ConfigNode:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return ConfigNode(data)


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(config: ConfigNode, overrides: list[str] | None) -> ConfigNode:
    if not overrides:
        return config
    updated = config.clone()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        updated.set_path(key.strip(), _parse_override_value(raw_value.strip()))
    return updated

