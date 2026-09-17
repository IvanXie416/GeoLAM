"""Simple class/function registry."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """Named registry for extensible builders."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str) -> Callable[[Any], Any]:
        if not key:
            raise ValueError(f"{self.name}: key cannot be empty")

        def decorator(item: Any) -> Any:
            if key in self._items:
                raise KeyError(f"{self.name}: key '{key}' already registered")
            self._items[key] = item
            return item

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            known = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"{self.name}: unknown key '{key}'. Known: [{known}]")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items.keys())

