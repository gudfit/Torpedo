"""Simple registry implementation for factories."""

from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    """Register constructors for configurable components."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._factories:
            raise KeyError(f"Factory '{name}' already registered")
        self._factories[name] = factory

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown factory '{name}'. Registered factories: {sorted(self._factories)}"
            ) from exc
        return factory(*args, **kwargs)


__all__ = ["Registry"]
