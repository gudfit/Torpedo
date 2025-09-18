"""Utility helpers for TorpedoCode."""

from .logging import configure_logging
from .registry import Registry
from .device import resolve_device

__all__ = ["configure_logging", "Registry", "resolve_device"]

try:  # optional torch dependency for checkpointing
    import torch  # noqa: F401
    from .checkpoint import save_checkpoint, load_checkpoint_metadata

    __all__ += ["save_checkpoint", "load_checkpoint_metadata"]
except Exception:  # pragma: no cover - optional path
    pass
