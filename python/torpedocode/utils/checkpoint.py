"""Checkpoint persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """Persist a model/training state to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """Load a checkpoint without instantiating tensors on GPU."""

    return torch.load(path, map_location="cpu")


__all__ = ["save_checkpoint", "load_checkpoint_metadata"]
