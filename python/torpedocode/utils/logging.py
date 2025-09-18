"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure a root logger with console and optional file sinks."""

    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


__all__ = ["configure_logging"]
