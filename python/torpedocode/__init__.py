"""Top-level package for the TorpedoCode research framework.

This package provides tooling for:

* ingesting and preprocessing limit order book event streams;
* constructing conventional microstructure and topological features;
* defining hybrid neural-temporal point process models with CUDA backends;
* coordinating training, evaluation, and calibration workflows; and
* exporting research artifacts for reproducibility studies.

The modules are intentionally lightweight so researchers can extend them with
custom CUDA kernels or Rust/C++ data services without rewriting the Python
orchestration layer.
"""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_PACKAGE_COMPONENTS = {
    "data": ".data",
    "features": ".features",
    "models": ".models",
    "training": ".training",
    "evaluation": ".evaluation",
    "utils": ".utils",
    "cuda": ".cuda",
}


def load_component(name: str) -> Any:
    """Dynamically import a package component."""

    try:
        module_path = _PACKAGE_COMPONENTS[name]
    except KeyError as exc:  # pragma: no cover - defensive programming.
        raise KeyError(
            f"Unknown component '{name}'. Available keys: {sorted(_PACKAGE_COMPONENTS)}"
        ) from exc

    return import_module(module_path, package=__name__)


__all__ = ["load_component", "__version__"]
