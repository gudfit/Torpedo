from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


def _try_build_and_load() -> bool:
    if os.environ.get("TORPEDOCODE_AUTO_BUILD_OPS", "0") not in {"1", "true", "TRUE"}:
        return False
    try:
        verbose = os.environ.get("TORPEDOCODE_VERBOSE", "0") in {"1", "true", "TRUE"}
        from torch.utils.cpp_extension import load as _load
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[3]
        src_cpp = root / "cpp" / "src" / "extension.cpp"
        if verbose:
            print("[torpedocode] JIT-building CPU-only Torch op (TORPEDOCODE_AUTO_BUILD_OPS=1)")
        _ = _load(
            name="torpedocode_kernels",
            sources=[str(src_cpp)],
            extra_cflags=["-O3"],
            include_dirs=[str(root / "cpp" / "include")],
            verbose=False,
        )
        if verbose:
            print("[torpedocode] Torch op loaded.")
        return True
    except Exception:
        return False

def _try_import_prebuilt() -> bool:
    # 1) Direct import by module name if already on sys.path
    try:
        import importlib

        importlib.import_module("torpedocode_kernels")
        _ = torch.ops.torpedocode.hybrid_forward
        return True
    except Exception:
        pass
    # 2) Search typical torch extension cache paths and load .so explicitly
    candidates: list[Path] = []
    env_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.home() / ".cache" / "torch_extensions")
    candidates.append(Path.cwd() / ".tmp" / "torch_extensions")

    try:
        import importlib.util
        import sys
        for root in candidates:
            if not root.exists():
                continue
            for so in root.rglob("torpedocode_kernels*.so"):
                # First try Torch's loader (registers ops without Python module naming constraints)
                try:
                    torch.ops.load_library(str(so))
                    _ = torch.ops.torpedocode.hybrid_forward
                    return True
                except Exception:
                    # Fallback: import as a Python extension module
                    try:
                        spec = importlib.util.spec_from_file_location("torpedocode_kernels", so)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                            sys.modules.setdefault("torpedocode_kernels", mod)
                            _ = torch.ops.torpedocode.hybrid_forward
                            return True
                    except Exception:
                        continue
    except Exception:
        pass
    return False


def has_torpedocode_op() -> bool:
    try:
        _ = torch.ops.torpedocode.hybrid_forward
        return True
    except (AttributeError, RuntimeError):
        # Try to import a prebuilt module first (CUDA/CPU), then JIT CPU fallback
        if _try_import_prebuilt():
            return True
        return _try_build_and_load()


def hybrid_fuse(
    features: torch.Tensor, topology: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fuse features and topology via native op if available; fallback to CPU add."""
    if has_torpedocode_op():
        try:
            out = torch.ops.torpedocode.hybrid_forward(
                features, topology, weights if weights is not None else torch.tensor([])
            )
            if isinstance(out, (tuple, list)):
                return out[0]
            return out
        except Exception:
            pass
    if features.shape[-1] == topology.shape[-1]:
        y = features + topology
    else:
        topo_red = topology.mean(dim=-1, keepdim=True).expand_as(features)
        y = features + topo_red
    if weights is not None and weights.numel() in (3, features.numel()):
        y = y * weights
    return y


__all__ = ["hybrid_fuse", "has_torpedocode_op"]
