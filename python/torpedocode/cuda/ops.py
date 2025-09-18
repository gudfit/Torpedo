"""PyTorch extension loader for custom CUDA kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import types
from torch.utils.cpp_extension import load as load_cpp_extension


@dataclass
class TorpedoKernels:

    module: types.ModuleType

    def forward_pass(self, features, topology, weights=None):  # type: ignore[no-untyped-def]
        """Run the fused hybrid forward CUDA kernel.

        The Torch op is registered as torpedocode.hybrid_forward(features, topology, weights).
        """

        w = weights if weights is not None else features.new_zeros(0)
        return self.module.torpedocode.hybrid_forward(features, topology, w)


def load_extension(
    name: str, sources: Optional[list[str]] = None, extra_cuda_cflags: Optional[list[str]] = None
) -> TorpedoKernels:
    """Compile and load the CUDA/C++ extension at runtime."""

    if sources is None:
        base_dir = Path(__file__).resolve().parents[2] / "cpp" / "src"
        sources = [str(base_dir / "lob_kernels.cu"), str(base_dir / "extension.cpp")]

    module = load_cpp_extension(
        name=name,
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags or ["-O3"],
        verbose=True,
    )
    return TorpedoKernels(module=module)


__all__ = ["TorpedoKernels", "load_extension"]
