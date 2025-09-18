"""Device management utilities (torch-optional)."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - optional dependency

    class _CPUDevice:
        def __init__(self):
            self.type = "cpu"

    class _TorchShim:
        def cuda_is_available(self):
            return False

        def device(self, name: str):
            return _CPUDevice()

    torch = _TorchShim()  # type: ignore


def resolve_device(prefer_cuda: bool = True):
    """Return a device instance consistent with user preferences."""

    try:
        if prefer_cuda and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            return torch.device("cuda")  # type: ignore[attr-defined]
        return torch.device("cpu")  # type: ignore[attr-defined]
    except Exception:
        return _CPUDevice()


__all__ = ["resolve_device"]
