from __future__ import annotations

import numpy as np
import pytest


def test_deeplobfull_forward_shapes():
    torch = pytest.importorskip("torch")
    from torpedocode.models.baselines import DeepLOBFull

    B, T, F = 2, 32, 8
    x = torch.randn(B, T, F)
    model = DeepLOBFull(fdim=F)
    y = model(x)
    assert y.shape == (B, T, 1)
