import numpy as np
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_deeplob2018_forward_shapes():
    import torch
    from torpedocode.cli import baselines as bl

    Fdim = 40
    T = 64
    model = bl.__dict__["DeepLOBFull"](Fdim)  # type: ignore[attr-defined]
    x = torch.randn(2, T, Fdim)
    y = model(x).squeeze(-1)
    assert y.shape == (2, T)
