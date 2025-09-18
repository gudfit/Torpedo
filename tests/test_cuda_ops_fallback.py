import pytest

torch = pytest.importorskip("torch")

from torpedocode.utils.ops import hybrid_fuse, has_torpedocode_op


def test_hybrid_fuse_fallback_cpu():
    x = torch.ones(2, 3)
    z = 2 * torch.ones(2, 3)
    y = hybrid_fuse(x, z)
    assert torch.allclose(y, 3 * torch.ones(2, 3))

    w = torch.full((2, 3), 0.5)
    y2 = hybrid_fuse(x, z, w)
    assert torch.allclose(y2, 1.5 * torch.ones(2, 3))

    # Op availability should not be required on CPU
    assert isinstance(has_torpedocode_op(), bool)
