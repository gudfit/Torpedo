import pytest

torch = pytest.importorskip("torch")

from torpedocode.utils.ops import hybrid_fuse


def test_hybrid_fuse_mismatched_dims():
    # features last dim != topology last dim
    x = torch.ones(2, 3, 4)
    z = 2 * torch.ones(2, 3, 3)
    y = hybrid_fuse(x, z)
    # Fallback reduces topology -> mean across last dim (value=2), broadcast to features
    # So result is ones + twos = threes
    assert torch.allclose(y, 3 * torch.ones_like(x))

