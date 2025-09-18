import pytest
import torch

from torpedocode.utils.ops import has_torpedocode_op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_fused_matches_cpu_simple():
    if not has_torpedocode_op():
        pytest.skip("native op not available")

    B, T, F = 2, 8, 4
    X = torch.randn(B, T, F, device="cuda", dtype=torch.float32)
    Z = torch.randn(B, T, F, device="cuda", dtype=torch.float32)
    W = torch.rand_like(X)

    # CPU reference
    ref = (X.cpu() + Z.cpu()) * W.cpu()

    # CUDA op
    y = torch.ops.torpedocode.hybrid_forward(X, Z, W)[0]
    assert torch.allclose(y.cpu(), ref, atol=1e-5, rtol=1e-5)

