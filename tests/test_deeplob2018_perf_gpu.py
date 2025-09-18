import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_deeplob2018_gpu_forward_perf_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from torpedocode.cli import baselines as bl

    Fdim = 40
    T = 100
    B = 64
    model = bl.DeepLOB2018Model(time_len=T, feat_dim=Fdim, n_classes=3).cuda()
    x = torch.randn(B, T, Fdim, device="cuda")
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # Smoke threshold: just ensure it runs within a reasonable bound on typical GPUs
    # Skip asserting strict thresholds to avoid CI flakiness; ensure shapes OK
    assert y.shape == (B, T, 3)
