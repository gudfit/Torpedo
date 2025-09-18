import numpy as np


def test_window_batches_includes_tpp_fields():
    from torpedocode.cli.train import _window_batches
    X = np.random.randn(20, 4).astype(np.float32)
    Z = np.random.randn(20, 3).astype(np.float32)
    y = (np.random.rand(20) > 0.5).astype(np.int64)
    et = np.random.randint(0, 2, size=20).astype(np.int64)
    dt = np.random.exponential(0.1, size=20).astype(np.float32)
    sz = np.exp(np.random.randn(20)).astype(np.float32)
    # bptt 5, batch size 2
    gen = _window_batches(X, Z, y, bptt=5, batch_size=2, balanced=False, event_type_ids=et, delta_t=dt, sizes=sz)
    batch = next(iter(gen))
    assert "event_type_ids" in batch and "delta_t" in batch and "sizes" in batch

