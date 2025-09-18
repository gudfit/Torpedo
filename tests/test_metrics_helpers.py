import numpy as np

from torpedocode.evaluation.metrics import (
    ks_statistic,
    kolmogorov_pvalue,
    stationary_block_indices,
)


def test_ks_wrappers_basic_properties():
    # Deterministic vector in [0,1]
    u = np.linspace(0.0, 1.0, 101)
    ks = ks_statistic(u)
    p = kolmogorov_pvalue(ks, len(u))
    assert np.isfinite(ks)
    assert 0.0 <= p <= 1.0


def test_stationary_block_indices_shape_and_range():
    rng = np.random.default_rng(0)
    n = 123
    L = 25.0
    idx = stationary_block_indices(n, L, rng)
    assert isinstance(idx, np.ndarray)
    assert idx.shape == (n,)
    # Indices should be valid positions
    assert idx.min() >= 0 and idx.max() < n
    # With moderate L, we expect some repetition but not all identical
    assert len(np.unique(idx)) > n // 4
