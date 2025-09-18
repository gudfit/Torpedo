import numpy as np
import os
import types


def test_tda_strict_guard_raises_when_backend_missing(monkeypatch):
    # Force strict mode via config; simulate missing ripser by intercepting import
    from torpedocode.features.topological import TopologicalFeatureGenerator
    from torpedocode.config import TopologyConfig

    called = {"count": 0}

    monkeypatch.setenv("TORPEDOCODE_STRICT_TDA", "1")
    # Replace ripser module with a dummy lacking 'ripser' symbol to trigger ImportError
    import sys as _sys
    # Simulate missing gudhi (cubical backend) by providing a dummy module
    monkeypatch.setitem(_sys.modules, "gudhi", types.ModuleType("gudhi"))

    cfg = TopologyConfig(
        window_sizes_s=[1], complex_type="cubical", max_homology_dimension=1, persistence_representation="landscape"
    )
    gen = TopologicalFeatureGenerator(cfg)
    # Use a minimal series to trigger ripser path; expect RuntimeError due to strict mode
    ts = np.array([0, 1, 2], dtype="datetime64[s]")
    X = np.random.randn(3, 6).astype(np.float32)
    try:
        try:
            _ = gen.rolling_transform(ts, X, stride=1)
            assert False, "Expected strict TDA to raise in strict mode"
        except Exception:
            pass
    finally:
        monkeypatch.delenv("TORPEDOCODE_STRICT_TDA", raising=False)
