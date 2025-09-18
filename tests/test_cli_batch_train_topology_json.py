import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_batch_train_uses_topology_json(tmp_path, monkeypatch):
    from torpedocode.cli import batch_train as bt

    monkeypatch.chdir(tmp_path)
    # Two instruments
    insts = ["I1", "I2"]
    for inst in insts:
        ts = pd.date_range("2025-01-01", periods=120, freq="s", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "event_type": ["MO+", "MO-", "LO+", "LO-"] * 30,
                "size": np.abs(np.random.randn(120)).astype(float),
                "price": 100 + np.cumsum(np.random.randn(120)).astype(float) * 0.01,
                "bid_price_1": 100 + np.random.randn(120).astype(float) * 0.01,
                "ask_price_1": 100.1 + np.random.randn(120).astype(float) * 0.01,
                "bid_size_1": np.random.randint(1, 100, size=120),
                "ask_size_1": np.random.randint(1, 100, size=120),
            }
        )
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(Path(f"{inst}.parquet"), index=False)
        except Exception:
            pytest.skip("pyarrow not available")

    topo_json = tmp_path / "topo.json"
    sel = {
        "window_sizes_s": [1],
        "complex_type": "cubical",
        "max_homology_dimension": 1,
        "persistence_representation": "landscape",
        "landscape_levels": 3,
    }
    topo_json.write_text(json.dumps(sel))

    art = tmp_path / "artifacts"
    argv = [
        "prog",
        "--cache-root", str(tmp_path),
        "--artifact-root", str(art),
        "--instruments", *insts,
        "--label-keys", "instability_s_1",
        "--epochs", "1",
        "--device", "cpu",
        "--topology-json", str(topo_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bt.main()
    # Check each instrument's feature_schema.json has selected topology
    for inst in insts:
        schema = json.loads((art / inst / "instability_s_1" / "feature_schema.json").read_text())
        topo = schema.get("topology", {})
        assert topo.get("persistence_representation") == "landscape"
        assert topo.get("landscape_levels") == 3

