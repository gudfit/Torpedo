import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_train_uses_topology_selected_json(tmp_path, monkeypatch):
    from torpedocode.cli import train as train_cli

    monkeypatch.chdir(tmp_path)
    instrument = "TSYM"
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
        df.to_parquet(Path(f"{instrument}.parquet"), index=False)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    # Write topology_selected.json with distinctive settings
    topo_dir = Path("artifacts/topo") / instrument
    topo_dir.mkdir(parents=True, exist_ok=True)
    topo_sel = {
        "window_sizes_s": [5],
        "complex_type": "vietoris_rips",
        "max_homology_dimension": 1,
        "persistence_representation": "image",
        "image_resolution": 32,
        "image_bandwidth": 0.05,
    }
    (topo_dir / "topology_selected.json").write_text(json.dumps(topo_sel))

    art = Path("artifacts"); art.mkdir(exist_ok=True)
    argv = [
        "prog",
        "--instrument", instrument,
        "--label-key", "instability_s_1",
        "--artifact-dir", str(art),
        "--epochs", "1",
        "--batch", "8",
        "--bptt", "16",
        "--device", "cpu",
        "--use-topo-selected",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    # Verify feature_schema.json topology matches selected
    schema = json.loads((art / "feature_schema.json").read_text())
    topo = schema.get("topology", {})
    assert topo.get("persistence_representation") == "image"
    assert topo.get("image_resolution") == 32

