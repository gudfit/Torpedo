import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_train_minimal(tmp_path, monkeypatch):
    from torpedocode.cli import train as train_cli

    # Create a minimal cached parquet in CWD expected by CLI (cache_root='.')
    # Use tmp_path as CWD by chdir
    monkeypatch.chdir(tmp_path)
    instrument = "TESTSYM"
    ts = pd.date_range("2025-01-01", periods=200, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * 50,
            "size": np.abs(np.random.randn(200)).astype(float),
            "price": 100 + np.cumsum(np.random.randn(200)).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(200).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(200).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=200),
            "ask_size_1": np.random.randint(1, 100, size=200),
        }
    )
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(Path(f"{instrument}.parquet"), index=False)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    art = Path("artifacts")
    argv = [
        "prog",
        "--instrument",
        instrument,
        "--label-key",
        "instability_s_1",
        "--artifact-dir",
        str(art),
        "--epochs",
        "1",
        "--batch",
        "8",
        "--bptt",
        "16",
        "--topo-stride",
        "4",
        "--hidden",
        "16",
        "--layers",
        "1",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    # Predictions should exist
    assert (art / "predictions_val.csv").exists()
    assert (art / "predictions_test.csv").exists()

