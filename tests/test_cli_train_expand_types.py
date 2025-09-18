import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_train_expand_types(tmp_path, monkeypatch):
    from torpedocode.cli import train as train_cli

    monkeypatch.chdir(tmp_path)
    instrument = "TESTEXPAND"
    ts = pd.date_range("2025-01-01", periods=160, freq="s", tz="UTC")
    et = ["LO+", "CX+", "LO-", "CX-", "MO+", "MO-"] * (160 // 6)
    lvl = [1, 2, 1, 2, np.nan, np.nan] * (160 // 6)
    df = pd.DataFrame(
        {
            "timestamp": ts[: len(et)],
            "event_type": et,
            "level": lvl,
            "size": np.abs(np.random.randn(len(et))).astype(float),
            "price": 100 + np.cumsum(np.random.randn(len(et))).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(len(et)).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(len(et)).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=len(et)),
            "ask_size_1": np.random.randint(1, 100, size=len(et)),
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
        "--expand-types-by-level",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    assert (art / "predictions_val.csv").exists()
    assert (art / "predictions_test.csv").exists()
