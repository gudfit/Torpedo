import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_pretrain_and_warm_start(tmp_path, monkeypatch):
    # Pretrain a tiny CTMC model and save checkpoint
    from torpedocode.cli import pretrain_ctmc as pre

    ckpt = tmp_path / "pretrained.pt"
    argv = [
        "prog",
        "--epochs",
        "1",
        "--steps",
        "5",
        "--batch",
        "4",
        "--T",
        "32",
        "--num-event-types",
        "4",
        "--feature-dim",
        "4",
        "--topo-dim",
        "2",
        "--hidden",
        "16",
        "--layers",
        "1",
        "--lr",
        "1e-3",
        "--device",
        "cpu",
        "--output",
        str(ckpt),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    pre.main()
    assert ckpt.exists()

    # Build a minimal cached parquet to run train CLI with warm-start
    from torpedocode.cli import train as train_cli

    monkeypatch.chdir(tmp_path)
    instrument = "TESTPRE"
    ts = pd.date_range("2025-01-01", periods=120, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * 30,
            "size": np.abs(np.random.randn(120)).astype(float),
            "price": 50 + np.cumsum(np.random.randn(120)).astype(float) * 0.01,
            "bid_price_1": 50 + np.random.randn(120).astype(float) * 0.01,
            "ask_price_1": 50.1 + np.random.randn(120).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=120),
            "ask_size_1": np.random.randint(1, 100, size=120),
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
        "--warm-start",
        str(ckpt),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    assert (art / "predictions_val.csv").exists()
    assert (art / "predictions_test.csv").exists()
