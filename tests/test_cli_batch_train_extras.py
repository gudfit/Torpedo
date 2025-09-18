import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import json


def _write_minimal_cache(tmp_path: Path, name: str, n: int = 120):
    ts = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["LO+", "CX+", "LO-", "CX-", "MO+", "MO-"] * (n // 6),
            "level": [1, 2, 1, 2, np.nan, np.nan] * (n // 6),
            "size": np.abs(np.random.randn(n)).astype(float),
            "price": 100 + np.cumsum(np.random.randn(n)).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(n).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(n).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=n),
            "ask_size_1": np.random.randint(1, 100, size=n),
        }
    )
    import pyarrow  # noqa: F401
    df.to_parquet(tmp_path / f"{name}.parquet", index=False)


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_batch_train_temperature_and_tpp(tmp_path, monkeypatch):
    from torpedocode.cli import batch_train as bt

    inst = "SYM"
    try:
        _write_minimal_cache(tmp_path, inst)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    art_root = tmp_path / "artifacts"
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--artifact-root",
        str(art_root),
        "--label-key",
        "instability_s_1",
        "--mode",
        "per_instrument",
        "--epochs",
        "1",
        "--batch",
        "8",
        "--bptt",
        "16",
        "--device",
        "cpu",
        "--temperature-scale",
        "--tpp-diagnostics",
        "--beta",
        "0.0001",
        "--seed",
        "9",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bt.main()
    art = art_root / inst / "instability_s_1"
    assert (art / "predictions_test.csv").exists()
    assert (art / "training_meta.json").exists()
    # temp + tpp artifacts optional but should exist when flags set
    assert (art / "temperature.json").exists()
    assert (art / "tpp_test_arrays.npz").exists()
    assert (art / "tpp_test_diagnostics.json").exists()

