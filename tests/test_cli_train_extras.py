import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import json


def _make_cache(tmp_path: Path, instrument: str = "TESTSYM") -> Path:
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
    import pyarrow  # noqa: F401
    path = tmp_path / f"{instrument}.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_train_meta_and_beta(tmp_path, monkeypatch):
    from torpedocode.cli import train as train_cli

    monkeypatch.chdir(tmp_path)
    instrument = "TESTSYM"
    try:
        _make_cache(tmp_path, instrument)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    art = Path("artifacts_beta")
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
        "--beta",
        "0.0002",
        "--seed",
        "11",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    meta = json.loads((art / "training_meta.json").read_text())
    assert pytest.approx(meta.get("beta"), rel=1e-6) == 2e-4
    assert meta.get("seed") == 11
    assert (art / "predictions_test.csv").exists()


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_train_temperature_and_tpp(tmp_path, monkeypatch):
    from torpedocode.cli import train as train_cli

    monkeypatch.chdir(tmp_path)
    instrument = "TESTSYM"
    try:
        _make_cache(tmp_path, instrument)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    art = Path("artifacts_temp")
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
        "--temperature-scale",
        "--tpp-diagnostics",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    # Outputs exist
    assert (art / "predictions_test.csv").exists()
    # Temperature and TPP artifacts persisted
    assert (art / "temperature.json").exists()
    assert (art / "tpp_test_arrays.npz").exists()
    assert (art / "tpp_test_diagnostics.json").exists()


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_train_market_embedding(tmp_path, monkeypatch):
    from torpedocode.cli import train as train_cli

    monkeypatch.chdir(tmp_path)
    instrument = "TESTSYM"
    try:
        _make_cache(tmp_path, instrument)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    art = Path("artifacts_me")
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
        "--include-market-embedding",
        "--market-vocab-size",
        "3",
        "--market-id",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_cli.main()
    assert (art / "predictions_test.csv").exists()

