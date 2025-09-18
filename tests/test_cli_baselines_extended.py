import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_minimal_cache(tmp_path: Path, name: str, n: int = 200):
    ts = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * (n // 4),
            "size": np.abs(np.random.randn(n)).astype(float),
            "price": 100 + np.cumsum(np.random.randn(n)).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(n).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(n).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=n),
            "ask_size_1": np.random.randint(1, 100, size=n),
        }
    )
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_baselines_deeplob(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import baselines as bl

    inst = "TESTDL"
    _write_minimal_cache(tmp_path, inst, n=160)
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--instrument",
        inst,
        "--label-key",
        "instability_s_1",
        "--baseline",
        "deeplob",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bl.main()
    out = capsys.readouterr().out
    assert "auroc" in out and "deeplob_small" in out


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_baselines_tpp(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import baselines as bl

    inst = "TESTTPP"
    _write_minimal_cache(tmp_path, inst, n=180)
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--instrument",
        inst,
        "--baseline",
        "tpp",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bl.main()
    out = capsys.readouterr().out
    assert "nll_per_event" in out and "neural_tpp" in out


def test_cli_baselines_hawkes(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import baselines as bl

    inst = "TESTHK"
    _write_minimal_cache(tmp_path, inst, n=140)
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--instrument",
        inst,
        "--baseline",
        "hawkes",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bl.main()
    out = capsys.readouterr().out
    assert "nll_per_event" in out and "hawkes_const_rate" in out
