import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _write_minimal_cache(tmp_path: Path, name: str, n: int = 160):
    ts = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "CX+"] * (n // 4),
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
        import pytest

        pytest.skip("pyarrow not available for parquet cache")


def test_cli_baselines_hawkes_exp(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import baselines as bl

    inst = "TESTHKEXP"
    _write_minimal_cache(tmp_path, inst, n=160)
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--instrument",
        inst,
        "--baseline",
        "hawkes_exp",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bl.main()
    out = capsys.readouterr().out
    assert "hawkes_exp" in out and "nll_per_event" in out
