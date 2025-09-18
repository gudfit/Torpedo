import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    except Exception:
        import pytest

        pytest.skip("pyarrow not available for parquet cache")


def test_cli_batch_types_info(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import batch_train as bt

    inst1 = "SYM1"
    inst2 = "SYM2"
    _write_minimal_cache(tmp_path, inst1)
    _write_minimal_cache(tmp_path, inst2)
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--label-key",
        "instability_s_1",
        "--mode",
        "per_instrument",
        "--expand-types-by-level",
        "--print-types-info",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bt.main()
    out = capsys.readouterr().out
    import json

    info = json.loads(out)
    assert inst1 in info and inst2 in info
    assert info[inst1] >= 1 and info[inst2] >= 1
