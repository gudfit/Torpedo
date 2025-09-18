import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


def _write_lobster_day(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    # Minimal messages: columns [time, lob_type, order_id, size, price, direction]
    # Use two rows, LOBSTER type 1 (add), direction 1 (bid)
    msg = pd.DataFrame(
        {
            0: [0.0, 1.0],  # time in seconds
            1: [1, 1],      # lob_type
            2: [1, 2],      # order_id
            3: [10, 12],    # size
            4: [10000, 10100],  # price in raw units
            5: [1, -1],     # direction (1=bid, -1=ask)
        }
    )
    ob = pd.DataFrame(
        {
            0: [100.0, 101.0],  # ask_price_1
            1: [5, 6],          # ask_size_1
            2: [99.9, 100.9],   # bid_price_1
            3: [7, 8],          # bid_size_1
        }
    )
    msg.to_csv(dirpath / "message_TEST.csv", header=False, index=False)
    ob.to_csv(dirpath / "orderbook_TEST.csv", header=False, index=False)


def test_cli_merge_lobster(tmp_path, monkeypatch):
    from torpedocode.cli import merge_lobster as cli

    d1 = tmp_path / "2024-06-01"
    d2 = tmp_path / "2024-06-02"
    _write_lobster_day(d1)
    _write_lobster_day(d2)

    cache_root = tmp_path / "cache"
    argv = [
        "prog",
        "--instrument", "AAPL",
        "--cache-root", str(cache_root),
        "--tick-size", "0.01",
        "--days", str(d1), str(d2),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pytest.skip("pyarrow not installed")
    cli.main()
    out = cache_root / "AAPL.parquet"
    assert out.exists()

