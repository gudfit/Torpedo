import numpy as np
import pandas as pd
from pathlib import Path

from torpedocode.config import DataConfig
from torpedocode.data.loader import LOBDatasetBuilder


def _write_cache(tmp: Path, name: str):
    ts = pd.date_range("2025-01-01", periods=8, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            # Mix of LO/CX (with levels) and MO
            "event_type": ["LO+", "CX+", "LO-", "CX-", "MO+", "MO-", "LO+", "CX+"],
            "level": [1, 2, 1, 2, np.nan, np.nan, 2, 1],
            "size": np.ones(8),
            "price": np.linspace(10.0, 10.1, 8),
            "bid_price_1": 10.0,
            "ask_price_1": 10.01,
            "bid_size_1": 10,
            "ask_size_1": 10,
        }
    )
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False
    df.to_parquet(tmp / f"{name}.parquet", index=False)
    return True


def test_event_types_expand_by_level(tmp_path: Path):
    inst = "EXPAND"
    if not _write_cache(tmp_path, inst):
        import pytest

        pytest.skip("pyarrow not available")
    # Default: no expansion
    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=[inst], levels=1)
    b = LOBDatasetBuilder(cfg)
    rec = b.build_sequence(inst)
    uniq = len(np.unique(rec["event_type_ids"]))
    assert uniq <= 6

    # With expansion flag: LO/CX types split by level
    cfg2 = DataConfig(
        raw_data_root=tmp_path,
        cache_root=tmp_path,
        instruments=[inst],
        levels=1,
        expand_event_types_by_level=True,
    )
    b2 = LOBDatasetBuilder(cfg2)
    rec2 = b2.build_sequence(inst)
    uniq2 = len(np.unique(rec2["event_type_ids"]))
    assert uniq2 > uniq
