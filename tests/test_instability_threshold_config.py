import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _write_cache(tmp_path: Path, name: str, n: int = 120, step: float = 0.001):
    ts = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    price = 100 + np.arange(n, dtype=float) * step
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * (n // 4),
            "size": np.ones(n, dtype=float),
            "price": price,
            "bid_price_1": price - 0.01,
            "ask_price_1": price + 0.01,
            "bid_size_1": np.random.randint(1, 10, size=n),
            "ask_size_1": np.random.randint(1, 10, size=n),
        }
    )
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")


def test_instability_threshold_eta_labels(tmp_path):
    from torpedocode.config import DataConfig
    from torpedocode.data.loader import LOBDatasetBuilder

    inst = "THETA"
    _write_cache(tmp_path, inst, n=120, step=0.001)

    # With high eta, all zeros
    dc = DataConfig(
        raw_data_root=tmp_path,
        cache_root=tmp_path,
        instruments=[inst],
        instability_threshold_eta=0.01,
    )
    builder = LOBDatasetBuilder(dc)
    tr, va, te, _ = builder.build_splits(inst, label_key="instability_s_1")
    assert int(tr["labels"].sum() + va["labels"].sum() + te["labels"].sum()) == 0

    # With eta=0.0, some ones expected
    dc2 = DataConfig(
        raw_data_root=tmp_path,
        cache_root=tmp_path,
        instruments=[inst],
        instability_threshold_eta=0.0,
    )
    builder2 = LOBDatasetBuilder(dc2)
    tr2, va2, te2, _ = builder2.build_splits(inst, label_key="instability_s_1")
    assert int(tr2["labels"].sum() + va2["labels"].sum() + te2["labels"].sum()) > 0
