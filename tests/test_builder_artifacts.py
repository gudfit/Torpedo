import json
from pathlib import Path

import numpy as np
import pandas as pd

from torpedocode.data.loader import LOBDatasetBuilder
from torpedocode.config import DataConfig, TopologyConfig


def test_builder_splits_and_artifacts(tmp_path):
    # Create a tiny cached parquet for an instrument
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return  # Skip if parquet writer not available

    instrument = "TESTSYM"
    cache_file = tmp_path / f"{instrument}.parquet"
    ts = pd.date_range("2025-01-01", periods=120, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * 30,
            "size": np.abs(np.random.randn(120)).astype(float),
            "price": 100 + np.cumsum(np.random.randn(120)).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(120).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(120).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=120),
            "ask_size_1": np.random.randint(1, 100, size=120),
        }
    )
    df.to_parquet(cache_file, index=False)

    data = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=[instrument], levels=1)
    builder = LOBDatasetBuilder(data)
    art_dir = tmp_path / "artifacts"
    train, val, test, scaler = builder.build_splits(
        instrument,
        label_key="instability_s_1",
        topology=TopologyConfig(window_sizes_s=[1]),
        topo_stride=5,
        artifact_dir=art_dir,
    )

    # Check artifacts
    schema_path = art_dir / "feature_schema.json"
    scaler_path = art_dir / "scaler_schema.json"
    assert schema_path.exists() and scaler_path.exists()
    schema = json.loads(schema_path.read_text())
    assert "feature_names" in schema and isinstance(schema["feature_names"], list)
    assert "topology" in schema

