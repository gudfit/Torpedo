import numpy as np
import pandas as pd
from pathlib import Path

import torch

from torpedocode.config import DataConfig, ModelConfig, TopologyConfig
from torpedocode.data.loader import LOBDatasetBuilder
from torpedocode.models.hybrid import HybridLOBModel


def _write_cache(tmp: Path, name: str):
    ts = pd.date_range("2025-01-01", periods=50, freq="s", tz="UTC")
    # Alternate LO/CX with levels and MO to ensure multiple expanded types
    ev = (["LO+", "CX+", "LO-", "CX-"] * 10) + (["MO+", "MO-"] * 5)
    lvl = ([1, 2, 1, 2] * 10) + ([np.nan, np.nan] * 5)
    df = pd.DataFrame(
        {
            "timestamp": ts[: len(ev)],
            "event_type": ev,
            "level": lvl,
            "size": np.abs(np.random.randn(len(ev))).astype(float),
            "price": 100 + np.cumsum(np.random.randn(len(ev))).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(len(ev)).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(len(ev)).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 50, size=len(ev)),
            "ask_size_1": np.random.randint(1, 50, size=len(ev)),
        }
    )
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False
    df.to_parquet(tmp / f"{name}.parquet", index=False)
    return True


def test_model_forward_with_expanded_event_types(tmp_path: Path):
    inst = "EXPMODEL"
    if not _write_cache(tmp_path, inst):
        import pytest

        pytest.skip("pyarrow not available")
    cfg = DataConfig(
        raw_data_root=tmp_path,
        cache_root=tmp_path,
        instruments=[inst],
        levels=1,
        expand_event_types_by_level=True,
    )
    b = LOBDatasetBuilder(cfg)
    tr, va, te, _ = b.build_splits(inst, label_key="instability_s_1", topology=TopologyConfig())

    def infer_types(*splits):
        mx = 0
        found = False
        for s in splits:
            et = s.get("event_type_ids")
            if et is not None and len(et) > 0:
                mx = max(mx, int(np.max(et)))
                found = True
        return (mx + 1) if found else 6

    num_types = infer_types(tr, va, te)
    F = tr["features"].shape[1]
    Z = tr["topology"].shape[1]
    model = HybridLOBModel(
        F,
        Z,
        num_event_types=int(num_types),
        config=ModelConfig(hidden_size=32, num_layers=1, include_market_embedding=False),
    )
    xb = torch.from_numpy(tr["features"]).unsqueeze(0)
    zb = torch.from_numpy(tr["topology"]).unsqueeze(0)
    out = model(xb, zb)
    assert isinstance(out.intensities, dict)
    assert len(out.intensities) == int(num_types)
