import numpy as np
import pandas as pd

from torpedocode.config import DataConfig, TopologyConfig
from pathlib import Path
from torpedocode.data.loader import LOBDatasetBuilder


class DummyTopo:
    last_series = None

    def __init__(self, cfg):
        self.cfg = cfg

    def rolling_transform(self, timestamps, series, *, window_sizes_s=None, stride=1):
        # capture the series passed (scaled or raw)
        DummyTopo.last_series = np.array(series, copy=True)
        T = series.shape[0]
        # return zeros with a tiny dim to keep downstream happy
        return np.zeros((T, 2), dtype=np.float32)


def _synthetic_df(T=20, L=2):
    ts = pd.date_range("2025-01-01", periods=T, freq="s", tz="UTC")
    data = {"timestamp": ts, "event_type": ["MO+"] * T}
    for l in range(1, L + 1):
        data[f"bid_size_{l}"] = np.linspace(1, 2, T) * l
        data[f"ask_size_{l}"] = np.linspace(2, 3, T) * (l + 1)
        data[f"bid_price_{l}"] = np.ones(T) * (100 + l)
        data[f"ask_price_{l}"] = np.ones(T) * (100.1 + l)
    data["size"] = np.ones(T) * 1.0
    return pd.DataFrame(data)


def test_tda_uses_raw_for_vr_when_enabled(monkeypatch):
    # Patch MarketDataLoader.load_events to return synthetic DF
    from torpedocode.data import loader as loader_mod

    df = _synthetic_df(T=20, L=2)

    def _load_events(self, instrument: str):
        return df

    monkeypatch.setattr(loader_mod.MarketDataLoader, "load_events", _load_events, raising=True)
    # Patch TopologicalFeatureGenerator to DummyTopo
    monkeypatch.setattr(loader_mod, "TopologicalFeatureGenerator", DummyTopo, raising=True)

    data = DataConfig(raw_data_root=Path("."), cache_root=Path("."), instruments=["X"])  # type: ignore
    builder = LOBDatasetBuilder(data)
    topo = TopologyConfig(complex_type="vietoris_rips", use_raw_for_vr=True, window_sizes_s=[1])
    # Build a sequence to reconstruct the expected raw features
    rec = builder.build_sequence("X")
    tr, va, te, scaler = builder.build_splits(
        "X", label_key="instability_s_1", topology=topo, topo_stride=1, artifact_dir=None
    )
    # Test split slice indices
    T = len(df)
    v0 = int(0.8 * T)
    # Reconstruct raw features for the test slice using the same builder pathway
    raw_slice = rec["features_raw"][v0:]
    seen = DummyTopo.last_series
    assert seen is not None and seen.shape == raw_slice.shape
    assert np.allclose(seen, raw_slice)


def test_tda_uses_scaled_for_vr_when_disabled(monkeypatch):
    from torpedocode.data import loader as loader_mod

    df = _synthetic_df(T=20, L=2)

    def _load_events(self, instrument: str):
        return df

    monkeypatch.setattr(loader_mod.MarketDataLoader, "load_events", _load_events, raising=True)
    monkeypatch.setattr(loader_mod, "TopologicalFeatureGenerator", DummyTopo, raising=True)

    data = DataConfig(raw_data_root=Path("."), cache_root=Path("."), instruments=["X"])  # type: ignore
    builder = LOBDatasetBuilder(data)
    topo = TopologyConfig(complex_type="vietoris_rips", use_raw_for_vr=False, window_sizes_s=[1])
    tr, va, te, scaler = builder.build_splits(
        "X", label_key="instability_s_1", topology=topo, topo_stride=1, artifact_dir=None
    )
    T = len(df)
    v0 = int(0.8 * T)
    # Scaled features equal split["features"] for the same slice
    scaled_slice = te["features"]  # builder returns scaled in splits
    seen = DummyTopo.last_series
    assert seen is not None and seen.shape == scaled_slice.shape
    assert np.allclose(seen, scaled_slice)
