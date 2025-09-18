import numpy as np
import pandas as pd

from torpedocode.features.lob import build_lob_feature_matrix


def test_queue_ages_all_levels_shapes():
    T, L = 6, 3
    ts = pd.date_range("2025-01-01", periods=T, freq="s", tz="UTC")
    data = {"timestamp": ts, "event_type": ["MO+"] * T}
    for l in range(1, L + 1):
        data[f"bid_size_{l}"] = np.ones(T) * l
        data[f"ask_size_{l}"] = np.ones(T) * (l + 1)
        data[f"bid_price_{l}"] = np.ones(T) * (100 + l)
        data[f"ask_price_{l}"] = np.ones(T) * (100.1 + l)
    df = pd.DataFrame(data)
    X, aux = build_lob_feature_matrix(df, levels=L)
    assert aux["queue_age_b"].shape == (T, L)
    assert aux["queue_age_a"].shape == (T, L)
    # Level-1 compatibility
    assert np.allclose(aux["queue_age_b1"], aux["queue_age_b"][:, 0])
    assert np.allclose(aux["queue_age_a1"], aux["queue_age_a"][:, 0])
