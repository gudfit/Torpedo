import numpy as np
import pandas as pd

from torpedocode.features.lob import build_lob_feature_matrix


def test_build_lob_feature_matrix_minimal():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="s", tz="UTC"),
            "bid_size_1": [10, 11, 12, 13, 14],
            "ask_size_1": [9, 9, 10, 10, 11],
            "bid_price_1": [100, 100, 100, 100, 100],
            "ask_price_1": [100.1, 100.1, 100.1, 100.1, 100.1],
            "event_type": ["MO+", "MO-", "LO+", "LO-", "CX+"],
        }
    )
    X, aux = build_lob_feature_matrix(df, levels=1)
    assert X.shape == (5, 2)
    assert aux["spreads"].shape == (5,)
    assert aux["imbalance@k"].shape[1] >= 1
    assert np.all(np.isfinite(aux["delta_t"]))
