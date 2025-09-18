import numpy as np
import pandas as pd

from torpedocode.features.lob import build_lob_feature_matrix


def test_returns_are_causal_no_lookahead():
    # Construct mid-price path via best bid/ask that increases linearly
    ts = pd.date_range("2025-01-01", periods=5, freq="s", tz="UTC")
    mid = np.array([100, 102, 104, 106, 108], dtype=float)
    # Create symmetric bid/ask around mid
    bid = mid - 0.05
    ask = mid + 0.05
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "bid_size_1": [10, 10, 10, 10, 10],
            "ask_size_1": [10, 10, 10, 10, 10],
            "bid_price_1": bid,
            "ask_price_1": ask,
            "event_type": ["MO+", "MO-", "LO+", "LO-", "CX+"],
        }
    )

    X, aux = build_lob_feature_matrix(df, levels=1, ret_horizons=(1, 2))
    # Expected causal log returns
    log_mid = np.log(mid)
    r1 = np.zeros_like(log_mid)
    r1[1:] = log_mid[1:] - log_mid[:-1]
    r2 = np.zeros_like(log_mid)
    r2[2:] = log_mid[2:] - log_mid[:-2]
    got = aux["ret"]
    assert got.shape == (5, 2)
    assert np.allclose(got[:, 0], r1.astype(np.float32))
    assert np.allclose(got[:, 1], r2.astype(np.float32))
