import pandas as pd
import numpy as np

from torpedocode.features.lob import build_lob_feature_matrix


def test_queue_age_uses_last_update_columns():
    # Build a tiny frame with explicit last update timestamps
    ts = pd.to_datetime([
        "2025-01-01T10:00:00Z",
        "2025-01-01T10:00:10Z",
        "2025-01-01T10:00:25Z",
    ], utc=True)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["LO+", "LO+", "LO+"],
            "bid_price_1": [100.0, 100.0, 100.0],
            "ask_price_1": [100.1, 100.1, 100.1],
            "bid_size_1": [10, 10, 10],
            "ask_size_1": [12, 12, 12],
            # last update happens exactly at t0, then at (t1-5s) for bid, and never for ask
            "last_update_bid_1": [ts[0], ts[1] - pd.Timedelta(seconds=5), ts[1] - pd.Timedelta(seconds=5)],
            "last_update_ask_1": [ts[0], ts[0], ts[0]],
        }
    )

    base, aux = build_lob_feature_matrix(df, levels=1)
    q_b = aux["queue_age_b1"]
    q_a = aux["queue_age_a1"]
    # Ages computed from exact last_update_* columns:
    # t0: 0s for both; t1: 10s for ask, 5s for bid; t2: 25s for ask, 20s for bid
    assert np.isclose(q_b[0], 0.0)
    assert np.isclose(q_a[0], 0.0)
    assert np.isclose(q_b[1], 5.0)
    assert np.isclose(q_a[1], 10.0)
    assert np.isclose(q_b[2], 20.0)
    assert np.isclose(q_a[2], 25.0)
