import pandas as pd
import numpy as np

from torpedocode.data.preprocess import drop_auction_and_halt_intervals


def _ts(s):
    return pd.to_datetime(s, utc=True)


def test_session_hour_alignment_london_tz():
    # London session 08:00-16:30 in Europe/London. In June (BST, UTC+1), 08:00 is 07:00Z.
    df = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-06-03 06:59:59Z"),  # before open in London local
                _ts("2024-06-03 07:00:00Z"),  # open
                _ts("2024-06-03 07:00:10Z"),
                _ts("2024-06-03 07:00:20Z"),
                _ts("2024-06-03 07:00:30Z"),
            ],
            "event_type": ["LO+"] * 5,
            "size": np.ones(5),
            "price": np.linspace(10.0, 10.1, 5),
        }
    )

    out = drop_auction_and_halt_intervals(
        df,
        session_start="08:00",
        session_end="16:30",
        local_tz="Europe/London",
    )
    ts_out = pd.to_datetime(out["timestamp"], utc=True)
    # Ensure pre-open is excluded
    assert _ts("2024-06-03 06:59:59Z") not in ts_out.tolist()
    # Ensure open time is included
    assert _ts("2024-06-03 07:00:00Z") in ts_out.tolist()
    # And subsequent within-session times are also included
    assert _ts("2024-06-03 07:00:30Z") in ts_out.tolist()
