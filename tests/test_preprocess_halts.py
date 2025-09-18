import pandas as pd
import numpy as np

from torpedocode.data.preprocess import drop_auction_and_halt_intervals


def _ts(s):
    # Helper: parse to UTC-aware timestamp
    return pd.to_datetime(s, utc=True)


def test_drop_rows_with_explicit_halt_flags():
    # Construct a small intraday series within regular session with an explicit halt code
    ts = [
        _ts("2024-06-03 13:31:00Z"),  # 09:31 ET
        _ts("2024-06-03 13:31:01Z"),
        _ts("2024-06-03 13:31:02Z"),
        _ts("2024-06-03 13:31:03Z"),
    ]
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["LO+", "MO+", "LO-", "CX+"],
            "size": [1.0, 1.0, 1.0, 1.0],
            "price": [10.0, 10.1, 9.9, np.nan],
            # Mark the third row as halted
            "trading_state": ["T", "T", "H", "T"],
        }
    )

    out = drop_auction_and_halt_intervals(df, session_start="09:30", session_end="16:00")
    # Expect the row with trading_state == 'H' to be removed
    assert len(out) == 3
    assert not any(
        out.get("trading_state", pd.Series([], dtype=str)).astype(str).str.upper() == "H"
    )


def test_halt_gap_heuristic_drops_edges_of_long_gap_same_day():
    # Build timestamps with a long intra-day gap (> 120s) and ensure edge rows are dropped
    # All times are within regular session (ET)
    ts = [
        _ts("2024-06-03 13:31:00Z"),  # 09:31:00 ET
        _ts("2024-06-03 13:31:01Z"),  # small gap
        _ts("2024-06-03 13:35:05Z"),  # large gap from previous (> 4 minutes)
        _ts("2024-06-03 13:35:06Z"),
        _ts("2024-06-03 13:35:07Z"),
    ]
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["LO+", "MO+", "LO-", "CX+", "LO+"],
            "size": [1.0, 1.0, 1.0, 1.0, 1.0],
            "price": [10.0, 10.1, 9.9, np.nan, 10.2],
        }
    )
    out = drop_auction_and_halt_intervals(df, session_start="09:30", session_end="16:00")
    # Indices 1 (before long gap) and 2 (after long gap) should be dropped by heuristic
    # Remaining should be indices 0, 3, 4 relative to original
    out_ts = pd.to_datetime(out["timestamp"], utc=True).tolist()
    assert _ts("2024-06-03 13:31:01Z") not in out_ts  # dropped
    assert _ts("2024-06-03 13:35:05Z") not in out_ts  # dropped
    assert _ts("2024-06-03 13:31:00Z") in out_ts
    assert _ts("2024-06-03 13:35:06Z") in out_ts
    assert _ts("2024-06-03 13:35:07Z") in out_ts
