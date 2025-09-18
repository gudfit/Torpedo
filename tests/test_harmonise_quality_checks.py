import json
from pathlib import Path

import pandas as pd


def test_harmonise_quality_checks_drop_invalid(tmp_path: Path):
    from torpedocode.data.preprocess import harmonise_ndjson, HarmoniseConfig

    nd = tmp_path / "x.ndjson"
    rows = [
        {"timestamp": "2025-01-01T00:00:00Z", "event_type": "LO+", "price": 100.0, "size": 1.0},
        {"timestamp": "not-a-time", "event_type": "MO-", "price": 99.9, "size": 2.0},  # drop NaT
        {"timestamp": "2025-01-01T00:00:01Z", "event_type": "LO-", "price": 0.0, "size": 1.0},  # drop price <= 0
        {"timestamp": "2025-01-01T00:00:02Z", "event_type": "CX+", "price": 100.1, "size": -1.0},  # drop size < 0
    ]
    nd.write_text("\n".join(json.dumps(r) for r in rows))

    cfg = HarmoniseConfig(time_zone="UTC", quality_checks=True, min_price=1e-12, min_size=0.0, drop_auctions=False)
    df = harmonise_ndjson(nd, cfg=cfg)
    # Only the first valid row should remain
    assert len(df) == 1
    assert pd.to_datetime(df.loc[0, "timestamp"], utc=True) == pd.Timestamp("2025-01-01T00:00:00Z")


def test_harmonise_quality_checks_can_disable(tmp_path: Path):
    from torpedocode.data.preprocess import harmonise_ndjson, HarmoniseConfig

    nd = tmp_path / "y.ndjson"
    rows = [
        {"timestamp": "2025-01-01T00:00:01Z", "event_type": "LO-", "price": 0.0, "size": 1.0},
        {"timestamp": "2025-01-01T00:00:02Z", "event_type": "CX+", "price": 100.1, "size": -1.0},
    ]
    nd.write_text("\n".join(json.dumps(r) for r in rows))

    cfg = HarmoniseConfig(time_zone="UTC", quality_checks=False, drop_auctions=False)
    df = harmonise_ndjson(nd, cfg=cfg)
    # Without quality checks, invalid rows are retained
    assert len(df) == 2

