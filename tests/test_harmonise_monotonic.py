import json
from pathlib import Path
import pandas as pd


def test_harmonise_enforce_monotonic(tmp_path: Path):
    from torpedocode.data.preprocess import harmonise_ndjson, HarmoniseConfig

    nd = tmp_path / "monotonic.ndjson"
    rows = [
        {"timestamp": "2025-01-01T00:00:02Z", "event_type": "LO+", "price": 100.0, "size": 1.0},
        {"timestamp": "2025-01-01T00:00:00Z", "event_type": "MO-", "price": 99.9, "size": 2.0},
        {"timestamp": "2025-01-01T00:00:01Z", "event_type": "LO-", "price": 100.1, "size": 1.0},
    ]
    nd.write_text("\n".join(json.dumps(r) for r in rows))

    cfg = HarmoniseConfig(time_zone="UTC", quality_checks=True, enforce_monotonic_timestamps=True, drop_auctions=False)
    df = harmonise_ndjson(nd, cfg=cfg)
    ts = pd.to_datetime(df["timestamp"], utc=True).to_list()
    assert ts == sorted(ts)

