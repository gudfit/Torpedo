from pathlib import Path

import pandas as pd

from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_preprocess_harmonise_and_cache(tmp_path):
    # Create an NDJSON file with minimal events
    nd = tmp_path / "sample.ndjson"
    # Use 14:30Z which is 09:30 US/Eastern (regular session), avoiding drop_auctions filter
    lines = [
        '{"timestamp":"2025-01-01T14:30:00Z","event_type":"LO+","price":100.0,"size":10}',
        '{"timestamp":"2025-01-01T14:30:01Z","event_type":"MO-","price":99.9,"size":5}',
        '{"timestamp":"2025-01-01T14:30:02Z","event_type":"CX+","price":100.1,"size":2}',
    ]
    nd.write_text("\n".join(lines))

    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=["TEST"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([nd], instrument="TEST")
    assert not df.empty
    df2 = pp.add_instability_labels(df)

    # Cache may require pyarrow; skip cache test if missing
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return
    p = pp.cache(df2, instrument="TEST")
    assert p.exists()
