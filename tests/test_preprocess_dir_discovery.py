from pathlib import Path

import pandas as pd

from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_directory_level_discovery_and_pairing(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    msg = raw / "TEST_20250101_message_2.csv"
    book = raw / "TEST_20250101_orderbook_2.csv"

    pd.DataFrame([[0.0, 1, 1, 10, 1.0, 1]]).to_csv(msg, index=False, header=False)
    pd.DataFrame([[1.1, 10, 1.2, 20, 0.9, 10, 0.8, 20]]).to_csv(book, index=False, header=False)

    cfg = DataConfig(raw_data_root=raw, cache_root=tmp_path, instruments=["TEST"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([raw], instrument="TEST")
    assert not df.empty and "event_type" in df.columns

