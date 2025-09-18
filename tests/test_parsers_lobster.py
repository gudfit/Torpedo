from pathlib import Path

import numpy as np
import pandas as pd

from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_lobster_parser_pair(tmp_path: Path):
    # Create small LOBSTER-like message and orderbook CSVs (L=2)
    msg = tmp_path / "TEST_20250101_093000_160000_message_2.csv"
    book = tmp_path / "TEST_20250101_093000_160000_orderbook_2.csv"
    # time,type,order_id,size,price,direction
    msg_lines = [
        [0.0, 1, 1001, 10, 1.0000, 1],   # add buy at 1.0000
        [0.5, 4, 1001, 5, 1.0000, 1],    # execute 5
        [1.0, 3, 1001, 5, 1.0000, 1],    # cancel 5
    ]
    pd.DataFrame(msg_lines).to_csv(msg, index=False, header=False)

    # orderbook columns: ask_p1, ask_s1, ask_p2, ask_s2, bid_p1, bid_s1, bid_p2, bid_s2
    book_rows = [
        [1.0010, 10, 1.0020, 20, 0.9990, 10, 0.9980, 20],
        [1.0010, 10, 1.0020, 20, 0.9995, 8,  0.9980, 20],
        [1.0015, 12, 1.0025, 22, 0.9990, 10, 0.9985, 18],
    ]
    pd.DataFrame(book_rows).to_csv(book, index=False, header=False)

    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=["TEST"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([msg, book], instrument="TEST", tick_size=0.0005)
    assert not df.empty
    assert {"timestamp", "event_type", "price", "size", "symbol", "venue"}.issubset(df.columns)
    # Check that prices are multiples of tick within tolerance
    p = df["price"].dropna().astype(float)
    ratios = p / 0.0005
    assert np.all(np.isclose(ratios, np.round(ratios), atol=1e-9))
