from pathlib import Path
import pandas as pd

from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_fi2010_converter_to_canonical(tmp_path: Path):
    # Create minimal FI-2010-like CSV
    csv = tmp_path / "fi2010.csv"
    df = pd.DataFrame(
        {
            "ask_price_1": [100.1, 100.2],
            "ask_size_1": [10, 11],
            "bid_price_1": [99.9, 100.0],
            "bid_size_1": [9, 8],
        }
    )
    df.to_csv(csv, index=False)

    # Convert to NDJSON using script function
    from scripts.fi2010_to_ndjson import convert

    out = tmp_path / "fi2010.ndjson"
    n = convert(csv, out, symbol="FI_TEST", dt_ns=1_000_000)
    assert n == 4  # 2 rows -> 4 events (bid+ask each)

    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=["FI_TEST"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df_ev = pp.harmonise([out], instrument="FI_TEST")
    assert not df_ev.empty
    assert set(["LO+", "LO-"]).issuperset(set(df_ev["event_type"]))

