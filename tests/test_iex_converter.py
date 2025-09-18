from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.iex_to_ndjson import convert_lines
from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_iex_converter_to_canonical(tmp_path: Path):
    # Minimal IEX-like messages: one book snapshot and one trade
    raw_lines = [
        json.dumps(
            {
                "type": "book",
                "timestamp": "2025-01-01T00:00:00Z",
                "symbol": "XYZ",
                "bids": [[100.0, 10.0]],
                "asks": [[100.5, 9.0]],
            }
        ),
        json.dumps(
            {
                "type": "trade",
                "timestamp": "2025-01-01T00:00:01Z",
                "symbol": "XYZ",
                "price": 100.25,
                "size": 1.5,
            }
        ),
    ]

    out_lines = convert_lines(raw_lines, default_symbol="XYZ")
    nd = tmp_path / "iex.ndjson"
    nd.write_text("\n".join(out_lines))

    cfg = DataConfig(
        raw_data_root=tmp_path, cache_root=tmp_path, instruments=["XYZ"], drop_auctions=False
    )
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([nd], instrument="XYZ")
    assert not df.empty
    # Expect LO+/LO- from book, and MO+ from trade
    assert set(["LO+", "LO-", "MO+"]).issuperset(set(df["event_type"]))
