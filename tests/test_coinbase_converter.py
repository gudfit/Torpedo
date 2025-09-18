from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.coinbase_to_ndjson import convert_lines
from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_coinbase_converter_to_canonical(tmp_path: Path):
    raw_lines = [
        '{"type":"match","time":"2025-01-01T00:00:00Z","product_id":"BTC-USD","price":"29000.1","size":"0.123"}',
        '{"type":"l2update","time":"2025-01-01T00:00:01Z","product_id":"BTC-USD","changes":[["buy","28999.9","1.2"],["sell","29000.3","0.8"]]}',
    ]
    out_lines = convert_lines(raw_lines)
    nd = tmp_path / "cb.ndjson"
    nd.write_text("\n".join(out_lines))

    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=["BTC-USD"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([nd], instrument="BTC-USD")
    assert not df.empty
    assert set(["MO+", "LO+", "LO-"]).issuperset(set(df["event_type"]))

