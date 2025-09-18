from pathlib import Path

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.binance_to_ndjson import convert_lines
from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig


def test_binance_converter_to_canonical(tmp_path: Path):
    raw_lines = [
        '{"e":"aggTrade","E":1690000000000,"s":"BTCUSDT","p":"29000.1","q":"0.123"}',
        '{"e":"bookTicker","E":1690000000100,"s":"BTCUSDT","b":"28999.9","B":"1.2","a":"29000.3","A":"0.8"}',
        '{"e":"depthUpdate","E":1690000000200,"s":"BTCUSDT","b":[["28999.8","0.5"]],"a":[["29000.4","0.2"]]}'
    ]
    out_lines = convert_lines(raw_lines)
    # Write to NDJSON
    nd = tmp_path / "bn.ndjson"
    nd.write_text("\n".join(out_lines))

    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=["BTCUSDT"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([nd], instrument="BTCUSDT")
    assert not df.empty
    assert set(["MO+", "LO+", "LO-"]).issuperset(set(df["event_type"]))
