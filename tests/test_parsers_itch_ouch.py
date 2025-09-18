import struct
from pathlib import Path

from torpedocode.data.preprocessing import LOBPreprocessor
from torpedocode.config import DataConfig
import pandas as pd


def _write_itch_minimal(path: Path):
    with path.open("wb") as f:
        # ts, 'A', order_id, side 'B', shares, price_i, stockhash
        f.write(struct.pack("<Q", 1))
        f.write(b"A")
        f.write(struct.pack("<Q", 111))
        f.write(b"B")
        f.write(struct.pack("<I", 10))
        f.write(struct.pack("<Q", int(10000)))  # 1.0000 in 1e-4
        f.write(struct.pack("<Q", 0))
        # ts, 'F' (add attributed), order_id, side 'S', shares, price_i, mpid(4 bytes)
        f.write(struct.pack("<Q", 1))
        f.write(b"F")
        f.write(struct.pack("<Q", 112))
        f.write(b"S")
        f.write(struct.pack("<I", 12))
        f.write(struct.pack("<Q", int(10100)))
        f.write(struct.pack("<I", 0))
        # ts, 'E', order_id, executed
        f.write(struct.pack("<Q", 2))
        f.write(b"E")
        f.write(struct.pack("<Q", 111))
        f.write(struct.pack("<I", 5))
        # ts, 'U' replace, orig_id, new_id, shares, price_i
        f.write(struct.pack("<Q", 2))
        f.write(b"U")
        f.write(struct.pack("<Q", 111))
        f.write(struct.pack("<Q", 113))
        f.write(struct.pack("<I", 7))
        f.write(struct.pack("<Q", int(10050)))


def _write_ouch_minimal(path: Path):
    with path.open("wb") as f:
        # ts, 'O', client_id, side 'S', shares, price_i
        f.write(struct.pack("<Q", 3))
        f.write(b"O")
        f.write(struct.pack("<Q", 222))
        f.write(b"S")
        f.write(struct.pack("<I", 7))
        f.write(struct.pack("<Q", int(9999)))
        # ts, 'U' replace
        f.write(struct.pack("<Q", 3))
        f.write(b"U")
        f.write(struct.pack("<Q", 222))
        f.write(struct.pack("<Q", 223))
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", int(10001)))
        # ts, 'X', client_id, canceled
        f.write(struct.pack("<Q", 4))
        f.write(b"X")
        f.write(struct.pack("<Q", 222))
        f.write(struct.pack("<I", 2))


def test_itch_ouch_parsers(tmp_path: Path):
    itch = tmp_path / "sample.itch"
    ouch = tmp_path / "sample.ouch"
    _write_itch_minimal(itch)
    _write_ouch_minimal(ouch)

    cfg = DataConfig(raw_data_root=tmp_path, cache_root=tmp_path, instruments=["TEST"], drop_auctions=False)
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise([itch, ouch], instrument="TEST", tick_size=0.0001)
    assert not df.empty
    assert set(df["event_type"]).issubset({"LO+", "LO-", "MO+", "CX+"})
    # Timestamps should be ascending after merge
    assert (df["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()
