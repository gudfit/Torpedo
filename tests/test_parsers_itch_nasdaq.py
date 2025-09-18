import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def _be_u48(x: int) -> bytes:
    return int(x).to_bytes(6, "big")


def _pad_stock(s: str) -> bytes:
    b = s.encode("ascii")
    return (b + b" " * 8)[:8]


def test_parse_nasdaq_itch_core_messages(tmp_path: Path):
    p = tmp_path / "nasdaq_itch_50.bin"
    with p.open("wb") as f:
        # S: System event
        f.write(b"S")
        f.write(struct.pack(">H", 0))
        f.write(struct.pack(">H", 0))
        f.write(_be_u48(5))
        f.write(b"O")
        # R: Stock directory
        f.write(b"R")
        f.write(struct.pack(">H", 0))
        f.write(struct.pack(">H", 0))
        f.write(_be_u48(6))
        f.write(_pad_stock("TEST"))
        f.write(b"N")  # market cat
        f.write(b"N")  # fin stat
        f.write(struct.pack(">I", 100))
        f.write(b"N")  # round lots only
        f.write(b"A")  # issue class
        f.write(b"AA")  # issue subtype
        f.write(b"P")  # authenticity
        f.write(b"N")  # short sale threshold
        f.write(b"N")  # ipo flag
        f.write(b"1")  # LULD tier
        f.write(b"N")  # ETP flag
        f.write(struct.pack(">I", 1))
        f.write(b"N")  # inverse
        # H: Trading action
        f.write(b"H")
        f.write(struct.pack(">H", 0))
        f.write(struct.pack(">H", 0))
        f.write(_be_u48(9))
        f.write(_pad_stock("TEST"))
        f.write(b"T")  # trading state
        f.write(b" ")  # reserved
        f.write(b"REAS")
        # A: Add (type 'A')
        f.write(b"A")
        f.write(struct.pack(">H", 1))  # stock locate
        f.write(struct.pack(">H", 1))  # tracking
        f.write(_be_u48(10))  # ts
        f.write(struct.pack(">Q", 111))  # order ref
        f.write(b"B")  # side
        f.write(struct.pack(">I", 100))  # shares
        f.write(_pad_stock("TEST"))  # stock 8
        f.write(struct.pack(">I", 10000))  # price 1.0000

        # F: Add with MPID
        f.write(b"F")
        f.write(struct.pack(">H", 2))
        f.write(struct.pack(">H", 2))
        f.write(_be_u48(11))
        f.write(struct.pack(">Q", 112))
        f.write(b"S")
        f.write(struct.pack(">I", 120))
        f.write(_pad_stock("TEST"))
        f.write(struct.pack(">I", 10100))
        f.write(b"MPID")

        # E: Executed
        f.write(b"E")
        f.write(struct.pack(">H", 3))
        f.write(struct.pack(">H", 3))
        f.write(_be_u48(12))
        f.write(struct.pack(">Q", 111))
        f.write(struct.pack(">I", 50))
        f.write(struct.pack(">Q", 5000))

        # C: Executed with price
        f.write(b"C")
        f.write(struct.pack(">H", 4))
        f.write(struct.pack(">H", 4))
        f.write(_be_u48(13))
        f.write(struct.pack(">Q", 111))
        f.write(struct.pack(">I", 20))
        f.write(struct.pack(">Q", 6000))
        f.write(b"\x01")  # printable
        f.write(struct.pack(">I", 10020))

        # X: Cancel
        f.write(b"X")
        f.write(struct.pack(">H", 5))
        f.write(struct.pack(">H", 5))
        f.write(_be_u48(14))
        f.write(struct.pack(">Q", 112))
        f.write(struct.pack(">I", 10))

        # D: Delete
        f.write(b"D")
        f.write(struct.pack(">H", 6))
        f.write(struct.pack(">H", 6))
        f.write(_be_u48(15))
        f.write(struct.pack(">Q", 112))

        # U: Replace
        f.write(b"U")
        f.write(struct.pack(">H", 7))
        f.write(struct.pack(">H", 7))
        f.write(_be_u48(16))
        f.write(struct.pack(">Q", 111))  # orig
        f.write(struct.pack(">Q", 113))  # new
        f.write(struct.pack(">I", 70))
        f.write(struct.pack(">I", 10050))

        # P: Trade (non-cross)
        f.write(b"P")
        f.write(struct.pack(">H", 8))
        f.write(struct.pack(">H", 8))
        f.write(_be_u48(17))
        f.write(struct.pack(">Q", 113))
        f.write(b"B")
        f.write(struct.pack(">I", 90))
        f.write(_pad_stock("TEST"))
        f.write(struct.pack(">I", 10030))
        f.write(struct.pack(">Q", 9000))
        # Q: Cross trade
        f.write(b"Q")
        f.write(struct.pack(">H", 9))
        f.write(struct.pack(">H", 9))
        f.write(_be_u48(18))
        f.write(struct.pack(">Q", 12345))  # shares u64
        f.write(_pad_stock("TEST"))
        f.write(struct.pack(">I", 10040))
        f.write(struct.pack(">Q", 9100))
        f.write(b"O")
        # B: Broken trade
        f.write(b"B")
        f.write(struct.pack(">H", 10))
        f.write(struct.pack(">H", 10))
        f.write(_be_u48(19))
        f.write(struct.pack(">Q", 9100))

    rows = ingest.parse_itch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-itch-5.0")
    assert isinstance(rows, list)
    assert len(rows) >= 9
    types = {r.get("event_type") for r in rows if isinstance(r, dict)}
    # Allow META (trading action) rows in addition to core canonical types
    assert {"LO+", "LO-", "MO+", "CX+", "META"}.issuperset(types)
