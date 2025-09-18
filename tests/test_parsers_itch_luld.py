import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def _be_u48(x: int) -> bytes:
    return int(x).to_bytes(6, "big")


def _pad_stock(s: str) -> bytes:
    b = s.encode("ascii")
    return (b + b" " * 8)[:8]


def test_parse_nasdaq_itch_luld_skip(tmp_path: Path):
    p = tmp_path / "nasdaq_itch_50_luld.bin"
    with p.open("wb") as f:
        # A: Add order
        f.write(b"A")
        f.write(struct.pack(">H", 1))  # stock locate
        f.write(struct.pack(">H", 1))  # tracking
        f.write(_be_u48(10))
        f.write(struct.pack(">Q", 111))  # order ref
        f.write(b"B")  # side
        f.write(struct.pack(">I", 100))  # shares
        f.write(_pad_stock("TEST"))
        f.write(struct.pack(">I", 10000))  # price

        # L: LULD auction collar update (headers + stock + 3 prices)
        f.write(b"L")
        f.write(struct.pack(">H", 2))  # stock locate
        f.write(struct.pack(">H", 2))  # tracking
        f.write(_be_u48(11))  # ts
        f.write(_pad_stock("TEST"))
        f.write(struct.pack(">I", 10000))  # reference price
        f.write(struct.pack(">I", 10500))  # upper collar
        f.write(struct.pack(">I", 9500))  # lower collar

        # E: Executed (should still be parsed after skipping L)
        f.write(b"E")
        f.write(struct.pack(">H", 3))
        f.write(struct.pack(">H", 3))
        f.write(_be_u48(12))
        f.write(struct.pack(">Q", 111))
        f.write(struct.pack(">I", 50))
        f.write(struct.pack(">Q", 5000))

    rows = ingest.parse_itch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-itch-5.0")
    assert isinstance(rows, list)
    # Expect one LO+ from Add, one MO+ from Executed; LULD is skipped
    types = [r.get("event_type") for r in rows if isinstance(r, dict)]
    assert types.count("LO+") == 1
    assert types.count("MO+") == 1
