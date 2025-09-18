import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def test_iter_wrappers_return_iterables(tmp_path: Path):
    # Build minimal ITCH-native test payload (little-endian minimal format)
    itch = tmp_path / "iter_min.itch"
    with itch.open("wb") as f:
        # ts, 'A', order_ref, side, shares, price, stock_id
        f.write(struct.pack("<Q", 1))
        f.write(b"A")
        f.write(struct.pack("<Q", 1))
        f.write(b"B")
        f.write(struct.pack("<I", 10))
        f.write(struct.pack("<Q", 10000))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 2))
        f.write(b"E")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 5))

    if not hasattr(ingest, "parse_itch_iter"):
        pytest.skip("native module not rebuilt with parse_itch_iter")
    it = ingest.parse_itch_iter(str(itch), tick_size=0.0001, symbol="TEST", spec="nasdaq-itch-5.0")
    rows = list(it)
    assert rows and isinstance(rows[0], dict)
    assert "level_inferred" in rows[0]

    # OUCH iter wrapper
    ouch = tmp_path / "iter_min.ouch"
    with ouch.open("wb") as f:
        # ts, 'O', order_ref, side, shares, price
        f.write(struct.pack("<Q", 1))
        f.write(b"O")
        f.write(struct.pack("<Q", 1))
        f.write(b"B")
        f.write(struct.pack("<I", 10))
        f.write(struct.pack("<Q", 10000))

    if not hasattr(ingest, "parse_ouch_iter"):
        pytest.skip("native module not rebuilt with parse_ouch_iter")
    it2 = ingest.parse_ouch_iter(str(ouch), tick_size=0.0001, symbol="TEST", spec="nasdaq-ouch-4.2")
    rows2 = list(it2)
    assert rows2 and isinstance(rows2[0], dict)
    assert "level_inferred" in rows2[0]
