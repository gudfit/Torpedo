import pytest
from pathlib import Path
import struct

ingest = pytest.importorskip("torpedocode_ingest")


def test_rust_native_itch_parser(tmp_path: Path):
    p = tmp_path / "native.itch"
    with p.open("wb") as f:
        # A add
        f.write(struct.pack("<Q", 1)); f.write(b"A"); f.write(struct.pack("<Q", 1)); f.write(b"B"); f.write(struct.pack("<I", 10)); f.write(struct.pack("<Q", 10000)); f.write(struct.pack("<Q", 0))
        # F add attributed
        f.write(struct.pack("<Q", 2)); f.write(b"F"); f.write(struct.pack("<Q", 2)); f.write(b"S"); f.write(struct.pack("<I", 12)); f.write(struct.pack("<Q", 10100)); f.write(struct.pack("<I", 0))
        # E execute
        f.write(struct.pack("<Q", 3)); f.write(b"E"); f.write(struct.pack("<Q", 1)); f.write(struct.pack("<I", 5))
        # C execute with price
        f.write(struct.pack("<Q", 4)); f.write(b"C"); f.write(struct.pack("<Q", 1)); f.write(struct.pack("<I", 2)); f.write(struct.pack("<Q", 10020))
        # U replace
        f.write(struct.pack("<Q", 5)); f.write(b"U"); f.write(struct.pack("<Q", 1)); f.write(struct.pack("<Q", 3)); f.write(struct.pack("<I", 7)); f.write(struct.pack("<Q", 10050))
        # X cancel
        f.write(struct.pack("<Q", 6)); f.write(b"X"); f.write(struct.pack("<Q", 3)); f.write(struct.pack("<I", 1))
        # D delete
        f.write(struct.pack("<Q", 7)); f.write(b"D"); f.write(struct.pack("<Q", 2))
        # P trade
        f.write(struct.pack("<Q", 8)); f.write(b"P"); f.write(struct.pack("<Q", 111)); f.write(b"B"); f.write(struct.pack("<I", 9)); f.write(struct.pack("<Q", 10030))

    rows = ingest.parse_itch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-itch-5.0")
    # Expect at least 8 rows mapped
    assert isinstance(rows, list)
    assert len(rows) >= 8


def test_rust_native_ouch_parser(tmp_path: Path):
    p = tmp_path / "native.ouch"
    with p.open("wb") as f:
        # O enter order
        f.write(struct.pack("<Q", 1)); f.write(b"O"); f.write(struct.pack("<Q", 1)); f.write(b"B"); f.write(struct.pack("<I", 10)); f.write(struct.pack("<Q", 10000))
        # U replace
        f.write(struct.pack("<Q", 2)); f.write(b"U"); f.write(struct.pack("<Q", 1)); f.write(struct.pack("<Q", 2)); f.write(struct.pack("<I", 8)); f.write(struct.pack("<Q", 10020))
        # E execute
        f.write(struct.pack("<Q", 3)); f.write(b"E"); f.write(struct.pack("<Q", 2)); f.write(struct.pack("<I", 5))
        # X cancel
        f.write(struct.pack("<Q", 4)); f.write(b"X"); f.write(struct.pack("<Q", 2)); f.write(struct.pack("<I", 5))
        # D delete
        f.write(struct.pack("<Q", 5)); f.write(b"D"); f.write(struct.pack("<Q", 2))

    rows = ingest.parse_ouch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-ouch-4.2")
    assert isinstance(rows, list)
    assert len(rows) >= 5

