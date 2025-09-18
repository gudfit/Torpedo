import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def _write_minimal_itch(path: Path):
    with path.open("wb") as f:
        # ts=1e9 ns (~1s), 'A' add order
        f.write(struct.pack("<Q", 1_000_000_000))
        f.write(b"A")
        f.write(struct.pack("<Q", 1))  # order ref
        f.write(b"B")
        f.write(struct.pack("<I", 10))
        f.write(struct.pack("<Q", 10000))
        f.write(struct.pack("<Q", 0))


def test_parse_itch_optional_kwargs_utc_and_adjust(tmp_path: Path):
    p = tmp_path / "min.itch"
    _write_minimal_itch(p)
    # If the native module doesn't support new kwargs, skip
    try:
        rows = ingest.parse_itch_file(
            str(p),
            tick_size=0.0001,
            symbol="TEST",
            spec="nasdaq-itch-5.0",
            session_date="2020-01-02",
            tz_offset_seconds=0,
            price_adjust_factor=2.0,
            price_adjust_mode="multiply",
        )
    except TypeError:
        pytest.skip("native module not rebuilt with optional kwargs")
    assert isinstance(rows, list) and len(rows) >= 1
    r0 = rows[0]
    assert isinstance(r0, dict)
    # UTC epoch ns: 2020-01-02 00:00:00 + 1s
    assert r0["timestamp"] >= 1_577_914_401_000_000_000
    # price scaled with adjust factor
    assert pytest.approx(r0["price"], rel=1e-9) == 2.0
