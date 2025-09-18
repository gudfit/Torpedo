import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def _token(s: str) -> bytes:
    b = s.encode("ascii")
    return (b + b" " * 14)[:14]


def test_parse_nasdaq_ouch_extra_messages(tmp_path: Path):
    p = tmp_path / "nasdaq_ouch_42_extra.bin"
    with p.open("wb") as f:
        # A: Accepted BUY @ 1.0010 x 100
        f.write(b"A")
        f.write(_token("TOK100"))
        f.write(b"B")
        f.write(struct.pack(">I", 100))
        f.write((b"TEST" + b" " * 4)[:8])  # stock padded to 8
        f.write(struct.pack(">I", 10010))

        # P: Priority update (token, new_price, new_shares) — should be consumed/skipped
        f.write(b"P")
        f.write(_token("TOK100"))
        f.write(struct.pack(">I", 10020))
        f.write(struct.pack(">I", 120))

        # D: System cancel (token)
        f.write(b"D")
        f.write(_token("TOK100"))

        # B: Trade correction (token, match, reason)
        f.write(b"B")
        f.write(_token("TOK100"))
        f.write(struct.pack(">Q", 424242))
        f.write(b"R")

        # T: Trade @ 1.0020 x 50
        f.write(b"T")
        f.write(_token("TOK999"))
        f.write(struct.pack(">I", 50))
        f.write(struct.pack(">I", 10020))
        f.write(struct.pack(">Q", 999999))

    rows = ingest.parse_ouch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-ouch-4.2")
    assert isinstance(rows, list)
    # Expect: A -> LO+, D -> CX+, B -> CX+ (neutralize), T -> MO+
    types = [r.get("event_type") for r in rows if isinstance(r, dict)]
    assert types.count("LO+") == 1
    assert types.count("CX+") >= 2  # D and B map to CX+
    assert "MO+" in types

    # Check tokens and reasons are exposed where applicable
    a = next(r for r in rows if r.get("event_type") == "LO+")
    assert a.get("token") == "TOK100"

    d = next(r for r in rows if r.get("event_type") == "CX+" and r.get("reason") == "SYSTEM_CANCEL")
    assert d.get("token") == "TOK100"

    b = next(
        r
        for r in rows
        if r.get("event_type") == "CX+" and r.get("reason") not in (None, "SYSTEM_CANCEL")
    )
    assert isinstance(b.get("match"), int)
    assert b.get("token") == "TOK100"

    t = next(r for r in rows if r.get("event_type") == "MO+")
    assert isinstance(t.get("match"), int)
    assert t.get("token") == "TOK999"
