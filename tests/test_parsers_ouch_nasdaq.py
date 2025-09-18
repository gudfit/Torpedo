import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def _token(s: str) -> bytes:
    b = s.encode("ascii")
    return (b + b" " * 14)[:14]


def _stock(s: str) -> bytes:
    b = s.encode("ascii")
    return (b + b" " * 8)[:8]


def test_parse_nasdaq_ouch_core_messages(tmp_path: Path):
    p = tmp_path / "nasdaq_ouch_42.bin"
    with p.open("wb") as f:
        # S: system event
        f.write(b"S")
        f.write(b"O")
        # U: Replace Order (orig, new, shares, price, tif)
        f.write(b"U")
        f.write(_token("TOK123"))
        f.write(_token("TOK124"))
        f.write(struct.pack(">I", 70))
        f.write(struct.pack(">I", 10050))
        f.write(struct.pack(">I", 0))

        # E: Executed (token, shares, match)
        f.write(b"E")
        f.write(_token("TOK124"))
        f.write(struct.pack(">I", 20))
        f.write(struct.pack(">Q", 1111))

        # X: Cancel (token, shares)
        f.write(b"X")
        f.write(_token("TOK124"))
        f.write(struct.pack(">I", 10))

        # C: Canceled with reason
        f.write(b"C")
        f.write(_token("TOK124"))
        f.write(struct.pack(">I", 5))
        f.write(b"R")

        # R: Rejected with reason
        f.write(b"R")
        f.write(_token("TOKBAD"))
        f.write(b"X")

        # A: accepted
        f.write(b"A")
        f.write(_token("TOK125"))
        f.write(b"S")
        f.write(struct.pack(">I", 50))
        f.write(_stock("TEST"))
        f.write(struct.pack(">I", 10020))

        # T: trade
        f.write(b"T")
        f.write(_token("TOK125"))
        f.write(struct.pack(">I", 25))
        f.write(struct.pack(">I", 10030))
        f.write(struct.pack(">Q", 2222))

    rows = ingest.parse_ouch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-ouch-4.2")
    assert isinstance(rows, list)
    assert len(rows) >= 7
    types = {r.get("event_type") for r in rows if isinstance(r, dict)}
    assert {"LO+", "LO-", "MO+", "CX+"}.issuperset(types)


def test_parse_nasdaq_ouch_message_details(tmp_path: Path):
    p = tmp_path / "nasdaq_ouch_42_detailed.bin"
    with p.open("wb") as f:
        # O: Enter Order (client-side, to be skipped). Total payload is 49 bytes.
        f.write(b"O")
        f.write(_token("CLIENT01"))  # 14
        f.write(b"B")  # 1
        f.write(struct.pack(">I", 100))  # 4
        f.write(_stock("TEST"))  # 8
        f.write(struct.pack(">I", 9990))  # 4
        f.write(b"\0" * 18)  # Pad to 49 bytes

        # A: Accepted (Buy Side) -> LO+
        f.write(b"A")
        f.write(_token("SRV_BUY"))
        f.write(b"B")
        f.write(struct.pack(">I", 50))
        f.write(_stock("TEST"))
        f.write(struct.pack(">I", 10010))  # price = 1.001

        # A: Accepted (Sell Side) -> LO-
        f.write(b"A")
        f.write(_token("SRV_SELL"))
        f.write(b"S")
        f.write(struct.pack(">I", 40))
        f.write(_stock("TEST"))
        f.write(struct.pack(">I", 10020))  # price = 1.002

        # T: Trade -> MO+
        f.write(b"T")
        f.write(_token("TRADE01"))
        f.write(struct.pack(">I", 25))
        f.write(struct.pack(">I", 10030))  # price = 1.003
        f.write(struct.pack(">Q", 3333))

        # C: Canceled -> CX+
        f.write(b"C")
        f.write(_token("SRV_BUY"))
        f.write(struct.pack(">I", 5))
        f.write(b"U")  # User reason

    rows = ingest.parse_ouch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-ouch-4.2")
    assert isinstance(rows, list)
    # Should parse 4 messages (A, A, T, C) and skip the client 'O' message
    assert len(rows) == 4

    # 1. Accepted Buy Order
    assert rows[0]["event_type"] == "LO+"
    assert rows[0]["side"] == "B"
    assert rows[0]["size"] == 50
    assert pytest.approx(rows[0]["price"]) == 1.001

    # 2. Accepted Sell Order
    assert rows[1]["event_type"] == "LO-"
    assert rows[1]["side"] == "S"
    assert rows[1]["size"] == 40
    assert pytest.approx(rows[1]["price"]) == 1.002

    # 3. Trade
    assert rows[2]["event_type"] == "MO+"
    assert rows[2]["size"] == 25
    assert pytest.approx(rows[2]["price"]) == 1.003

    # 4. Canceled
    assert rows[3]["event_type"] == "CX+"
    assert rows[3]["size"] == 5
    # Price is not applicable for a cancel
    assert rows[3]["price"] != rows[3]["price"]

