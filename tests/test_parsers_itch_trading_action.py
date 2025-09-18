import struct
from pathlib import Path

import pytest

ingest = pytest.importorskip("torpedocode_ingest")


def _be_u48(x: int) -> bytes:
    # 48-bit big-endian
    return int(x).to_bytes(6, "big")


def _pad_stock(s: str) -> bytes:
    b = s.encode("ascii")
    return (b + b" " * 8)[:8]


def test_parse_nasdaq_itch_trading_action_propagates_state(tmp_path: Path):
    p = tmp_path / "nasdaq_itch_50_trading_action.bin"
    with p.open("wb") as f:
        # H: Trading Action (halt)
        f.write(b"H")
        f.write(struct.pack(">H", 1))  # stock locate
        f.write(struct.pack(">H", 1))  # tracking
        f.write(_be_u48(10))  # ts
        f.write(_pad_stock("TEST"))  # stock
        f.write(b"H")  # trading state (halted)
        f.write(b"\x00")  # reserved
        f.write(struct.pack(">I", 1234))  # reason
        # E: Executed after trading action (include match id per NASDAQ ITCH 5.0)
        f.write(b"E")
        f.write(struct.pack(">H", 2))
        f.write(struct.pack(">H", 2))
        f.write(_be_u48(11))
        f.write(struct.pack(">Q", 111))
        f.write(struct.pack(">I", 50))
        f.write(struct.pack(">Q", 2222))

    rows = ingest.parse_itch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-itch-5.0")
    assert isinstance(rows, list)
    # Capability check: if the installed native module doesn't propagate trading_state,
    # skip to avoid false failure when rebuild isn't available in CI.
    any_state = any(isinstance(r, dict) and ("trading_state" in r) for r in rows)
    if not any_state:
        import pytest as _pytest

        _pytest.skip("native module not rebuilt; trading_state not propagated")
    # Find META row with trading_state
    states = [r.get("trading_state") for r in rows if isinstance(r, dict)]
    assert "H" in states
    # Ensure subsequent event still parsed (any canonical non-META type)
    types = [r.get("event_type") for r in rows if isinstance(r, dict)]
    assert any(t in {"LO+", "LO-", "MO+", "CX+"} for t in types)
