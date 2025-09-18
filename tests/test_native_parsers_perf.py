import os
import time
import struct
from pathlib import Path

import pytest

from torpedocode.data.itch import ITCHParseConfig, parse_itch_minimal
from torpedocode.data.ouch import OUCHParseConfig, parse_ouch_minimal


ingest = pytest.importorskip("torpedocode_ingest")


@pytest.mark.skipif(
    os.environ.get("TORPEDOCODE_RUN_PERF", "0") != "1",
    reason="perf test gated by TORPEDOCODE_RUN_PERF=1",
)
def test_native_itch_faster_than_python(tmp_path: Path):
    # Generate a moderately large minimal ITCH file
    N = 20000
    p = tmp_path / "bench.itch"
    with p.open("wb") as f:
        ts = 1
        for i in range(N):
            # Alternate A (add) and E (execute) patterns to vary payload sizes
            if i % 4 in (0, 1):
                # 'A' 8+1+4+8+8
                f.write(struct.pack("<Q", ts))
                f.write(b"A")
                f.write(struct.pack("<Q", i))
                f.write(b"B" if (i % 2 == 0) else b"S")
                f.write(struct.pack("<I", 10))
                f.write(struct.pack("<Q", 10000 + (i % 50)))
                f.write(struct.pack("<Q", 0))
            else:
                # 'E' 8+4
                f.write(struct.pack("<Q", ts))
                f.write(b"E")
                f.write(struct.pack("<Q", i))
                f.write(struct.pack("<I", 5))
            ts += 1

    # Time Python fallback
    t0 = time.perf_counter()
    df_py = parse_itch_minimal(p, cfg=ITCHParseConfig(tick_size=0.0001, symbol="TEST"))
    t1 = time.perf_counter()

    # Time Rust native
    t2 = time.perf_counter()
    rows = ingest.parse_itch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-itch-5.0")
    t3 = time.perf_counter()

    # Sanity checks
    assert len(df_py) > 0 and isinstance(rows, list) and len(rows) > 0

    py_time = t1 - t0
    rs_time = t3 - t2
    assert rs_time <= py_time, f"native={rs_time:.4f}s slower than python={py_time:.4f}s"


@pytest.mark.skipif(
    os.environ.get("TORPEDOCODE_RUN_PERF", "0") != "1",
    reason="perf test gated by TORPEDOCODE_RUN_PERF=1",
)
def test_native_ouch_faster_than_python(tmp_path: Path):
    # Generate a moderately large minimal OUCH file
    N = 20000
    p = tmp_path / "bench.ouch"
    with p.open("wb") as f:
        ts = 1
        for i in range(N):
            if i % 3 == 0:
                # O: ts, 'O', client_id(8), side, shares, price
                f.write(struct.pack("<Q", ts))
                f.write(b"O")
                f.write(struct.pack("<Q", i))
                f.write(b"B" if (i % 2 == 0) else b"S")
                f.write(struct.pack("<I", 10 + (i % 7)))
                f.write(struct.pack("<Q", 10000 + (i % 50)))
            elif i % 3 == 1:
                # E: ts, 'E', client_id, executed
                f.write(struct.pack("<Q", ts))
                f.write(b"E")
                f.write(struct.pack("<Q", i))
                f.write(struct.pack("<I", 5))
            else:
                # X: ts, 'X', client_id, canceled
                f.write(struct.pack("<Q", ts))
                f.write(b"X")
                f.write(struct.pack("<Q", i))
                f.write(struct.pack("<I", 3))
            ts += 1

    t0 = time.perf_counter()
    df_py = parse_ouch_minimal(p, cfg=OUCHParseConfig(tick_size=0.0001, symbol="TEST"))
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    rows = ingest.parse_ouch_file(str(p), tick_size=0.0001, symbol="TEST", spec="nasdaq-ouch-4.2")
    t3 = time.perf_counter()

    assert len(df_py) > 0 and isinstance(rows, list) and len(rows) > 0
    py_time = t1 - t0
    rs_time = t3 - t2
    assert rs_time <= py_time, f"native={rs_time:.4f}s slower than python={py_time:.4f}s"
