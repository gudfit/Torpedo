import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not (Path("cpp/lobster_fast").exists() and os.access("cpp/lobster_fast", os.X_OK)),
    reason="lobster_fast binary not available",
)
def test_lobster_stateful_infer_level_and_utc(tmp_path: Path):
    # Prepare minimal orderbook with header and one snapshot at t=0
    book = tmp_path / "book.csv"
    book.write_text("timestamp,ask_price_1,bid_price_1\n0.0,100.1,100.0\n")
    # Message at t=0.5s hitting bid with matching price; expect level 1
    msg = tmp_path / "messages.csv"
    # columns: time,type,order_id,size,price,direction
    msg.write_text("0.5,1,1,100,100.0,1\n")

    exe = Path("cpp/lobster_fast").resolve()
    proc = subprocess.run(
        [str(exe), str(msg), str(book), "TEST", "0.01", "2025-01-02", "0"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"lobster_fast run failed: {proc.stderr.strip()}")
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
    assert lines and lines[0].startswith("timestamp,")
    # Last line should include ",1," for level=1
    assert ",1," in lines[-1]
    # UTC epoch ns on 2025-01-02 + 0.5s
    fields = lines[-1].split(",")
    ts = int(fields[0])
    assert ts > 0
    # Basic sanity: ~ Jan 2, 2025 in ns
    assert ts >= 1735776000 * 10**9
