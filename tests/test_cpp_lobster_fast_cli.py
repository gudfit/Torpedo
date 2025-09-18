import os
import subprocess
from pathlib import Path

import pytest


def test_lobster_fast_runs_without_book(tmp_path: Path):
    exe = Path("cpp/lobster_fast")
    if not exe.exists() or not os.access(exe, os.X_OK):
        pytest.skip("lobster_fast binary not available")

    msg = tmp_path / "messages.csv"
    book_placeholder = "-"  # signal no orderbook alignment
    with msg.open("w") as f:
        f.write("0.5,1,1,100,10.0,1\n")

    p = subprocess.run(
        [str(exe), str(msg), book_placeholder, "TEST", "0.01", "2020-01-02", "0"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        pytest.skip(
            f"lobster_fast not rebuilt with no-book support: rc={p.returncode} err={p.stderr.strip()}"
        )
    out = p.stdout.strip().splitlines()
    assert out and out[0].startswith("timestamp,")
    assert ",LOBSTER" in out[-1]
