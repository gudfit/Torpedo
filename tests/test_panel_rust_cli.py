import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("torpedocode-panel") is None
    and not (Path("bin") / ("torpedocode-panel" + (".exe" if os.name == "nt" else ""))).exists(),
    reason="torpedocode-panel binary not available",
)
def test_panel_rust_end_to_end(tmp_path):
    # Write minimal CSV
    csv_path = tmp_path / "panel.csv"
    csv_path.write_text(
        "market,symbol,median_daily_notional,tick_size\n"
        "XNAS,AAPL,1000000,0.01\n"
        "XNAS,MSFT,800000,0.01\n"
        "XNYS,IBM,500000,0.01\n"
    )
    out = tmp_path / "out.json"
    bin_path = shutil.which("torpedocode-panel") or str(
        Path("bin") / ("torpedocode-panel" + (".exe" if os.name == "nt" else ""))
    )
    cmd = [
        bin_path,
        "--input",
        str(csv_path),
        "--by",
        "liq_decile",
        "tick_size",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    obj = json.loads(out.read_text())
    assert isinstance(obj, list) and len(obj) == 3
    # Each row should include liq_decile and match_group
    for r in obj:
        assert "liq_decile" in r and "match_group" in r
