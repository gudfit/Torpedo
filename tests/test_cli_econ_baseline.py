import json
from pathlib import Path
import numpy as np


def test_cli_econ_baseline_combined(tmp_path, monkeypatch):
    from torpedocode.cli import econ_baseline as eco

    # Build simple combined CSV
    n = 100
    rng = np.random.default_rng(0)
    preds = rng.uniform(size=n)
    rets = rng.normal(0, 0.01, size=n)
    # Inject a few large negatives to make VaR meaningful
    rets[:5] -= 0.1
    csv = tmp_path / "rp.csv"
    with open(csv, "w") as f:
        f.write("ret,pred\n")
        for r, p in zip(rets, preds):
            f.write(f"{r},{p}\n")
    out = tmp_path / "econ.json"
    exc = tmp_path / "exc.csv"
    argv = [
        "prog",
        "--input",
        str(csv),
        "--alpha",
        "0.95",
        "--grid",
        "--grid-pcts",
        "50",
        "80",
        "--exceedance-out",
        str(exc),
        "--output",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", argv)
    eco.main()
    obj = json.loads(out.read_text())
    assert "protected" in obj and "baseline_vol_matched" in obj
    assert exc.exists()
