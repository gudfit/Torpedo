import json
import runpy
from pathlib import Path


def test_cli_metrics_regime_csv(tmp_path: Path, monkeypatch):
    # Build a tiny CSV with pred,label,ret and check JSON keys
    csv = tmp_path / "preds.csv"
    csv.write_text("\n".join([
        "pred,label,ret",
        "0.1,0,0.01",
        "0.9,1,-0.05",
        "0.8,1,-0.02",
        "0.2,0,0.03",
    ]))
    out = tmp_path / "out.json"

    import sys
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--input", str(csv),
        "--output", str(out),
    ])
    runpy.run_module("torpedocode.cli.metrics_regime", run_name="__main__")
    obj = json.loads(out.read_text())
    assert "overall" in obj
    # realized-volatility split present
    assert any(k in obj for k in ("calm", "volatile"))

