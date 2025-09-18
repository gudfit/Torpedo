import sys
from pathlib import Path
import json
import numpy as np


def test_cli_aggregate_block_bootstrap(tmp_path, monkeypatch):
    from torpedocode.cli import aggregate as agg

    # Create nested prediction files under label paths
    def mkcsv(path: Path):
        p = np.linspace(0.0, 1.0, 60)
        y = (p > 0.5).astype(int)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(["idx,pred,label"] + [f"{i},{p[i]},{y[i]}" for i in range(len(p))]))

    root = tmp_path / "artifacts"
    mkcsv(root / "X" / "instability_s_1" / "predictions_test.csv")
    mkcsv(root / "Y" / "instability_s_1" / "predictions_test.csv")

    argv = [
        "prog",
        "--pred-root", str(root),
        "--pred-pattern", "*/instability_s_1/predictions_test.csv",
        "--block-bootstrap", "--n-boot", "50",
        "--output", str(tmp_path / "agg.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    agg.main()
    obj = json.loads((tmp_path / "agg.json").read_text())
    assert "micro_ci_block" in obj
    assert "auroc_ci" in obj["micro_ci_block"]

