import json
import numpy as np
from pathlib import Path

from torpedocode.cli import econ_baseline as eco


def test_threshold_objective_var_exceed(tmp_path: Path, monkeypatch):
    # Simple synthetic: high preds align with negative returns; minimizing exceedances should pick high threshold
    r = np.array([0.01, 0.02, -0.5, 0.01, -0.4, 0.02], dtype=float)
    p = np.array([0.1, 0.2, 0.95, 0.2, 0.99, 0.1], dtype=float)
    path = tmp_path / "data.csv"
    import pandas as pd

    pd.DataFrame({"ret": r, "pred": p}).to_csv(path, index=False)
    out = tmp_path / "out.json"
    argv = [
        "prog",
        "--input",
        str(path),
        "--alpha",
        "0.95",
        "--threshold-objective",
        "var_exceed",
        "--output",
        str(out),
    ]
    import sys

    monkeypatch.setattr(sys, "argv", argv)
    eco.main()
    obj = json.loads(out.read_text())
    assert "threshold" in obj

