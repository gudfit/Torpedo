import json
import sys
from io import StringIO

import numpy as np


def test_cli_eval_npz_basic(monkeypatch, capsys, tmp_path):
    from torpedocode.cli import eval as eval_cli

    # Prepare simple predictions with clear signal
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(int)
    np.savez(tmp_path / "preds.npz", pred=p, label=y)

    argv = [
        "prog",
        "--npz",
        str(tmp_path / "preds.npz"),
        "--pred-key",
        "pred",
        "--label-key",
        "label",
        "--alpha",
        "0.05",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    eval_cli.main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "auroc" in data and "auprc" in data and "brier" in data and "ece" in data
    assert "auroc_ci" in data and len(data["auroc_ci"]) == 2

