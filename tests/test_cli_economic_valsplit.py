import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_cli_economic_valsplit(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import economic as econ

    # Build test CSV (pred,label[,ret]) and separate validation CSV
    test = tmp_path / "test.csv"
    val = tmp_path / "val.csv"

    df_te = pd.DataFrame({"pred": [0.2, 0.8], "label": [0, 1], "ret": [0.01, -0.02]})
    df_va = pd.DataFrame({"pred": [0.1, 0.9], "label": [0, 1]})
    df_te.to_csv(test, index=False)
    df_va.to_csv(val, index=False)

    argv = [
        "prog",
        "--input",
        str(test),
        "--val-input",
        str(val),
        "--choose-threshold",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    econ.main()
    out = capsys.readouterr().out
    import json

    res = json.loads(out)
    # With val preds [0.1,0.9], labels [0,1], best utility threshold = 0.9
    assert "threshold" in res and abs(res["threshold"] - 0.9) < 1e-9
