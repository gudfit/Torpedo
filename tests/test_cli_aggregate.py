import sys
from pathlib import Path
import json
import numpy as np


def test_cli_aggregate_from_eval_json(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import aggregate as agg

    # Create two eval JSONs
    a = {"auroc": 0.8, "auprc": 0.6, "brier": 0.2, "ece": 0.1}
    b = {"auroc": 0.9, "auprc": 0.7, "brier": 0.15, "ece": 0.05}
    (tmp_path / "X").mkdir(); (tmp_path / "Y").mkdir()
    (tmp_path / "X" / "eval_test.json").write_text(json.dumps(a))
    (tmp_path / "Y" / "eval_test.json").write_text(json.dumps(b))

    argv = ["prog", "--root", str(tmp_path), "--pattern", "*/eval_test.json"]
    monkeypatch.setattr(sys, "argv", argv)
    agg.main()
    out = capsys.readouterr().out
    res = json.loads(out)
    assert "auroc" in res and "macro" in res["auroc"]


def test_cli_aggregate_from_pred_csv(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import aggregate as agg

    def mkcsv(path: Path):
        p = np.linspace(0, 1, 50)
        y = (p > 0.5).astype(int)
        path.write_text("\n".join(["idx,pred,label"] + [f"{i},{p[i]},{y[i]}" for i in range(len(p))]))

    (tmp_path / "A").mkdir(); (tmp_path / "B").mkdir()
    mkcsv(tmp_path / "A" / "predictions_test.csv")
    mkcsv(tmp_path / "B" / "predictions_test.csv")

    argv = ["prog", "--pred-root", str(tmp_path), "--pred-pattern", "*/predictions_test.csv"]
    monkeypatch.setattr(sys, "argv", argv)
    agg.main()
    out = capsys.readouterr().out
    res = json.loads(out)
    assert "macro" in res and "micro" in res

