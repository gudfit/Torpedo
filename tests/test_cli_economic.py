import json
from pathlib import Path


def test_cli_economic_from_csv(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import economic as eco
    # create simple CSV
    p = [0.1, 0.9, 0.2, 0.8]
    y = [0, 1, 0, 1]
    r = [0.01, -0.05, 0.02, -0.03]
    csv = tmp_path / "preds.csv"
    csv.write_text("\n".join(["pred,label,ret"] + [f"{p[i]},{y[i]},{r[i]}" for i in range(len(p))]))

    argv = ["prog", "--input", str(csv), "--alpha", "0.9"]
    import sys
    monkeypatch.setattr(sys, "argv", argv)
    eco.main()
    out = capsys.readouterr().out
    res = json.loads(out)
    assert "VaR" in res and "ES" in res

