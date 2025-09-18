import json
import numpy as np
from pathlib import Path


def test_cli_tpp_eval_confusion(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import tpp_eval as tppc

    T, M = 40, 3
    intensities = np.zeros((T, M), dtype=np.float32)
    # Create confident predictions matching true labels half the time
    true = np.arange(T) % M
    for i in range(T):
        intensities[i, :] = 0.1
        intensities[i, true[i]] = 5.0
    delta_t = np.ones((T,), dtype=np.float32) * 0.2
    npz = tmp_path / "arrs.npz"
    np.savez(npz, intensities=intensities, event_type_ids=true.astype(np.int64), delta_t=delta_t)

    argv = ["prog", "--npz", str(npz)]
    import sys

    monkeypatch.setattr(sys, "argv", argv)
    tppc.main()
    out = capsys.readouterr().out
    res = json.loads(out)
    cm = res.get("confusion_matrix")
    assert isinstance(cm, list) and len(cm) == M and all(len(r) == M for r in cm)
    total = sum(sum(row) for row in cm)
    assert total == T
    # Diagonal should be dominant
    diag = sum(cm[i][i] for i in range(M))
    assert diag >= T // 2
    # Row/col normalized matrices present and have proper shape
    cm_row = res.get("confusion_matrix_row_normalized")
    cm_col = res.get("confusion_matrix_col_normalized")
    assert isinstance(cm_row, list) and len(cm_row) == M and all(len(r) == M for r in cm_row)
    assert isinstance(cm_col, list) and len(cm_col) == M and all(len(r) == M for r in cm_col)
    # Per-type metrics present with precision/recall/f1
    pt = res.get("per_type_metrics")
    assert isinstance(pt, list) and len(pt) == M
    assert set(["precision", "recall", "f1"]).issubset(pt[0].keys())
