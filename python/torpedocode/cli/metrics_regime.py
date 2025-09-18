"""CLI: Classification metrics split by realized volatility regimes.

Reads prediction CSV/NPZ with columns/arrays pred,label,ret. Computes AUROC/AUPRC,
Brier, and ECE overall and for calm vs. volatile regimes defined by realized
volatility median split. Optionally writes JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.io import load_preds_labels_csv, load_preds_labels_npz
from ..evaluation.economic import realized_volatility
import warnings

try:
    from sklearn.exceptions import UndefinedMetricWarning as _UMW  # type: ignore

    warnings.filterwarnings("ignore", category=_UMW)
except Exception:
    pass


def _load(path_csv: Path | None, path_npz: Path | None):
    if path_csv is not None:
        p, y, r, _p2 = load_preds_labels_csv(
            path_csv, pred_col="pred", label_col="label", ret_col="ret"
        )
        return p, y, r
    assert path_npz is not None
    p, y, r, _p2 = load_preds_labels_npz(
        path_npz, pred_key="pred", label_key="label", ret_key="ret"
    )
    return p, y, r


def main():
    ap = argparse.ArgumentParser(description="Metrics by realized-volatility regimes")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="CSV with pred,label[,ret]")
    g.add_argument("--npz", type=Path, help="NPZ with pred,label[,ret]")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    p, y, r = _load(args.input, args.npz)
    out = {"overall": {}}
    m_all = compute_classification_metrics(p, y)
    out["overall"].update(
        {
            "auroc": float(m_all.auroc),
            "auprc": float(m_all.auprc),
            "brier": float(m_all.brier),
            "ece": float(m_all.ece),
        }
    )
    if r is not None:
        rv = realized_volatility(r)
        med = float(np.median(rv))
        mask_calm = rv <= med
        mask_vol = rv > med
        for name, mask in ("calm", mask_calm), ("volatile", mask_vol):
            if np.any(mask):
                yy = y[mask]
                mm = compute_classification_metrics(p[mask], yy)
                if np.all(yy == 0) or np.all(yy == 1):
                    auroc = 0.5
                else:
                    auroc = float(mm.auroc)
                out[name] = {
                    "auroc": auroc,
                    "auprc": float(mm.auprc),
                    "brier": float(mm.brier),
                    "ece": float(mm.ece),
                    "count": int(mask.sum()),
                }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
