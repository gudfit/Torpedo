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
from ..evaluation.economic import realized_volatility
import warnings

try:  # suppress benign sklearn warnings when a split has one class
    from sklearn.exceptions import UndefinedMetricWarning as _UMW  # type: ignore
    warnings.filterwarnings("ignore", category=_UMW)
except Exception:
    pass


def _load_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    p = df["pred"].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    r = df["ret"].to_numpy(dtype=float) if "ret" in df.columns else None
    return p, y, r


def _load_npz(path: Path):
    obj = np.load(path, allow_pickle=False)
    p = obj["pred"]
    y = obj["label"]
    r = obj["ret"] if "ret" in obj else None
    return p, y, r


def main():
    ap = argparse.ArgumentParser(description="Metrics by realized-volatility regimes")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="CSV with pred,label[,ret]")
    g.add_argument("--npz", type=Path, help="NPZ with pred,label[,ret]")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    p, y, r = _load_csv(args.input) if args.input is not None else _load_npz(args.npz)
    out = {"overall": {}}
    m_all = compute_classification_metrics(p, y)
    out["overall"].update({
        "auroc": float(m_all.auroc),
        "auprc": float(m_all.auprc),
        "brier": float(m_all.brier),
        "ece": float(m_all.ece),
    })
    if r is not None:
        rv = realized_volatility(r)
        med = float(np.median(rv))
        mask_calm = rv <= med
        mask_vol = rv > med
        for name, mask in ("calm", mask_calm), ("volatile", mask_vol):
            if np.any(mask):
                m = compute_classification_metrics(p[mask], y[mask])
                out[name] = {
                    "auroc": float(m.auroc),
                    "auprc": float(m.auprc),
                    "brier": float(m.brier),
                    "ece": float(m.ece),
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
