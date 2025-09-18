"""CLI: compute AUROC/AUPRC/Brier/ECE and DeLong CI/tests from stored predictions.

Run as:
  python -m torpedocode.cli.eval --input predictions.csv --pred-col pred --label-col label [--pred2-col pred_b]
  python -m torpedocode.cli.eval --npz preds_labels.npz --pred-key pred --label-key label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ..evaluation.metrics import (
    compute_classification_metrics,
    delong_ci_auroc,
    delong_test_auroc,
)
from ..evaluation.calibration import TemperatureScaler


def _load_from_csv(path: Path, pred_col: str, label_col: str, pred2_col: str | None):
    import pandas as pd

    df = pd.read_csv(path)
    p = df[pred_col].to_numpy(dtype=float)
    y = df[label_col].astype(int).to_numpy()
    p2 = df[pred2_col].to_numpy(dtype=float) if pred2_col and pred2_col in df.columns else None
    return p, y, p2


def _load_from_npz(path: Path, pred_key: str, label_key: str, pred2_key: str | None):
    obj = np.load(path, allow_pickle=False)
    p = obj[pred_key]
    y = obj[label_key]
    p2 = obj[pred2_key] if pred2_key and pred2_key in obj else None
    return p, y, p2


def main():
    ap = argparse.ArgumentParser(description="Compute classification metrics and DeLong stats.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="CSV file with predictions and labels")
    g.add_argument("--npz", type=Path, help="NPZ file with arrays")
    ap.add_argument("--pred-col", default="pred", help="Prediction column (CSV)")
    ap.add_argument("--label-col", default="label", help="Label column (CSV)")
    ap.add_argument("--pred2-col", default=None, help="Second model prediction column (CSV)")
    ap.add_argument("--pred-key", default="pred", help="Prediction key (NPZ)")
    ap.add_argument("--label-key", default="label", help="Label key (NPZ)")
    ap.add_argument("--pred2-key", default=None, help="Second model prediction key (NPZ)")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI significance level")
    ap.add_argument(
        "--temperature-scale",
        action="store_true",
        help="Apply temperature scaling to logits derived from probabilities before metrics",
    )
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON output path")

    args = ap.parse_args()

    if args.input is not None:
        p, y, p2 = _load_from_csv(args.input, args.pred_col, args.label_col, args.pred2_col)
    else:
        p, y, p2 = _load_from_npz(args.npz, args.pred_key, args.label_key, args.pred2_key)

    m = compute_classification_metrics(p, y)
    auc, lo, hi = delong_ci_auroc(p, y, alpha=args.alpha)
    out = {
        "auroc": auc,
        "auroc_ci": [lo, hi],
        "auprc": m.auprc,
        "brier": m.brier,
        "ece": m.ece,
    }

    if args.temperature_scale:
        eps = 1e-6
        logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
        scaler = TemperatureScaler()
        T = scaler.fit(logits, y)
        z_cal = scaler.transform(logits)
        p_cal = 1.0 / (1.0 + np.exp(-z_cal))
        m_cal = compute_classification_metrics(p_cal, y)
        auc_cal, lo_cal, hi_cal = delong_ci_auroc(p_cal, y, alpha=args.alpha)
        out.update(
            {
                "temperature": float(T),
                "calibrated": {
                    "auroc": auc_cal,
                    "auroc_ci": [lo_cal, hi_cal],
                    "auprc": m_cal.auprc,
                    "brier": m_cal.brier,
                    "ece": m_cal.ece,
                },
            }
        )

    if p2 is not None:
        delta, z, pval = delong_test_auroc(p, p2, y)
        out.update({"delta_auroc": delta, "delong_z": z, "delong_p": pval})

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
