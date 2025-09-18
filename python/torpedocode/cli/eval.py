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
from ..evaluation.io import load_preds_labels_csv, load_preds_labels_npz
from ..evaluation.helpers import temperature_scale_from_probs


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
        p, y, _r, p2 = load_preds_labels_csv(
            args.input, pred_col=args.pred_col, label_col=args.label_col, pred2_col=args.pred2_col
        )
    else:
        p, y, _r, p2 = load_preds_labels_npz(
            args.npz, pred_key=args.pred_key, label_key=args.label_key, pred2_key=args.pred2_key
        )

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
        T, p_cal, meta = temperature_scale_from_probs(p, y)
        out.update(
            {
                "temperature": T,
                "calibrated": {
                    "auroc": meta["auroc"],
                    "auroc_ci": [meta["auroc_ci_low"], meta["auroc_ci_high"]],
                    "auprc": meta["auprc"],
                    "brier": meta["brier"],
                    "ece": meta["ece"],
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
