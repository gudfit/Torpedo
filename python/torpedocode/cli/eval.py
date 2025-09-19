"""CLI: compute AUROC/AUPRC/Brier/ECE and DeLong CI/tests from stored predictions.

Run as:
  python -m torpedocode.cli.eval --input predictions.csv --pred-col pred --label-col label [--pred2-col pred_b]
  python -m torpedocode.cli.eval --npz preds_labels.npz --pred-key pred --label-key label

Note:
  Temporal point process (TPP) diagnostics, including per‑type KS p‑values, are written by the
  training pipeline when invoked with --tpp-diagnostics. See:
    <artifact_dir>/tpp_test_diagnostics.json
  Look for key "ks_p_value_per_type" (mapping event_type_id -> p-value).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ..evaluation.metrics import (
    compute_classification_metrics,
    compute_calibration_report,
    delong_ci_auroc,
    delong_test_auroc,
    block_bootstrap_micro_ci,
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
    ap.add_argument("--ece-bins", type=int, default=15, help="Number of bins for ECE calculation")
    ap.add_argument(
        "--block-bootstrap",
        action="store_true",
        help="Report stationary block bootstrap CIs for AUROC/AUPRC/Brier/ECE",
    )
    ap.add_argument("--block-length", type=float, default=50.0, help="Expected block length")
    ap.add_argument("--n-boot", type=int, default=200, help="Bootstrap replicates")
    ap.add_argument(
        "--auto-block",
        action="store_true",
        help="Estimate expected block length via Politis–White rule",
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

    # Base metrics
    m = compute_classification_metrics(p, y)
    auc, lo, hi = delong_ci_auroc(p, y, alpha=args.alpha)
    # Override ECE with requested bin count (if different)
    if int(args.ece_bins) != 15:
        from math import isfinite as _isfinite

        num_bins = int(max(1, args.ece_bins))
        cal = compute_calibration_report(p, y, num_bins=num_bins)
        n = len(p)
        base = n // num_bins
        rem = n % num_bins
        bin_sizes = np.full(num_bins, base, dtype=float)
        if rem > 0:
            bin_sizes[:rem] += 1.0
        weights = bin_sizes / float(max(n, 1))
        ece_bins = float(np.sum(np.abs(cal.bin_accuracy - cal.bin_confidence) * weights))
        ece_val = ece_bins
    else:
        ece_val = m.ece

    out = {
        "auroc": auc,
        "auroc_ci": [lo, hi],
        "auprc": m.auprc,
        "brier": m.brier,
        "ece": ece_val,
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

    if args.block_bootstrap:
        L = None if bool(args.auto_block) else float(args.block_length)
        ci = block_bootstrap_micro_ci(
            p,
            y,
            expected_block_length=L,
            n_boot=int(args.n_boot),
            alpha=float(args.alpha),
        )
        out["block_bootstrap_ci"] = ci

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
