"""CLI: Economic metrics and backtests from predictions and returns.

Inputs: CSV with columns pred,label,ret; or NPZ with arrays pred,label,ret.
Outputs: VaR/ES (alpha), Kupiec and Christoffersen p-values, regime-split metrics by realized volatility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ..evaluation.economic import (
    var_es,
    kupiec_pof_test,
    christoffersen_independence_test,
    choose_threshold_by_utility,
    realized_volatility,
    block_bootstrap_var_es,
)
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.io import load_preds_labels_csv, load_preds_labels_npz


def _threshold_sensitivity(p: np.ndarray, y: np.ndarray, *, grid: np.ndarray | None = None) -> dict:
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 19)
    out = []
    for t in grid:
        yhat = (p >= t).astype(int)
        tp = float(np.sum((yhat == 1) & (y == 1)))
        fp = float(np.sum((yhat == 1) & (y == 0)))
        fn = float(np.sum((yhat == 0) & (y == 1)))
        tn = float(np.sum((yhat == 0) & (y == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1})
    return {"grid": out}


def main():
    ap = argparse.ArgumentParser(description="Compute VaR/ES backtests and regime-split metrics")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="CSV with pred,label[,ret]")
    g.add_argument("--npz", type=Path, help="NPZ with pred,label[,ret]")
    gv = ap.add_mutually_exclusive_group(required=False)
    gv.add_argument(
        "--val-input", type=Path, help="CSV with validation pred,label for threshold selection"
    )
    gv.add_argument(
        "--val-npz", type=Path, help="NPZ with validation pred,label for threshold selection"
    )
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument(
        "--choose-threshold",
        action="store_true",
        help="Choose threshold by utility on data (requires label)",
    )
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-neg", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="Report VaR/ES confidence intervals via stationary block bootstrap",
    )
    ap.add_argument("--boot-n", type=int, default=200, help="Bootstrap replicates for VaR/ES CIs")
    ap.add_argument(
        "--block-l", type=float, default=50.0, help="Expected block length for stationary bootstrap"
    )
    ap.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Emit threshold sensitivity grid over validation or data",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit decision threshold; overrides selection",
    )
    args = ap.parse_args()

    if args.input is not None:
        p, y, r, _p2 = load_preds_labels_csv(
            args.input, pred_col="pred", label_col="label", ret_col="ret"
        )
    else:
        p, y, r, _p2 = load_preds_labels_npz(
            args.npz, pred_key="pred", label_key="label", ret_key="ret"
        )
    pv = yv = None
    if args.val_input is not None:
        pv, yv, _rv, _p2 = load_preds_labels_csv(
            args.val_input, pred_col="pred", label_col="label", ret_col="ret"
        )
    elif args.val_npz is not None:
        pv, yv, _rv, _p2 = load_preds_labels_npz(
            args.val_npz, pred_key="pred", label_key="label", ret_key="ret"
        )
    out = {}
    if y is not None:
        m = compute_classification_metrics(p, y)
        out.update({"auroc": m.auroc, "auprc": m.auprc, "brier": m.brier, "ece": m.ece})
    if r is not None:
        var, es = var_es(r, alpha=args.alpha)
        exceed = (-r) > var
        kupiec = kupiec_pof_test(exceed, alpha=args.alpha)
        christ = christoffersen_independence_test(exceed)
        out.update({"VaR": var, "ES": es, "kupiec_p": kupiec, "christoffersen_p": christ})
        if bool(args.bootstrap_ci):
            ci = block_bootstrap_var_es(
                r,
                alpha=args.alpha,
                expected_block_length=float(args.block_l),
                n_boot=int(args.boot_n),
            )
            out.update({"VaR_CI": ci.get("var_ci"), "ES_CI": ci.get("es_ci")})
        rv = realized_volatility(r)
        med = float(np.median(rv))
        mask_low = rv <= med
        mask_high = rv > med
        for name, mask in ("calm", mask_low), ("volatile", mask_high):
            if np.any(mask):
                v, e = var_es(r[mask], alpha=args.alpha)
                out[f"{name}_VaR"] = v
                out[f"{name}_ES"] = e
    if args.choose_threshold:
        if args.threshold is not None:
            theta = float(args.threshold)
        elif pv is not None and yv is not None:
            theta = choose_threshold_by_utility(pv, yv, w_pos=args.w_pos, w_neg=args.w_neg)
        elif y is not None:
            theta = choose_threshold_by_utility(p, y, w_pos=args.w_pos, w_neg=args.w_neg)
        else:
            theta = float(np.median(p))
        out["threshold"] = float(theta)
        if args.threshold_sweep and y is not None:
            out["threshold_sensitivity"] = _threshold_sensitivity(p, y)
    elif args.threshold_sweep and y is not None:
        out["threshold_sensitivity"] = _threshold_sensitivity(p, y)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
