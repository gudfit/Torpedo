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
)
from ..evaluation.metrics import compute_classification_metrics


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
        "--threshold",
        type=float,
        default=None,
        help="Explicit decision threshold; overrides selection",
    )
    args = ap.parse_args()

    p, y, r = _load_csv(args.input) if args.input is not None else _load_npz(args.npz)
    pv = yv = None
    if args.val_input is not None:
        pv, yv, _ = _load_csv(args.val_input)
    elif args.val_npz is not None:
        obj = np.load(args.val_npz, allow_pickle=False)
        pv = obj["pred"] if "pred" in obj else None
        yv = obj["label"] if "label" in obj else None
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

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
