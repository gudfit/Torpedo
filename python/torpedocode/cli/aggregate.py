"""CLI: Aggregate results across instruments/markets and horizons.

Supports two modes:
1) From eval JSONs (produced by eval CLI): computes macro (mean) and weighted means.
   True micro requires raw predictions, so not provided in this mode.
2) From prediction CSVs (columns: pred,label): computes true micro (pooled) and macro,
   and provides simple bootstrap CIs.

Examples:
  # Aggregate eval JSONs under artifacts/*/eval_test.json
  python -m torpedocode.cli.aggregate --root artifacts --pattern "*/eval_test.json" --weights weights.csv

  # Aggregate prediction CSVs
  python -m torpedocode.cli.aggregate --pred-root artifacts --pred-pattern "*/predictions_test.csv"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..evaluation.metrics import (
    compute_classification_metrics,
    delong_ci_auroc,
    bootstrap_confint_metric,
    benjamini_hochberg,
)


def _load_weights(path: Path | None) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    w: Dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                w[parts[0]] = float(parts[1])
            except Exception:
                continue
    return w


def _aggregate_eval_json(files: List[Path], weights: Dict[str, float]) -> Dict:
    metrics = {"auroc": [], "auprc": [], "brier": [], "ece": []}
    wts = []
    names = []
    for p in files:
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        for k in metrics:
            if k in obj:
                metrics[k].append(float(obj[k]))
        name = p.parent.name
        names.append(name)
        wts.append(float(weights.get(name, 1.0)))

    out: Dict[str, Dict] = {}
    w = np.asarray(wts, dtype=float)
    w = w / (w.sum() + 1e-12)
    for k, vals in metrics.items():
        if not vals:
            continue
        v = np.asarray(vals, dtype=float)
        macro = float(np.nanmean(v))
        weighted = float(np.nansum(v * w)) if len(w) == len(v) else macro
        _, lo, hi = bootstrap_confint_metric(v, n_boot=500)
        out[k] = {"macro": macro, "weighted": weighted, "macro_ci": [lo, hi]}
    return out


def _read_pred_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    p = df["pred"].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    return p, y


def _aggregate_pred_csv(files: List[Path]) -> Dict:
    per_metrics = []
    all_p = []
    all_y = []
    for p in files:
        try:
            preds, labels = _read_pred_csv(p)
        except Exception:
            continue
        m = compute_classification_metrics(preds, labels)
        per_metrics.append(m)
        all_p.append(preds)
        all_y.append(labels)

    if not per_metrics or not all_p:
        return {}
    macro = {
        "auroc": float(np.nanmean([m.auroc for m in per_metrics])),
        "auprc": float(np.nanmean([m.auprc for m in per_metrics])),
        "brier": float(np.nanmean([m.brier for m in per_metrics])),
        "ece": float(np.nanmean([m.ece for m in per_metrics])),
    }
    macro_ci = {}
    for k in ["auroc", "auprc", "brier", "ece"]:
        arr = np.array([getattr(m, k) for m in per_metrics], dtype=float)
        _, lo, hi = bootstrap_confint_metric(arr, n_boot=500)
        macro_ci[k] = [lo, hi]

    P = np.concatenate(all_p)
    Y = np.concatenate(all_y)
    cm = compute_classification_metrics(P, Y)
    auc, lo, hi = delong_ci_auroc(P, Y)
    micro = {"auroc": auc, "auprc": cm.auprc, "brier": cm.brier, "ece": cm.ece}
    micro_ci = {"auroc": [lo, hi]}

    return {"macro": macro, "macro_ci": macro_ci, "micro": micro, "micro_ci": micro_ci}


def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics across instruments/markets")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=Path, help="Root directory for eval JSON glob (eval mode)")
    g.add_argument("--pred-root", type=Path, help="Root directory for prediction CSVs (pred mode)")
    ap.add_argument(
        "--pattern", type=str, default="*/eval_test.json", help="Glob under root for eval JSONs"
    )
    ap.add_argument(
        "--pred-pattern",
        type=str,
        default="*/predictions_test.csv",
        help="Glob under pred-root for CSVs",
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="CSV mapping name,weight for weighted averages (eval mode)",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--apply-fdr", action="store_true", help="Apply BH FDR control to p-values across JSONs"
    )
    ap.add_argument(
        "--pkey", type=str, default="delong_p", help="Key in eval JSON providing p-values for FDR"
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR level (BH)")
    ap.add_argument(
        "--block-bootstrap",
        action="store_true",
        help="Use stationary block bootstrap for pooled micro CIs (pred mode)",
    )
    ap.add_argument(
        "--block-length",
        type=float,
        default=50.0,
        help="Expected block length for stationary bootstrap",
    )
    ap.add_argument(
        "--auto-block",
        action="store_true",
        help="Estimate expected block length via Politis–White rule",
    )
    ap.add_argument("--n-boot", type=int, default=200, help="Number of bootstrap samples")

    args = ap.parse_args()

    if args.root is not None:
        files = [Path(p) for p in glob.glob(str(args.root / args.pattern))]
        res = _aggregate_eval_json(files, _load_weights(args.weights))
        if args.apply_fdr:
            pv = {}
            for p in files:
                try:
                    obj = json.loads(Path(p).read_text())
                except Exception:
                    continue
                if args.pkey in obj:
                    pv[Path(p).parent.name] = float(obj[args.pkey])
            if pv:
                import numpy as np

                names = list(pv.keys())
                arr = np.array([pv[n] for n in names], dtype=float)
                bh = benjamini_hochberg(arr, alpha=float(args.alpha))
                res["fdr"] = {
                    "pkey": args.pkey,
                    "alpha": float(args.alpha),
                    "threshold": float(bh["threshold"]),
                    "rejected": {n: bool(bh["rejected"][i]) for i, n in enumerate(names)},
                    "qvalues": {n: float(bh["qvalues"][i]) for i, n in enumerate(names)},
                }
    else:
        files = [Path(p) for p in glob.glob(str(args.pred_root / args.pred_pattern))]
        res = _aggregate_pred_csv(files)
        if args.block_bootstrap and res and "micro" in res:
            import pandas as pd

            p_all = []
            y_all = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    p_all.append(df["pred"].to_numpy(dtype=float))
                    y_all.append(df["label"].astype(int).to_numpy())
                except Exception:
                    continue
                if p_all:
                    import numpy as np

                    P = np.concatenate(p_all)
                Y = np.concatenate(y_all)
                from ..evaluation.metrics import (
                    block_bootstrap_micro_ci,
                    politis_white_expected_block_length,
                )

                L = (
                    None
                    if args.auto_block
                    else float(
                        args.block_length if args.block_length and args.block_length > 0 else 50.0
                    )
                )
                ci = block_bootstrap_micro_ci(P, Y, expected_block_length=L, n_boot=args.n_boot)
                res["micro_ci_block"] = ci

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
