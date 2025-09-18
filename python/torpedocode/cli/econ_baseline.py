"""CLI: Volatility-matched economic baseline vs de-risking rule.

Inputs: CSV with columns 'ret' (PnL/returns) and 'pred' (instability probability).
Computes de-risked returns by zeroing exposure when pred >= threshold (chosen by utility
or provided), then builds a volatility-matched baseline by scaling raw returns to match
the standard deviation of de-risked returns. Reports VaR/ES and Kupiec/Christoffersen
for both, with block-bootstrap CIs.
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
    block_bootstrap_var_es,
    choose_threshold_by_utility,
)


def _load_csv(
    path: Path, pred_col: str = "pred", ret_col: str = "ret"
) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    if pred_col not in df.columns or ret_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{ret_col}' and '{pred_col}'")
    r = df[ret_col].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    return r, p


def main():
    ap = argparse.ArgumentParser(description="Volatility-matched economic baseline vs de-risking")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="CSV with 'ret' and 'pred' columns")
    g.add_argument("--pred-csv", type=Path, help="CSV with predictions only")
    ap.add_argument(
        "--ret-csv",
        type=Path,
        default=None,
        help="CSV with returns only (required if --pred-csv used)",
    )
    ap.add_argument("--pred-col", type=str, default="pred")
    ap.add_argument("--ret-col", type=str, default="ret")
    ap.add_argument("--alpha", type=float, default=0.99, help="VaR/ES confidence level")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for de-risking; if omitted, choose by utility",
    )
    ap.add_argument(
        "--threshold-objective",
        type=str,
        default="utility",
        choices=["utility", "var_exceed", "es"],
        help="Objective for threshold selection: maximize utility, or minimize VaR exceedances, or minimize ES",
    )
    ap.add_argument(
        "--grid",
        action="store_true",
        help="Evaluate a percentile grid of thresholds in addition to single best",
    )
    ap.add_argument(
        "--grid-pcts",
        type=int,
        nargs="*",
        default=[50, 60, 70, 80, 90, 95],
        help="Percentiles (0-100) for threshold grid on predictions",
    )
    ap.add_argument(
        "--w-pos", type=float, default=1.0, help="Utility TP weight for threshold selection"
    )
    ap.add_argument(
        "--w-neg", type=float, default=1.0, help="Utility FP penalty for threshold selection"
    )
    ap.add_argument(
        "--block-length", type=float, default=50.0, help="Expected block length for bootstrap"
    )
    ap.add_argument("--n-boot", type=int, default=200, help="Bootstrap samples")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--exceedance-out", type=Path, default=None, help="Optional CSV to write exceedance series"
    )
    args = ap.parse_args()

    if args.input is not None:
        r, p = _load_csv(args.input, pred_col=args.pred_col, ret_col=args.ret_col)
    else:
        if args.ret_csv is None:
            raise SystemExit("--ret-csv is required when using --pred-csv")
        import pandas as pd

        pr = pd.read_csv(args.pred_csv)[args.pred_col].to_numpy(dtype=float)
        rr = pd.read_csv(args.ret_csv)[args.ret_col].to_numpy(dtype=float)
        if pr.shape[0] != rr.shape[0]:
            raise SystemExit("predictions and returns must have the same length")
        p, r = pr, rr
    y = (r < 0).astype(int)
    if args.threshold is not None:
        t = float(args.threshold)
    elif args.threshold_objective == "utility":
        t = choose_threshold_by_utility(p, y, w_pos=float(args.w_pos), w_neg=float(args.w_neg))
    else:
        # Grid-based selection for risk objectives
        grid = sorted({min(max(int(q), 0), 100) for q in [50, 60, 70, 80, 90, 95]})
        cand = [float(np.quantile(p, q / 100.0)) for q in grid]
        def _score(th: float) -> float:
            act = (p >= th).astype(int)
            r_prot = r * (1 - act)
            v, e = var_es(r_prot, alpha=float(args.alpha))
            return float(np.sum((-r_prot) > v)) if args.threshold_objective == "var_exceed" else float(e)
        best = None
        for th in cand:
            s = _score(th)
            if best is None or s < best[0]:
                best = (s, th)
        t = float(best[1]) if best is not None else float(np.median(p))
    action = (p >= t).astype(int)
    r_prot = r * (1 - action)
    std_raw = float(np.std(r)) if r.size else 1.0
    std_prot = float(np.std(r_prot))
    s = (std_prot / std_raw) if std_raw > 0 else 1.0
    r_base = r * s

    def _exceed(loss: np.ndarray, var: float) -> np.ndarray:
        return (loss > var).astype(int)

    var_p, es_p = var_es(r_prot, alpha=float(args.alpha))
    var_b, es_b = var_es(r_base, alpha=float(args.alpha))
    exc_p = _exceed(-r_prot, var_p)
    exc_b = _exceed(-r_base, var_b)
    kup_p = kupiec_pof_test(exc_p, alpha=float(args.alpha))
    kup_b = kupiec_pof_test(exc_b, alpha=float(args.alpha))
    chr_p = christoffersen_independence_test(exc_p)
    chr_b = christoffersen_independence_test(exc_b)

    ci_p = block_bootstrap_var_es(
        r_prot,
        alpha=float(args.alpha),
        expected_block_length=float(args.block_length),
        n_boot=int(args.n_boot),
    )
    ci_b = block_bootstrap_var_es(
        r_base,
        alpha=float(args.alpha),
        expected_block_length=float(args.block_length),
        n_boot=int(args.n_boot),
    )

    out = {
        "threshold": float(t),
        "protected": {
            "var": float(var_p),
            "es": float(es_p),
            "kupiec_p": float(kup_p),
            "christoffersen_p": float(chr_p),
            "ci": ci_p,
        },
        "baseline_vol_matched": {
            "scale": float(s),
            "var": float(var_b),
            "es": float(es_b),
            "kupiec_p": float(kup_b),
            "christoffersen_p": float(chr_b),
            "ci": ci_b,
        },
    }

    if args.grid:
        grid = sorted({min(max(int(q), 0), 100) for q in args.grid_pcts})
        grid_res = []
        for q in grid:
            thr = float(np.quantile(p, q / 100.0))
            act = (p >= thr).astype(int)
            r_prot_q = r * (1 - act)
            std_raw = float(np.std(r)) if r.size else 1.0
            std_prot_q = float(np.std(r_prot_q))
            s_q = (std_prot_q / std_raw) if std_raw > 0 else 1.0
            r_base_q = r * s_q
            var_pq, es_pq = var_es(r_prot_q, alpha=float(args.alpha))
            var_bq, es_bq = var_es(r_base_q, alpha=float(args.alpha))
            grid_res.append(
                {
                    "percentile": q,
                    "threshold": float(thr),
                    "protected": {"var": float(var_pq), "es": float(es_pq)},
                    "baseline": {"var": float(var_bq), "es": float(es_bq)},
                }
            )
        out["grid"] = grid_res

    if getattr(args, "exceedance_out", None):
        exc_df = np.vstack([exc_p, exc_b]).T
        import pandas as pd

        df_exc = pd.DataFrame(exc_df, columns=["exc_protected", "exc_baseline"])
        args.exceedance_out.parent.mkdir(parents=True, exist_ok=True)
        df_exc.to_csv(args.exceedance_out, index=False)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
