"""CLI: TPP diagnostics from saved arrays (NPZ/CSV).

Accepts NPZ with keys: intensities[T,M], event_type_ids[T], delta_t[T].
Outputs KS p-value (time-rescaling), NLL/event proxy, and coverage error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ..evaluation.tpp import (
    TPPArrays,
    rescaled_times,
    rescaled_times_per_type,
    model_and_empirical_frequencies,
    nll_per_event_from_arrays,
)
from ..evaluation.metrics import compute_point_process_diagnostics, ks_statistic, kolmogorov_pvalue


def _load_npz(path: Path) -> TPPArrays:
    obj = np.load(path, allow_pickle=False)
    lam = obj["intensities"]
    et = obj["event_type_ids"]
    dt = obj["delta_t"]
    return TPPArrays(intensities=lam, event_type_ids=et, delta_t=dt)


def main():
    ap = argparse.ArgumentParser(description="Temporal point process diagnostics from arrays")
    ap.add_argument(
        "--npz", type=Path, required=True, help="NPZ with intensities,event_type_ids,delta_t"
    )
    ap.add_argument(
        "--per-type-hist-bins",
        type=int,
        default=0,
        help="If >0, include per-type histograms of model mass vs empirical events",
    )
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    arr = _load_npz(args.npz)
    xi = rescaled_times(arr)
    emp, mod = model_and_empirical_frequencies(arr)
    diag = compute_point_process_diagnostics(xi, empirical_frequencies=emp, model_frequencies=mod)
    nll_evt = nll_per_event_from_arrays(arr.intensities, arr.event_type_ids, arr.delta_t)

    per_type = []
    M = arr.intensities.shape[1]
    xi_by_type = rescaled_times_per_type(arr)
    for m in range(M):
        Xm = xi_by_type.get(m, np.array([], dtype=float))
        U_m = 1.0 - np.exp(-Xm)
        try:
            from scipy.stats import kstest as _kstest  # type: ignore

            ks_p = float(_kstest(U_m, "uniform").pvalue) if U_m.size > 0 else float("nan")
        except Exception:
            ks = ks_statistic(U_m)
            ks_p = kolmogorov_pvalue(ks, len(U_m))
        per_type.append({"event_type": int(m), "ks_p_value": ks_p})
    preds = np.argmax(arr.intensities, axis=1).astype(int)
    M = int(arr.intensities.shape[1])
    cm = np.zeros((M, M), dtype=int)
    for t in range(len(preds)):
        i = int(arr.event_type_ids[t])
        j = int(preds[t])
        if 0 <= i < M and 0 <= j < M:
            cm[i, j] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        cm_col = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums != 0)

    per_type_metrics = []
    for k in range(M):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - tp)
        fn = float(cm[k, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_type_metrics.append(
            {
                "event_type": int(k),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )

    out = {
        "nll_per_event": nll_evt,
        "nll_proxy_per_event": diag.nll_per_event,
        "ks_p_value": diag.ks_p_value,
        "coverage_error": diag.coverage_error,
        "empirical_freq": emp.tolist(),
        "model_coverage": mod.tolist(),
        "confusion_matrix": cm.tolist(),
        "per_type": per_type,
        "confusion_matrix_row_normalized": cm_row.tolist(),
        "confusion_matrix_col_normalized": cm_col.tolist(),
        "per_type_metrics": per_type_metrics,
    }

    if int(args.per_type_hist_bins) and int(args.per_type_hist_bins) > 0:
        bins = int(args.per_type_hist_bins)
        lam = arr.intensities
        lam_sum = lam.sum(axis=1, keepdims=True)
        lam_sum[lam_sum <= 0] = 1.0
        mass = lam / lam_sum
        edges = np.linspace(0.0, 1.0, bins + 1)
        hist = []
        for m in range(M):
            w = mass[:, m]
            h_m, _ = np.histogram(w, bins=edges)
            e = (arr.event_type_ids == m).astype(float)
            he_m, _ = np.histogram(w, bins=edges, weights=e)
            hist.append(
                {
                    "event_type": int(m),
                    "bin_edges": edges.tolist(),
                    "model_mass_hist": h_m.astype(float).tolist(),
                    "empirical_event_hist": he_m.astype(float).tolist(),
                }
            )
        out["per_type_histograms"] = hist
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
