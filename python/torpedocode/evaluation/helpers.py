"""Evaluation helpers for shared routines (TPP diagnostics, TDA backends, temperature scaling)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import json
import numpy as np

from .metrics import (
    compute_point_process_diagnostics,
    delong_ci_auroc,
    ks_statistic as _ks_statistic,
    kolmogorov_pvalue as _ks_pvalue,
)
from .tpp import (
    TPPArrays,
    rescaled_times,
    rescaled_times_per_type,
    model_and_empirical_frequencies,
    nll_per_event_from_arrays,
)


def write_tda_backends_json(path: Path) -> None:
    """Write a small report with availability/version of ripser/gudhi/persim to path."""

    def _check_mod(name: str):
        try:
            mod = __import__(name)
            ver = None
            for key in ("__version__", "version", "__VERSION__"):
                if hasattr(mod, key):
                    ver = getattr(mod, key)
                    break
            return {
                "available": True,
                "version": (ver if isinstance(ver, (str, float, int)) else str(ver)),
            }
        except Exception:
            return {"available": False, "version": None}

    report = {
        "torpedocode_tda": _check_mod("torpedocode_tda"),
        "ripser": _check_mod("ripser"),
        "gudhi": _check_mod("gudhi"),
        "persim": _check_mod("persim"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def save_tpp_arrays_and_diagnostics(
    out_dir: Path, intensities: np.ndarray, event_type_ids: np.ndarray, delta_t: np.ndarray
) -> Dict[str, float]:
    """Save TPP arrays to NPZ and a JSON diagnostics report in out_dir. Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "tpp_test_arrays.npz",
        intensities=intensities.astype(np.float32),
        event_type_ids=event_type_ids.astype(np.int64),
        delta_t=delta_t.astype(np.float32),
    )
    arr = TPPArrays(intensities=intensities, event_type_ids=event_type_ids, delta_t=delta_t)
    xi = rescaled_times(arr)
    emp, mod = model_and_empirical_frequencies(arr)
    nll_evt = nll_per_event_from_arrays(intensities, event_type_ids, delta_t)
    diag_pp = compute_point_process_diagnostics(
        xi, empirical_frequencies=emp, model_frequencies=mod
    )
    # Per-type KS p-values via time-rescaling (Uniform[0,1] check)
    per_type_ks: dict[str, float] = {}
    try:
        xi_types = rescaled_times_per_type(arr)
        for m, x in xi_types.items():
            if x.size == 0:
                per_type_ks[str(int(m))] = float("nan")
                continue
            u = 1.0 - np.exp(-np.asarray(x, dtype=float))
            ks = _ks_statistic(u)
            pv = _ks_pvalue(ks, int(u.size))
            per_type_ks[str(int(m))] = float(pv)
    except Exception:
        per_type_ks = {}
    diag = {
        "nll_per_event": float(nll_evt),
        "ks_p_value": float(diag_pp.ks_p_value),
        "coverage_error": float(diag_pp.coverage_error),
        "ks_p_value_per_type": per_type_ks,
    }
    with open(out_dir / "tpp_test_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    return diag


def temperature_scale_from_probs(
    preds: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, Dict[str, float]]:
    """Fit a temperature on logits derived from probabilities and return (T, calibrated_probs, metrics)."""
    from .calibration import TemperatureScaler
    from .metrics import compute_classification_metrics

    eps = 1e-6
    logits = np.log(np.clip(preds, eps, 1 - eps) / np.clip(1 - preds, eps, 1 - eps))
    scaler = TemperatureScaler()
    T = scaler.fit(logits, labels)
    z_cal = scaler.transform(logits)
    p_cal = 1.0 / (1.0 + np.exp(-z_cal))
    m = compute_classification_metrics(p_cal, labels)
    auc, lo, hi = delong_ci_auroc(p_cal, labels)
    meta = {
        "temperature": float(T),
        "auroc": float(auc),
        "auroc_ci_low": float(lo),
        "auroc_ci_high": float(hi),
        "auprc": float(m.auprc),
        "brier": float(m.brier),
        "ece": float(m.ece),
    }
    return float(T), p_cal, meta


__all__ = [
    "write_tda_backends_json",
    "save_tpp_arrays_and_diagnostics",
    "temperature_scale_from_probs",
]
