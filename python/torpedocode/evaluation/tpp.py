"""Temporal point process diagnostics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np


@dataclass
class TPPArrays:
    """Container for required arrays.

    - intensities: [T, M] nonnegative intensities per event type
    - event_type_ids: [T] ints in [0, M-1]
    - delta_t: [T] inter-event times (seconds)
    """

    intensities: np.ndarray
    event_type_ids: np.ndarray
    delta_t: np.ndarray


def nll_per_event_from_arrays(
    intensities: np.ndarray, event_type_ids: np.ndarray, delta_t: np.ndarray
) -> float:
    """Compute the exact TPP negative log-likelihood per event.

    NLL = -sum_i log lambda_{m_i}(t_i) + sum_i (sum_m lambda_m(t_i)) * (t_{i+1}-t_i)
    Assuming piecewise-constant intensities between events as in the methodology.
    Returns NLL divided by number of events.
    """
    lam = np.asarray(intensities, dtype=float)
    et = np.asarray(event_type_ids, dtype=int).reshape(-1)
    dt = np.asarray(delta_t, dtype=float).reshape(-1)
    if lam.ndim != 2 or lam.shape[0] != et.shape[0] or et.shape[0] != dt.shape[0]:
        return float("nan")
    T, M = lam.shape
    et = np.clip(et, 0, max(M - 1, 0))
    idx = (np.arange(T), et)
    lam_evt = lam[idx]
    lam_evt = np.clip(lam_evt, 1e-12, None)
    log_lam_sum = float(np.sum(np.log(lam_evt)))
    comp = float(np.sum(np.sum(lam, axis=1) * dt))
    nll = -log_lam_sum + comp
    return nll / max(T, 1)


def rescaled_times(arr: TPPArrays) -> np.ndarray:
    """Compute rescaled inter-arrival times xi_i = ∫ sum_m lambda_m dt."""
    lam_sum = np.sum(arr.intensities, axis=1)
    dt = np.asarray(arr.delta_t, dtype=float).reshape(-1)
    xi = lam_sum.reshape(-1) * dt
    return xi


def rescaled_times_per_type(arr: TPPArrays) -> Dict[int, np.ndarray]:
    """Per-type rescaled inter-arrival times using type-specific intensities.

    For each event of type m occurring at time index i, compute
      xi_i^(m) = sum_{k=prev_i}^{i-1} lambda_m[k] * delta_t[k],
    where prev_i is the previous index with event type m (or 0 if none).
    Returns a dict mapping m -> array of xi for that type in event order.
    """
    lam = np.asarray(arr.intensities, dtype=float)
    et = np.asarray(arr.event_type_ids, dtype=int).reshape(-1)
    dt = np.asarray(arr.delta_t, dtype=float).reshape(-1)
    if lam.ndim != 2 or lam.shape[0] != et.shape[0] or et.shape[0] != dt.shape[0]:
        return {}
    T, M = lam.shape
    out: Dict[int, List[float]] = {m: [] for m in range(M)}
    last_idx = {m: 0 for m in range(M)}
    for i in range(T):
        m = int(et[i])
        m = max(0, min(M - 1, m))
        j0 = last_idx[m]
        if i > j0:
            xi = float(np.sum(lam[j0:i, m] * dt[j0:i]))
        else:
            xi = float(0.0)
        out[m].append(xi)
        last_idx[m] = i
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def model_and_empirical_frequencies(arr: TPPArrays) -> Tuple[np.ndarray, np.ndarray]:
    """Return (empirical_freq[M], model_freq[M]) for event types."""
    T, M = arr.intensities.shape
    dt = np.asarray(arr.delta_t, dtype=float).reshape(-1)
    lam = np.asarray(arr.intensities, dtype=float)
    counts = np.bincount(np.asarray(arr.event_type_ids, dtype=int).reshape(-1), minlength=M).astype(
        float
    )
    empirical = counts / (counts.sum() + 1e-12)
    mass_per_type = (lam * dt[:, None]).sum(axis=0)
    model_freq = mass_per_type / (mass_per_type.sum() + 1e-12)
    return empirical, model_freq


__all__ = [
    "TPPArrays",
    "rescaled_times",
    "rescaled_times_per_type",
    "model_and_empirical_frequencies",
    "nll_per_event_from_arrays",
]
