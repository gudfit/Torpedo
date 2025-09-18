"""Economic evaluation utilities: VaR/ES, Kupiec/Christoffersen, thresholding."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def var_es(returns: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Compute one-sided loss VaR/ES at level alpha. returns are PnL; losses = -returns."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    loss = -r
    var = np.quantile(loss, alpha)
    es = loss[loss >= var].mean() if np.any(loss >= var) else var
    return float(var), float(es)


def kupiec_pof_test(exceedances: np.ndarray, alpha: float) -> float:
    """Kupiec unconditional coverage test p-value (likelihood ratio) for exceedance rate.
    exceedances: 1 if loss > VaR, 0 otherwise.
    """
    x = np.asarray(exceedances, dtype=int).reshape(-1)
    n = len(x)
    if n == 0:
        return float("nan")
    pi_hat = x.mean()
    pi0 = 1.0 - alpha
    k = x.sum()
    eps = 1e-12
    lr = 2.0 * (
        k * (np.log(pi_hat + eps) - np.log(pi0 + eps))
        + (n - k) * (np.log(1 - pi_hat + eps) - np.log(1 - pi0 + eps))
    )
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(max(lr, 0.0), df=1))
    except Exception:
        from math import erf

        z = float(np.sqrt(max(lr, 0.0)))
        p = 1.0 - 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        return float(2.0 * p)


def christoffersen_independence_test(exceedances: np.ndarray) -> float:
    """Christoffersen independence test p-value based on 2x2 transition counts.
    Returns p-value of LR test for independence of exceedances.
    """
    x = np.asarray(exceedances, dtype=int).reshape(-1)
    if len(x) < 2:
        return float("nan")
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(x)):
        a, b = x[i - 1], x[i]
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1 + 1e-12)
    eps = 1e-12
    num = (1 - pi) ** (n00 + n10) * (pi ** (n01 + n11) + eps)
    den = ((1 - pi01) ** n00) * (pi01**n01 + eps) * ((1 - pi11) ** n10) * (pi11**n11 + eps)
    lr = -2.0 * np.log((num + eps) / (den + eps))
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(max(lr, 0.0), df=1))
    except Exception:
        from math import erf

        z = float(np.sqrt(max(lr, 0.0)))
        p = 1.0 - 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        return float(2.0 * p)


def choose_threshold_by_utility(
    p: np.ndarray, y: np.ndarray, *, w_pos: float = 1.0, w_neg: float = 1.0
) -> float:
    """Pick decision threshold maximizing weighted utility on validation data.
    Utility = w_pos * TP - w_neg * FP.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    grid = np.unique(p)
    best_t, best_u = 0.5, -1e18
    for t in grid:
        yhat = (p >= t).astype(int)
        tp = float(np.sum((yhat == 1) & (y == 1)))
        fp = float(np.sum((yhat == 1) & (y == 0)))
        u = w_pos * tp - w_neg * fp
        if u > best_u:
            best_u, best_t = u, float(t)
    return float(best_t)


def realized_volatility(returns: np.ndarray, window: int = 50) -> np.ndarray:
    r = np.asarray(returns, dtype=float).reshape(-1)
    out = np.zeros_like(r)
    for i in range(len(r)):
        j0 = max(0, i - window + 1)
        out[i] = np.sqrt(np.sum(r[j0 : i + 1] ** 2))
    return out


from .metrics import stationary_block_indices as _stationary_block_indices


def block_bootstrap_var_es(
    returns: np.ndarray,
    *,
    alpha: float = 0.99,
    expected_block_length: float = 50.0,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Stationary block bootstrap CIs for VaR/ES given a returns series."""
    rng = rng or np.random.default_rng(0)
    r = np.asarray(returns, dtype=float).reshape(-1)
    n = len(r)
    if n == 0:
        return {
            "var_ci": [float("nan"), float("nan")],
            "es_ci": [float("nan"), float("nan")],
            "var": float("nan"),
            "es": float("nan"),
        }
    var0, es0 = var_es(r, alpha=alpha)
    var_s = []
    es_s = []
    for _ in range(int(n_boot)):
        idx = _stationary_block_indices(n, float(expected_block_length), rng)
        vb, eb = var_es(r[idx], alpha=alpha)
        var_s.append(vb)
        es_s.append(eb)
    lo_q, hi_q = 0.025, 0.975
    return {
        "var": float(var0),
        "es": float(es0),
        "var_ci": [float(np.quantile(var_s, lo_q)), float(np.quantile(var_s, hi_q))],
        "es_ci": [float(np.quantile(es_s, lo_q)), float(np.quantile(es_s, hi_q))],
    }


__all__ = [
    "var_es",
    "kupiec_pof_test",
    "christoffersen_independence_test",
    "choose_threshold_by_utility",
    "realized_volatility",
    "block_bootstrap_var_es",
]
