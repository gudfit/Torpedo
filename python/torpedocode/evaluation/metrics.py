"""Metric computation scaffolding for TorpedoCode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import math

try:
    from sklearn.metrics import roc_auc_score as _roc_auc_score
    from sklearn.metrics import average_precision_score as _average_precision_score
except Exception:  # pragma: no cover - optional
    _roc_auc_score = None
    _average_precision_score = None

try:
    from scipy.stats import kstest as _kstest
except Exception:  # pragma: no cover - optional
    _kstest = None


@dataclass
class ClassificationMetrics:
    """Binary classification diagnostics."""

    auroc: float
    auprc: float
    brier: float
    ece: float


@dataclass
class CalibrationReport:
    """Container for reliability diagrams and summary scores."""

    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray


@dataclass
class PointProcessDiagnostics:
    """Diagnostic statistics for temporal point process models."""

    # Historically this field held mean(xi), which is a proxy to exact NLL/event when
    # intensities are not available. We keep it for backward compatibility.
    nll_per_event: float
    ks_p_value: float
    coverage_error: float
    # Explicit alias to avoid confusion: equals mean(xi) in this function.
    nll_proxy_per_event: float | None = None


def compute_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> ClassificationMetrics:
    """Compute AUROC, AUPRC, Brier score, and ECE (15 bins)."""

    p = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.size == 0:
        return ClassificationMetrics(auroc=np.nan, auprc=np.nan, brier=np.nan, ece=np.nan)

    # AUROC
    if _roc_auc_score is not None:
        try:
            auroc = float(_roc_auc_score(y, p))
        except Exception:
            auroc = float("nan")
    else:
        auroc = _auroc_numpy(p, y)

    # AUPRC
    if _average_precision_score is not None:
        try:
            auprc = float(_average_precision_score(y, p))
        except Exception:
            auprc = float("nan")
    else:
        auprc = _auprc_numpy(p, y)

    # Brier
    brier = float(np.mean((p - y) ** 2))

    # ECE (equal-frequency bins but weight by actual bin sizes to match definition)
    num_bins = 15
    calib = compute_calibration_report(p, y, num_bins=num_bins)
    n = len(p)
    # Reconstruct bin sizes used in compute_calibration_report (equal frequency with remainder)
    base = n // num_bins
    rem = n % num_bins
    bin_sizes = np.full(num_bins, base, dtype=float)
    if rem > 0:
        bin_sizes[:rem] += 1.0
    weights = bin_sizes / float(max(n, 1))
    ece = float(np.sum(np.abs(calib.bin_accuracy - calib.bin_confidence) * weights))

    return ClassificationMetrics(auroc=auroc, auprc=auprc, brier=brier, ece=ece)


def delong_ci_auroc(
    predictions: np.ndarray, labels: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Exact DeLong AUROC variance and CI for a single classifier."""
    p = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if p.size == 0 or y.size != p.size:
        return float("nan"), float("nan"), float("nan")
    X = p[y == 1]
    Y = p[y == 0]
    m, n = X.size, Y.size
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    # Midrank helper
    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        i = 0
        N = len(x)
        while i < N:
            j = i
            while j < N and x[order[j]] == x[order[i]]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return ranks

    z = np.concatenate([X, Y])
    r = _midrank(z)
    r_x = r[:m]
    r_y = r[m:]
    tx = _midrank(X)
    ty = _midrank(Y)

    auc = float((np.sum(r_x) - m * (m + 1) / 2.0) / (m * n))
    v10 = (r_x - tx) / n
    v01 = 1.0 - (r_y - ty) / m
    s10 = float(np.var(v10, ddof=1)) if m > 1 else float("nan")
    s01 = float(np.var(v01, ddof=1)) if n > 1 else float("nan")
    if not np.isfinite(s10) or not np.isfinite(s01):
        return auc, float("nan"), float("nan")
    var_auc = s10 / m + s01 / n
    se = math.sqrt(max(var_auc, 1e-24))
    zval = 1.959963984540054  # approx qnorm(0.975)
    lo = max(0.0, auc - zval * se)
    hi = min(1.0, auc + zval * se)
    return auc, lo, hi


def delong_test_auroc(
    predictions1: np.ndarray, predictions2: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """Two-sample correlated DeLong test for AUROC difference on the same dataset.

    Returns (delta_auc, z_stat, p_value). Uses normal approximation.
    """
    p1 = np.asarray(predictions1, dtype=float).reshape(-1)
    p2 = np.asarray(predictions2, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if p1.size == 0 or p2.size != p1.size or y.size != p1.size:
        return float("nan"), float("nan"), float("nan")
    X1, Y1 = p1[y == 1], p1[y == 0]
    X2, Y2 = p2[y == 1], p2[y == 0]
    m, n = X1.size, Y1.size
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        i = 0
        N = len(x)
        while i < N:
            j = i
            while j < N and x[order[j]] == x[order[i]]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return ranks

    # AUCs via joint midranks for each classifier
    def _auc_v(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        z = np.concatenate([x_pos, x_neg])
        r = _midrank(z)
        r_x = r[: len(x_pos)]
        r_y = r[len(x_pos) :]
        tx = _midrank(x_pos)
        ty = _midrank(x_neg)
        auc = (np.sum(r_x) - len(x_pos) * (len(x_pos) + 1) / 2.0) / (len(x_pos) * len(x_neg))
        v10 = (r_x - tx) / len(x_neg)
        v01 = 1.0 - (r_y - ty) / len(x_pos)
        return float(auc), v10, v01

    auc1, v10_1, v01_1 = _auc_v(X1, Y1)
    auc2, v10_2, v01_2 = _auc_v(X2, Y2)
    s10_11 = np.cov(v10_1, v10_2, ddof=1)[0, 1] if m > 1 else float("nan")
    s01_11 = np.cov(v01_1, v01_2, ddof=1)[0, 1] if n > 1 else float("nan")
    var1 = (
        np.var(v10_1, ddof=1) / m + np.var(v01_1, ddof=1) / n if m > 1 and n > 1 else float("nan")
    )
    var2 = (
        np.var(v10_2, ddof=1) / m + np.var(v01_2, ddof=1) / n if m > 1 and n > 1 else float("nan")
    )
    cov12 = (
        (s10_11 / m) + (s01_11 / n) if np.isfinite(s10_11) and np.isfinite(s01_11) else float("nan")
    )
    if not (np.isfinite(var1) and np.isfinite(var2) and np.isfinite(cov12)):
        return float(auc1 - auc2), float("nan"), float("nan")
    var_delta = float(var1 + var2 - 2.0 * cov12)
    if not np.isfinite(var_delta) or var_delta <= 0:
        return float(auc1 - auc2), float("nan"), float("nan")
    z = float((auc1 - auc2) / math.sqrt(var_delta))
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(auc1 - auc2), z, p


def bootstrap_ci_auroc(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for AUROC by resampling pairs (p, y) with replacement.

    Returns (auc, lo, hi). Falls back gracefully when degenerate (single-class) samples occur.
    """
    p = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if p.size == 0 or y.size != p.size:
        return float("nan"), float("nan"), float("nan")
    # Point AUROC
    try:
        if _roc_auc_score is not None:
            auc = float(_roc_auc_score(y, p))
        else:
            auc = _auroc_numpy(p, y)
    except Exception:
        auc = float("nan")
    rng = rng or np.random.default_rng(0)
    n = len(p)
    samples = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        pb = p[idx]
        yb = y[idx]
        if np.all(yb == 0) or np.all(yb == 1):
            continue  # skip degenerate resample
        try:
            if _roc_auc_score is not None:
                ab = float(_roc_auc_score(yb, pb))
            else:
                ab = _auroc_numpy(pb, yb)
            samples.append(ab)
        except Exception:
            continue
    if not samples:
        return auc, float("nan"), float("nan")
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    return auc, float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q))


def bootstrap_confint_metric(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar metric given per-sample contributions."""
    rng = rng or np.random.default_rng(0)
    vals = np.asarray(values, dtype=float)
    if vals.size == 0 or not np.isfinite(vals).any():
        return float("nan"), float("nan"), float("nan")
    boot = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(float(np.mean(vals[idx])))
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return float(np.mean(vals)), lo, hi


def micro_macro_average(
    metrics_per_group: dict[str, float], weights: dict[str, float] | None = None
) -> tuple[float, float]:
    """Return (micro, macro) averages given grouped metrics and optional weights for micro."""
    if not metrics_per_group:
        return float("nan"), float("nan")
    macro = float(np.mean(list(metrics_per_group.values())))
    if weights is None:
        micro = macro
    else:
        keys = list(metrics_per_group.keys())
        w = np.array([weights.get(k, 0.0) for k in keys], dtype=float)
        v = np.array([metrics_per_group[k] for k in keys], dtype=float)
        w = w / (w.sum() + 1e-12)
        micro = float((w * v).sum())
    return micro, macro


def _stationary_block_indices(n: int, L: float, rng: np.random.Generator) -> np.ndarray:
    """Generate indices for one stationary block bootstrap sample of length n."""
    p = 1.0 / max(L, 1e-6)
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        k = int(rng.geometric(p))
        for j in range(k):
            idx.append((start + j) % n)
            if len(idx) >= n:
                break
    return np.asarray(idx[:n], dtype=int)


def block_bootstrap_micro_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    expected_block_length: float | None = 50.0,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Stationary block bootstrap CIs for micro metrics on time-ordered pairs."""
    rng = rng or np.random.default_rng(0)
    p = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    n = len(p)
    if n == 0:
        return {
            "auroc_ci": [float("nan"), float("nan")],
            "auprc_ci": [float("nan"), float("nan")],
            "brier_ci": [float("nan"), float("nan")],
            "ece_ci": [float("nan"), float("nan")],
        }
    if expected_block_length is None or expected_block_length <= 0:
        expected_block_length = politis_white_expected_block_length(p, y)

    auc_samples = []
    prc_samples = []
    brier_samples = []
    ece_samples = []
    for _ in range(int(n_boot)):
        idx = _stationary_block_indices(n, float(expected_block_length), rng)
        pb = p[idx]
        yb = y[idx]
        m = compute_classification_metrics(pb, yb)
        auc = delong_ci_auroc(pb, yb)[0]
        auc_samples.append(auc)
        prc_samples.append(m.auprc)
        brier_samples.append(m.brier)
        ece_samples.append(m.ece)

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    return {
        "auroc_ci": [float(np.quantile(auc_samples, lo_q)), float(np.quantile(auc_samples, hi_q))],
        "auprc_ci": [float(np.quantile(prc_samples, lo_q)), float(np.quantile(prc_samples, hi_q))],
        "brier_ci": [
            float(np.quantile(brier_samples, lo_q)),
            float(np.quantile(brier_samples, hi_q)),
        ],
        "ece_ci": [float(np.quantile(ece_samples, lo_q)), float(np.quantile(ece_samples, hi_q))],
    }


def politis_white_expected_block_length(
    predictions: np.ndarray, labels: np.ndarray, *, max_lag: int = 50
) -> float:
    """Estimate expected block length for stationary bootstrap via a Politis–White-style rule.

    Uses residual series s_t = p_t - y_t, truncated autocovariances up to `max_lag` to
    form μ0 = γ(0) + 2∑_{k=1}^K γ(k) and μ2 = 2∑_{k=1}^K k γ(k), then
    L* ≈ (2 μ2 / μ0^2)^{1/3} n^{1/3}. Falls back to n^{1/3} when degenerate.
    """
    p = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    n = len(p)
    if n <= 1 or n != len(y):
        return max(1.0, float(n) ** (1.0 / 3.0))
    s = p - y
    s = s - float(np.mean(s))
    K = int(min(max_lag, n - 1))
    if K <= 0:
        return max(1.0, float(n) ** (1.0 / 3.0))

    def acov(k: int) -> float:
        a = s[: n - k]
        b = s[k:]
        return float(np.dot(a, b) / (n - k))

    g0 = acov(0)
    if not np.isfinite(g0) or abs(g0) < 1e-18:
        return max(1.0, float(n) ** (1.0 / 3.0))
    g = np.array([acov(k) for k in range(1, K + 1)], dtype=float)
    mask = np.abs(g) > (0.01 * abs(g0))
    if np.any(mask):
        last = int(np.where(mask)[0][-1]) + 1
        g = g[:last]
    mu0 = g0 + 2.0 * float(np.sum(g))
    mu2 = 2.0 * float(np.sum((np.arange(1, len(g) + 1)) * g))
    if not np.isfinite(mu0) or abs(mu0) < 1e-18:
        return max(1.0, float(n) ** (1.0 / 3.0))
    ratio = max(1e-24, 2.0 * mu2 / (mu0 * mu0))
    L = float((ratio) ** (1.0 / 3.0) * (n ** (1.0 / 3.0)))
    return max(1.0, L)


def compute_calibration_report(
    predictions: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> CalibrationReport:
    """Compute equal-frequency reliability bins."""

    p = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = len(p)
    if n == 0:
        return CalibrationReport(
            bin_confidence=np.zeros((num_bins,)), bin_accuracy=np.zeros((num_bins,))
        )

    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bin_sizes = np.full(num_bins, n // num_bins)
    bin_sizes[: n % num_bins] += 1
    starts = np.cumsum(np.concatenate([[0], bin_sizes[:-1]]))
    ends = np.cumsum(bin_sizes)
    conf = np.zeros((num_bins,), dtype=float)
    acc = np.zeros((num_bins,), dtype=float)
    for i, (s, e) in enumerate(zip(starts, ends)):
        if s == e:
            conf[i] = 0.0
            acc[i] = 0.0
            continue
        sl = slice(s, e)
        conf[i] = float(np.mean(p_sorted[sl]))
        acc[i] = float(np.mean(y_sorted[sl]))
    return CalibrationReport(bin_confidence=conf, bin_accuracy=acc)


def compute_point_process_diagnostics(
    rescaled_times: np.ndarray, empirical_frequencies: np.ndarray, model_frequencies: np.ndarray
) -> PointProcessDiagnostics:
    """Diagnostics via time-rescaling KS, coverage error, and NLL/event."""

    xi = np.asarray(rescaled_times, dtype=float)
    if xi.size == 0:
        return PointProcessDiagnostics(
            nll_per_event=np.nan, ks_p_value=np.nan, coverage_error=np.nan
        )

    nll_proxy = float(np.mean(xi))
    u = 1.0 - np.exp(-xi)
    if _kstest is not None:
        try:
            stat = _kstest(u, "uniform")
            ks_p = float(stat.pvalue)
        except Exception:
            ks_p = float("nan")
    else:
        ks = _ks_statistic(u)
        ks_p = float(_kolmogorov_pvalue(ks, len(u)))

    ef = np.asarray(empirical_frequencies, dtype=float)
    mf = np.asarray(model_frequencies, dtype=float)
    ef = ef / (ef.sum() + 1e-12)
    mf = mf / (mf.sum() + 1e-12)
    coverage_error = float(np.sum(np.abs(ef - mf)))

    return PointProcessDiagnostics(
        nll_per_event=nll_proxy,
        nll_proxy_per_event=nll_proxy,
        ks_p_value=ks_p,
        coverage_error=coverage_error,
    )


def _auroc_numpy(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(p))
    pos = y == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _auprc_numpy(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y == 1).sum()), 1)
    return float(np.trapz(precision, recall))


def _ks_statistic(u: np.ndarray) -> float:
    n = len(u)
    if n == 0:
        return float("nan")
    u_sorted = np.sort(u)
    cdf = np.arange(1, n + 1) / n
    d_plus = np.max(cdf - u_sorted)
    d_minus = np.max(u_sorted - (np.arange(n) / n))
    return float(max(d_plus, d_minus))


def _kolmogorov_pvalue(ks: float, n: int) -> float:
    if n <= 0 or not np.isfinite(ks):
        return float("nan")
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks
    terms = [2 * (-1) ** (k - 1) * np.exp(-2 * (k * k) * (lam * lam)) for k in range(1, 101)]
    return float(max(0.0, min(1.0, 1.0 - 2.0 * sum(terms))))


__all__ = [
    "ClassificationMetrics",
    "CalibrationReport",
    "PointProcessDiagnostics",
    "compute_classification_metrics",
    "compute_calibration_report",
    "compute_point_process_diagnostics",
    "delong_ci_auroc",
    "delong_test_auroc",
    "bootstrap_ci_auroc",
    "bootstrap_confint_metric",
    "micro_macro_average",
    "block_bootstrap_micro_ci",
    "politis_white_expected_block_length",
]


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> dict:
    """Benjamini–Hochberg FDR control.

    Returns a dict with fields:
      - rejected: boolean mask of rejections at level alpha
      - qvalues: BH-adjusted p-values
      - threshold: critical value t such that p_(k) <= (k/m)*alpha
    """
    p = np.asarray(pvals, dtype=float).reshape(-1)
    m = len(p)
    if m == 0:
        return {
            "rejected": np.array([], dtype=bool),
            "qvalues": np.array([], dtype=float),
            "threshold": float("nan"),
        }
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    crit = ranks * (alpha / m)
    le = p_sorted <= crit
    k = int(np.max(np.where(le)[0]) + 1) if np.any(le) else 0
    thr = float(crit[k - 1]) if k > 0 else float("nan")
    # q-values (adjusted p): min over j>=i of (m/j) p_(j)
    q = np.empty_like(p_sorted)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = (m / float(i + 1)) * p_sorted[i]
        prev = min(prev, val)
        q[i] = prev
    q = np.clip(q, 0.0, 1.0)
    q_unsorted = np.empty_like(q)
    q_unsorted[order] = q
    rejected = np.zeros(m, dtype=bool)
    if k > 0:
        rejected[order[:k]] = True
    return {"rejected": rejected, "qvalues": q_unsorted, "threshold": thr}
