import sys
import types
import numpy as np


def _kupiec_fallback_pvalue(exceedances, alpha):
    x = np.asarray(exceedances, dtype=int).reshape(-1)
    n = len(x)
    pi_hat = x.mean()
    pi0 = 1.0 - alpha
    k = x.sum()
    eps = 1e-12
    lr = 2.0 * (
        k * (np.log(pi_hat + eps) - np.log(pi0 + eps))
        + (n - k) * (np.log(1 - pi_hat + eps) - np.log(1 - pi0 + eps))
    )
    z = float(np.sqrt(max(lr, 0.0)))
    # 2*(1-Phi(z)) in erf form
    from math import erf

    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / np.sqrt(2.0)))))


def _christoffersen_fallback_pvalue(exceedances):
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
    z = float(np.sqrt(max(lr, 0.0)))
    from math import erf

    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / np.sqrt(2.0)))))


def test_economic_pvalues_scipy_fallback(monkeypatch):
    # Block SciPy so economic module uses fallback
    dummy = types.ModuleType("scipy")
    monkeypatch.setitem(sys.modules, "scipy", dummy)
    import importlib

    eco = importlib.import_module("torpedocode.evaluation.economic")

    rng = np.random.default_rng(0)
    n = 500
    alpha = 0.99
    # Simulate exceedance process with higher-than-expected rate to trigger nontrivial LR
    p_true = 0.03  # expected 1 - alpha = 0.01
    x = (rng.uniform(size=n) < p_true).astype(int)

    pk = eco.kupiec_pof_test(x, alpha)
    pk_fb = _kupiec_fallback_pvalue(x, alpha)
    assert np.isfinite(pk)
    assert 0.0 <= pk <= 1.0
    assert abs(pk - pk_fb) < 1e-12

    pc = eco.christoffersen_independence_test(x)
    pc_fb = _christoffersen_fallback_pvalue(x)
    assert np.isfinite(pc)
    assert 0.0 <= pc <= 1.0
    assert abs(pc - pc_fb) < 1e-12
