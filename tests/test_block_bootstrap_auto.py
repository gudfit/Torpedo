import numpy as np
from torpedocode.evaluation.metrics import politis_white_expected_block_length


def _simulate_series(n=500, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    p = 1.0 / (1.0 + np.exp(-x))
    y = (rng.uniform(size=n) < p).astype(float)
    return p, y


def test_politis_white_block_length_monotonic():
    p_lo, y_lo = _simulate_series(rho=0.1, n=400, seed=1)
    p_hi, y_hi = _simulate_series(rho=0.9, n=400, seed=2)
    L_lo = politis_white_expected_block_length(p_lo, y_lo)
    L_hi = politis_white_expected_block_length(p_hi, y_hi)
    assert L_hi > L_lo >= 1.0


def test_block_bootstrap_auto_in_aggregate(tmp_path, monkeypatch, capsys):
    # Create two small prediction CSVs
    import pandas as pd
    from torpedocode.cli import aggregate as agg

    rng = np.random.default_rng(0)
    for k in range(2):
        n = 200
        x = rng.normal(size=n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = (rng.uniform(size=n) < p).astype(int)
        df = pd.DataFrame({"pred": p, "label": y})
        sub = tmp_path / f"inst{k}"
        sub.mkdir()
        df.to_csv(sub / "predictions_test.csv", index=False)

    out = tmp_path / "agg.json"
    argv = [
        "prog",
        "--pred-root",
        str(tmp_path),
        "--pred-pattern",
        "*/predictions_test.csv",
        "--output",
        str(out),
        "--block-bootstrap",
        "--auto-block",
        "--n-boot",
        "20",
    ]
    import sys

    monkeypatch.setattr(sys, "argv", argv)
    agg.main()
    s = out.read_text()
    assert "micro_ci_block" in s and "auroc_ci" in s
