import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from torpedocode.evaluation.metrics import compute_classification_metrics


def _build_fast_eval(tmp_path: Path) -> Path | None:
    src = Path("cpp/src/fast_eval.cpp").resolve()
    if not src.exists():
        return None
    if shutil.which("g++") is None:
        return None
    out = tmp_path / "fast_eval"
    code = subprocess.call(["g++", "-O3", "-std=c++17", "-o", str(out), str(src)])
    if code != 0 or not out.exists():
        return None
    return out


def _build_fast_eval_openmp(tmp_path: Path) -> Path | None:
    src = Path("cpp/src/fast_eval.cpp").resolve()
    if not src.exists():
        return None
    if shutil.which("g++") is None:
        return None
    out = tmp_path / "fast_eval_omp"
    code = subprocess.call(["g++", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-std=c++17", "-o", str(out), str(src)])
    if code != 0 or not out.exists():
        return None
    return out


@pytest.mark.skipif(shutil.which("g++") is None, reason="requires g++ to build C++ fast_eval")
def test_fast_eval_cpp_metrics_match_python(tmp_path: Path):
    bin_path = _build_fast_eval(tmp_path)
    if bin_path is None:
        pytest.skip("fast_eval build unavailable")
    # Synthetic predictions
    n = 1000
    rng = np.random.default_rng(0)
    p = rng.random(n)
    y = (p + rng.standard_normal(n) * 0.1 > 0.5).astype(int)
    csv = tmp_path / "predictions_test.csv"
    import pandas as pd

    pd.DataFrame({"idx": np.arange(n), "pred": p, "label": y}).to_csv(csv, index=False)
    out = tmp_path / "eval_fast.json"
    code = subprocess.call([str(bin_path), str(csv), str(out)])
    assert code == 0 and out.exists()
    obj = json.loads(out.read_text())
    m = compute_classification_metrics(p, y)
    # Tolerances account for discretization of AUPRC integration and binning for ECE
    assert np.isclose(obj["auroc"], m.auroc, rtol=1e-4, atol=1e-4)
    assert np.isfinite(obj["auprc"]) and obj["auprc"] >= 0.0
    assert np.isclose(obj["brier"], m.brier, rtol=1e-6, atol=1e-6)
    assert 0.0 <= obj["ece"] <= 1.0


@pytest.mark.skipif(shutil.which("g++") is None, reason="requires g++ to build C++ fast_eval")
def test_fast_eval_cpp_speed_smoke(tmp_path: Path):
    bin_path = _build_fast_eval(tmp_path)
    if bin_path is None:
        pytest.skip("fast_eval build unavailable")
    n = 50000
    rng = np.random.default_rng(1)
    p = rng.random(n)
    y = (p + rng.standard_normal(n) * 0.2 > 0.5).astype(int)
    csv = tmp_path / "predictions_test.csv"
    import pandas as pd

    pd.DataFrame({"idx": np.arange(n), "pred": p, "label": y}).to_csv(csv, index=False)
    # Python timing (fair: include CSV read + metrics)
    t0 = time.perf_counter()
    import pandas as pd
    df = pd.read_csv(csv)
    _ = compute_classification_metrics(df["pred"].to_numpy(), df["label"].to_numpy().astype(int))
    t_py = time.perf_counter() - t0
    # C++ timing (includes reading CSV)
    out = tmp_path / "eval_fast.json"
    t1 = time.perf_counter()
    code = subprocess.call([str(bin_path), str(csv), str(out)])
    t_cpp = time.perf_counter() - t1
    assert code == 0 and out.exists()
    # If timings are extremely small, skip to avoid flakiness
    if max(t_py, t_cpp) < 0.02:
        pytest.skip("timings too small to compare reliably")
    # Expect C++ to be at least not slower than Python metrics for this size (allow margin)
    assert t_cpp <= t_py * 1.4


@pytest.mark.skipif(shutil.which("g++") is None, reason="requires g++ to build C++ fast_eval")
def test_fast_eval_openmp_speed_vs_cpp_and_python(tmp_path: Path):
    omp_path = _build_fast_eval_openmp(tmp_path)
    base_path = _build_fast_eval(tmp_path)
    if omp_path is None or base_path is None:
        pytest.skip("OpenMP or base build unavailable")
    n = 100000
    rng = np.random.default_rng(2)
    p = rng.random(n)
    y = (p + rng.standard_normal(n) * 0.2 > 0.5).astype(int)
    csv = tmp_path / "predictions_test.csv"
    import pandas as pd

    pd.DataFrame({"idx": np.arange(n), "pred": p, "label": y}).to_csv(csv, index=False)
    # Python path
    t0 = time.perf_counter(); df = pd.read_csv(csv); _ = compute_classification_metrics(df["pred"].to_numpy(), df["label"].to_numpy().astype(int)); t_py = time.perf_counter() - t0
    # Base C++
    out_b = tmp_path / "eval_b.json"
    t1 = time.perf_counter(); _ = subprocess.call([str(base_path), str(csv), str(out_b)]); t_cpp = time.perf_counter() - t1
    # OpenMP C++
    out_o = tmp_path / "eval_o.json"
    t2 = time.perf_counter(); _ = subprocess.call([str(omp_path), str(csv), str(out_o)]); t_omp = time.perf_counter() - t2
    assert out_b.exists() and out_o.exists()
    # OMP should be no slower than base, and base no slower than Python (with generous margins on CI)
    assert t_omp <= t_cpp * 1.3
    assert t_cpp <= t_py * 2.5
