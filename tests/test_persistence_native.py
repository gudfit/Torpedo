import math

import numpy as np
import pytest

from torpedocode.config import TopologyConfig
import torpedocode.features.topological as topo


tda = pytest.importorskip("torpedocode_tda")


def _ref_persistence_image(births, pers, resolution, sigma, birth_range=None, pers_range=None):
    births = np.asarray(births, dtype=float)
    pers = np.asarray(pers, dtype=float)
    n = min(births.size, pers.size)
    points = []
    for i in range(n):
        b = float(births[i])
        p = float(pers[i])
        if math.isfinite(b) and math.isfinite(p):
            points.append((b, p, max(p, 0.0)))
    res = max(int(resolution), 1)
    if not points:
        return np.zeros((res * res,), dtype=np.float32)

    if birth_range is None:
        b_lo = min(b for b, _, _ in points)
        b_hi = max(b for b, _, _ in points)
    else:
        b_lo, b_hi = map(float, birth_range)
    if not math.isfinite(b_lo) or not math.isfinite(b_hi) or b_hi <= b_lo:
        b_lo, b_hi = float(b_lo), float(b_lo + 1.0)

    if pers_range is None:
        p_lo = min(p for _, p, _ in points)
        p_hi = max(p for _, p, _ in points)
    else:
        p_lo, p_hi = map(float, pers_range)
    if not math.isfinite(p_lo) or not math.isfinite(p_hi) or p_hi <= p_lo:
        p_lo, p_hi = float(p_lo), float(p_lo + 1.0)

    dx = (b_hi - b_lo) / (res - 1) if res > 1 else 1.0
    dy = (p_hi - p_lo) / (res - 1) if res > 1 else 1.0
    sigma = abs(float(sigma)) or 1e-9
    inv_sigma_sq = 1.0 / (2.0 * sigma * sigma)

    out = np.zeros((res, res), dtype=np.float64)
    for iy in range(res):
        py = p_lo + iy * dy
        for ix in range(res):
            px = b_lo + ix * dx
            acc = 0.0
            for b, p, w in points:
                db = px - b
                dp = py - p
                acc += math.exp(-(db * db + dp * dp) * inv_sigma_sq) * w
            out[iy, ix] = acc
    return out.astype(np.float32).ravel()


def _ref_persistence_landscape(diagram, k, resolution, summary_mode="mean"):
    diag = np.asarray(diagram, dtype=float)
    levels = max(int(k), 1)
    if diag.size == 0:
        return np.zeros((levels,), dtype=np.float32)

    pts = []
    for row in diag:
        b = float(row[0])
        p = float(row[1])
        d = b + p
        if math.isfinite(b) and math.isfinite(d) and d > b:
            pts.append((b, d))
    if not pts:
        return np.zeros((levels,), dtype=np.float32)

    x_min = min(b for b, _ in pts)
    x_max = max(d for _, d in pts)
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = float(x_min), float(x_min + 1.0)

    res = max(int(resolution), 2)
    xs = np.linspace(x_min, x_max, res)
    values = np.zeros((levels, res), dtype=np.float64)
    for xi, x in enumerate(xs):
        vals = []
        for b, d in pts:
            if x <= b or x >= d:
                continue
            left = x - b
            right = d - x
            val = left if left < right else right
            if val > 0.0:
                vals.append(val)
        if vals:
            vals.sort(reverse=True)
            for level in range(min(levels, len(vals))):
                values[level, xi] = vals[level]

    if summary_mode == "max":
        summary = values.max(axis=1)
    else:
        summary = values.mean(axis=1)
    return summary.astype(np.float32)


def test_persistence_image_native_matches_reference():
    births = np.array([0.05, 0.2, 0.7], dtype=float)
    pers = np.array([0.1, 0.05, 0.3], dtype=float)
    res = 32
    sigma = 0.07
    birth_range = (0.0, 1.0)
    pers_range = (0.0, 0.5)

    native = np.asarray(
        tda.persistence_image(births, pers, res, sigma, birth_range, pers_range),
        dtype=np.float32,
    )
    ref = _ref_persistence_image(births, pers, res, sigma, birth_range, pers_range)
    np.testing.assert_allclose(native, ref, rtol=1e-5, atol=1e-6)


def test_persistence_landscape_native_matches_reference():
    diagram = np.array([[0.05, 0.15], [0.2, 0.3], [0.1, 0.05]], dtype=float)
    k = 3
    resolution = 64
    for summary_mode in ("mean", "max"):
        native = np.asarray(
            tda.persistence_landscape(diagram, k, resolution, summary_mode),
            dtype=np.float32,
        )
        ref = _ref_persistence_landscape(diagram, k, resolution, summary_mode)
        np.testing.assert_allclose(native, ref, rtol=1e-6, atol=1e-6)


def _rolling_reference_and_native(cfg: TopologyConfig, timestamps: np.ndarray, series: np.ndarray, monkeypatch) -> tuple[np.ndarray, np.ndarray]:
    native_mod = topo._tda_native
    if native_mod is None:
        pytest.skip("torpedocode_tda native module unavailable")
    monkeypatch.setattr(topo, "_tda_native", None)
    gen_py = topo.TopologicalFeatureGenerator(cfg)
    ref = gen_py.rolling_transform(timestamps, series)
    monkeypatch.setattr(topo, "_tda_native", native_mod)
    gen_native = topo.TopologicalFeatureGenerator(cfg)
    out = gen_native.rolling_transform(timestamps, series)
    return ref, out


def test_rolling_transform_native_cubical_matches_python(monkeypatch):
    pytest.importorskip("gudhi")
    cfg = TopologyConfig(
        window_sizes_s=[1, 2],
        complex_type="cubical",
        persistence_representation="landscape",
        landscape_levels=3,
        max_homology_dimension=1,
    )
    rng = np.random.default_rng(123)
    series = rng.standard_normal((12, 20)).astype(np.float32)
    timestamps = (np.arange(series.shape[0], dtype=np.int64) * 1_000_000_000)
    ref, out = _rolling_reference_and_native(cfg, timestamps, series, monkeypatch)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_rolling_transform_native_vr_image_matches_python(monkeypatch):
    pytest.importorskip("ripser")
    pytest.importorskip("persim")
    cfg = TopologyConfig(
        window_sizes_s=[1, 2],
        complex_type="vietoris_rips",
        persistence_representation="image",
        image_resolution=8,
        image_bandwidth=0.05,
        max_homology_dimension=1,
    )
    rng = np.random.default_rng(321)
    series = rng.standard_normal((10, 6)).astype(np.float32)
    timestamps = (np.arange(series.shape[0], dtype=np.int64) * 1_000_000_000)
    ref, out = _rolling_reference_and_native(cfg, timestamps, series, monkeypatch)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
