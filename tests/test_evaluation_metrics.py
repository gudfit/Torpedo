import numpy as np

from torpedocode.evaluation.metrics import (
    compute_calibration_report,
    compute_classification_metrics,
    compute_point_process_diagnostics,
)


def test_compute_classification_metrics_basic():
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(int)
    m = compute_classification_metrics(p, y)
    assert 0.0 <= m.brier <= 1.0
    assert np.isfinite(m.auroc)
    assert np.isfinite(m.auprc)
    assert np.isfinite(m.ece)


def test_compute_calibration_report_equal_frequency_shapes():
    p = np.linspace(0, 1, 57)
    y = (p > 0.2).astype(int)
    r = compute_calibration_report(p, y, num_bins=15)
    assert r.bin_confidence.shape == (15,)
    assert r.bin_accuracy.shape == (15,)


def test_point_process_diagnostics_numpy_path():
    rng = np.random.default_rng(0)
    xi = rng.exponential(1.0, size=1000)
    diag = compute_point_process_diagnostics(
        xi, empirical_frequencies=np.array([0.7, 0.3]), model_frequencies=np.array([0.5, 0.5])
    )
    assert 0.0 <= diag.coverage_error <= 2.0
    assert np.isfinite(diag.nll_per_event)
