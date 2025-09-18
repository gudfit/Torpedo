import numpy as np

from torpedocode.evaluation.metrics import compute_classification_metrics, compute_calibration_report


def test_ece_weights_use_bin_sizes():
    # 57 samples into 15 bins gives some bins with size 4 and some with 3
    p = np.linspace(0, 1, 57)
    y = (p > 0.4).astype(int)
    m = compute_classification_metrics(p, y)
    # manual ECE using the same equal-frequency partition
    rep = compute_calibration_report(p, y, num_bins=15)
    n = len(p)
    base = n // 15
    rem = n % 15
    sizes = np.full(15, base, dtype=float)
    sizes[:rem] += 1
    weights = sizes / n
    ece_manual = float(np.sum(np.abs(rep.bin_accuracy - rep.bin_confidence) * weights))
    assert np.isclose(m.ece, ece_manual)

