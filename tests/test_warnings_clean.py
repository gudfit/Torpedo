import warnings
import numpy as np
import pandas as pd

from torpedocode.data.preprocess import label_instability


def test_label_instability_no_warnings():
    ts = pd.date_range("2025-01-01", periods=10, freq="s", tz="UTC")
    mid = pd.Series(np.linspace(100, 101, 10))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        labels = label_instability(mid, pd.Series(ts), horizons_s=[1], horizons_events=[2], threshold_eta=0.0)
    # Assert no warnings were captured
    assert len(rec) == 0
    assert "instability_s_1" in labels and "instability_e_2" in labels
