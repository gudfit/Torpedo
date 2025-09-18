import numpy as np
import pytest

tda = pytest.importorskip("torpedocode_tda")


def test_queue_age_levels_basic():
    T, L = 5, 2
    sz = np.ones((T, L), dtype=float)
    pr = np.array([[100, 101], [100, 101], [100, 101], [100, 101], [100, 101]], dtype=float)
    dt = np.ones(T, dtype=np.float32)
    ages = tda.queue_age_levels(sz, pr, dt, None)
    assert isinstance(ages, list) and len(ages) == T * L
    ages = np.array(ages, dtype=np.float32).reshape(T, L)
    # With no changes, ages accumulate per step
    assert np.allclose(ages[:, 0], np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert np.allclose(ages[:, 1], np.array([0, 1, 2, 3, 4], dtype=np.float32))

    # Introduce a change at t=3 for level 1 and keep same value at t=4
    sz2 = sz.copy()
    sz2[3, 0] = 2.0
    sz2[4, 0] = 2.0
    ages2 = np.array(tda.queue_age_levels(sz2, pr, dt, None)).reshape(T, L)
    assert ages2[3, 0] == 0.0 and ages2[4, 0] == 1.0
    # Halting resets both levels
    halted = np.array([False, False, False, True, False])
    ages3 = np.array(tda.queue_age_levels(sz, pr, dt, halted)).reshape(T, L)
    assert ages3[3, 0] == 0.0 and ages3[4, 0] == 1.0
