import numpy as np
import torch


def test_tpp_nll_exact_compensator():
    # Tiny example: T=3, M=2
    lam = np.array([
        [0.5, 1.5],
        [1.0, 1.0],
        [2.0, 0.5],
    ], dtype=np.float64)
    et = np.array([0, 1, 0], dtype=np.int64)
    dt = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    # Expected NLL = -sum log lambda_{m_i}(t_i) + sum (sum_m lambda_m) * dt
    log_sum = np.log(lam[np.arange(3), et]).sum()
    comp = (lam.sum(axis=1) * dt).sum()
    expected = (-log_sum + comp) / 3.0

    from torpedocode.evaluation.tpp import nll_per_event_from_arrays

    got = nll_per_event_from_arrays(lam, et, dt)
    assert np.isclose(got, expected, rtol=1e-6, atol=1e-8)


def test_smoothness_penalty_matches_diff_of_intensities():
    # Build outputs-like object with known intensities across time
    B, T, M = 1, 4, 2
    lam0 = torch.tensor([1.0, 2.0, 2.0, 1.0], dtype=torch.float32).view(1, T, 1)
    lam1 = torch.tensor([1.0, 1.0, 3.0, 1.0], dtype=torch.float32).view(1, T, 1)
    intensities = {"event_0": lam0, "event_1": lam1}

    class _Out:
        pass

    out = _Out()
    out.intensities = intensities
    out.mark_params = {f"event_{i}": (torch.zeros((B, T, 1)), torch.zeros((B, T, 1))) for i in range(M)}
    out.instability_logits = torch.zeros((B, T, 1))

    # Batch with required keys for TPP block but no sizes to skip mark NLL
    batch = {
        "features": torch.zeros((B, T, 1)),
        "event_type_ids": torch.tensor([[0, 1, 0, 0]]),
        "delta_t": torch.ones((B, T)),
    }

    from torpedocode.training.losses import HybridLossComputer

    loss = HybridLossComputer(lambda_cls=0.0, beta=1.0, gamma=0.0)
    lo = loss(out, batch, [])

    # Expected smoothness = sum over time diffs of squared diffs across types
    # dl over time (t1-t0, t2-t1, t3-t2):
    # type0: [1, 0, -1] -> squares [1,0,1]
    # type1: [0, 2, -2] -> squares [0,4,4]
    expected_smooth = float(1 + 0 + 1 + 0 + 4 + 4)

    assert np.isclose(float(lo.smoothness.detach().cpu()), expected_smooth, rtol=1e-6, atol=1e-8)
