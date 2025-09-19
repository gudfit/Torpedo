import math

import numpy as np
import pytest
import torch

from torpedocode.utils.ops import has_torpedocode_op


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


def _manual_tpp_loss(
    intensities: torch.Tensor,
    mark_mu: torch.Tensor,
    mark_log_sigma: torch.Tensor,
    event_types: torch.Tensor,
    delta_t: torch.Tensor,
    sizes: torch.Tensor | None,
    mask: torch.Tensor | None,
    smoothness_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = intensities.dtype
    device = intensities.device
    lam = intensities.clamp_min(1e-12)
    gather_idx = event_types.clamp_min(0).unsqueeze(-1)
    event_mask = (event_types >= 0).to(dtype=dtype)
    lam_evt = torch.gather(lam, dim=-1, index=gather_idx).squeeze(-1)
    log_lam_evt = torch.log(lam_evt) * event_mask
    comp = (lam.sum(dim=-1) * delta_t).sum(dim=1)
    if sizes is not None:
        log_sizes = torch.log(sizes.clamp_min(1e-12))
        mu_evt = torch.gather(mark_mu, dim=-1, index=gather_idx).squeeze(-1)
        log_sig_evt = torch.gather(mark_log_sigma, dim=-1, index=gather_idx).squeeze(-1)
        z = (log_sizes - mu_evt) / (torch.exp(log_sig_evt) + 1e-12)
        const = 0.5 * torch.log(torch.tensor(2 * math.pi, device=device, dtype=dtype))
        mark = ((0.5 * z.pow(2) + log_sig_evt + log_sizes + const) * event_mask).sum(dim=1)
    else:
        mark = torch.zeros_like(comp)
    nll = -(log_lam_evt.sum(dim=1)) + comp + mark
    nll_mean = nll.mean()

    dl = lam[:, 1:] - lam[:, :-1]
    if mask is not None:
        pair_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        diff_sq = dl.pow(2) * pair_mask
        if smoothness_mode == "none":
            smooth = diff_sq.sum()
        elif smoothness_mode == "per_seq":
            per_seq = diff_sq.sum(dim=(1, 2))
            denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
            smooth = (per_seq / denom).mean()
        else:
            denom = pair_mask.sum().clamp_min(1.0)
            smooth = diff_sq.sum() / denom
    else:
        diff_sq = dl.pow(2)
        if smoothness_mode == "none":
            smooth = diff_sq.sum()
        elif smoothness_mode == "per_seq":
            per_seq = diff_sq.sum(dim=(1, 2))
            denom = torch.tensor(
                diff_sq.shape[1] * diff_sq.shape[2], device=device, dtype=dtype
            ).clamp_min(1.0)
            smooth = (per_seq / denom).mean()
        else:
            smooth = diff_sq.sum(dim=(-1, -2)).mean()
    return nll_mean, smooth


@pytest.mark.parametrize("smoothness_mode,mode_idx", [("none", 0), ("global", 1), ("per_seq", 2)])
@pytest.mark.parametrize("use_sizes", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_native_tpp_loss_matches_python(monkeypatch, smoothness_mode, mode_idx, use_sizes, use_mask):
    monkeypatch.setenv("TORPEDOCODE_AUTO_BUILD_OPS", "1")
    if not has_torpedocode_op():
        pytest.skip("native op not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    B, T, M = 2, 5, 3
    intensities = torch.rand(B, T, M, device=device, dtype=dtype) + 0.05
    mark_mu = torch.randn(B, T, M, device=device, dtype=dtype)
    mark_log_sigma = torch.randn(B, T, M, device=device, dtype=dtype) * 0.2
    event_types = torch.randint(low=-1, high=M, size=(B, T), device=device)
    delta_t = torch.rand(B, T, device=device, dtype=dtype)
    sizes = torch.rand(B, T, device=device, dtype=dtype) + 0.05 if use_sizes else None
    mask = torch.rand(B, T, device=device, dtype=dtype) if use_mask else None
    if mask is not None:
        mask[:, -1] = 0.0

    expected_nll, expected_smooth = _manual_tpp_loss(
        intensities,
        mark_mu,
        mark_log_sigma,
        event_types,
        delta_t,
        sizes,
        mask,
        smoothness_mode,
    )

    out = torch.ops.torpedocode.tpp_loss(
        intensities,
        mark_mu,
        mark_log_sigma,
        event_types,
        delta_t,
        sizes,
        mask,
        mode_idx,
    )

    assert isinstance(out, (tuple, list))
    got_nll, got_smooth = out
    assert torch.allclose(got_nll, expected_nll, atol=1e-6, rtol=1e-5)
    assert torch.allclose(got_smooth, expected_smooth, atol=1e-6, rtol=1e-5)
