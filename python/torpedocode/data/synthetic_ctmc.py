"""Simple synthetic marked CTMC generator for pretraining.

Generates an event stream with M event types driven by a continuous-time
Markov chain on a small latent state space, with exponential waiting times
and log-normal marks. Features are emitted from state-dependent Gaussians
to tie the observation process to latent dynamics (rather than white noise).
Optionally, a lightweight synthetic "topology" embedding is produced from
rolling windows over the features to exercise the topology head during
pretraining without external PH dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CTMCConfig:
    num_states: int = 3
    num_event_types: int = 6
    avg_rate: float = 5.0
    mark_mu: float = 0.0
    mark_sigma: float = 0.5
    T: int = 1000
    # Feature emission and topology options
    feature_dim: int | None = None
    feature_state_scale: float = 1.0  # magnitude of state-dependent mean shifts
    feature_noise_sigma: float = 0.5  # observation noise std
    topo_window: int = 16            # rolling window length for synthetic topology
    topo_levels: int = 3             # number of levels in landscape-like summary
    emit_topology: bool = True       # whether to compute synthetic topology


def _row_stochastic(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    m = np.clip(m, 0.0, None)
    m = m / (m.sum(axis=1, keepdims=True) + 1e-12)
    return m


def generate_ctmc_sequence(
    cfg: CTMCConfig, rng: np.random.Generator | None = None
) -> dict[str, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    S, M, T = cfg.num_states, cfg.num_event_types, cfg.T
    A = rng.uniform(size=(S, S))
    A = _row_stochastic(A)
    E = rng.uniform(size=(S, M))
    E = _row_stochastic(E)
    state_rate = rng.uniform(0.5, 1.5, size=S)

    state = int(rng.integers(0, S))
    etypes = np.zeros((T,), dtype=np.int64)
    dt = np.zeros((T,), dtype=np.float32)
    sizes = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        etypes[t] = int(rng.choice(M, p=E[state]))
        rate = max(cfg.avg_rate * state_rate[state], 1e-6)
        dt[t] = float(rng.exponential(1.0 / rate))
        sizes[t] = float(np.exp(rng.normal(cfg.mark_mu, cfg.mark_sigma)))
        state = int(rng.choice(S, p=A[state]))

    # State-dependent feature emissions in R^F
    F = int(cfg.feature_dim) if cfg.feature_dim is not None else int(M)
    means = rng.normal(0.0, 1.0, size=(S, F)).astype(float)
    # Scale means by state index to separate states if desired
    for s in range(S):
        means[s] *= (1.0 + cfg.feature_state_scale * (s / max(S - 1, 1)))
    sig = float(max(cfg.feature_noise_sigma, 1e-6))
    features = np.zeros((T, F), dtype=np.float32)
    for t in range(T):
        mu = means[int(etypes[t] % S)]  # tie to emitting state indirectly via event type
        features[t] = (mu + rng.normal(0.0, sig, size=(F,))).astype(np.float32)

    out = {
        "event_type_ids": etypes,
        "delta_t": dt,
        "sizes": sizes,
        "features": features,
    }

    # Lightweight synthetic topology: rolling window "landscape-like" summary.
    # For each time t, consider last W rows of features; for each column, take K
    # quantiles of (x - min(x)) as a crude persistence proxy, then average across
    # columns to produce a fixed-length vector of length K.
    if bool(cfg.emit_topology) and cfg.topo_levels > 0 and cfg.topo_window > 1:
        W = int(cfg.topo_window)
        K = int(cfg.topo_levels)
        topo = np.zeros((T, K), dtype=np.float32)
        qs = np.linspace(0.2, 1.0, K)  # avoid extreme tails
        for i in range(T):
            j0 = max(0, i - W + 1)
            slab = features[j0 : i + 1]
            # normalize per-feature to highlight spread (birth-persistence analogue)
            mn = np.min(slab, axis=0, initial=0.0)
            span = np.maximum(np.max(slab, axis=0, initial=0.0) - mn, 1e-8)
            norm = (slab - mn) / span
            # aggregate over time within window using quantiles per feature, average across features
            q_feat = np.quantile(norm, qs, axis=0)  # [K, F]
            topo[i] = np.mean(q_feat, axis=1).astype(np.float32)
        out["topology"] = topo

    return out


__all__ = ["CTMCConfig", "generate_ctmc_sequence"]
