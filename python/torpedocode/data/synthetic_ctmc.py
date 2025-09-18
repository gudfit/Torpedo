"""State-dependent LOB CTMC generator for pretraining (Cont et al.-style).

Implements a continuous-time Markov chain on the order book state consisting of
volumes at L bid/ask levels and a mid-price proxy. Event types m in
{MO+, MO-, LO+, LO-, CX+, CX-} occur with state-dependent intensities λ_m(X_t),
piecewise constant between events. Upon each event, the state is updated by
applying a local transition (e.g., reducing best-ask queue for MO+, inserting
volume for LO+, removing for CX+, with price/level shifts when queues deplete).

If emit_topology is enabled, a liquidity-surface persistent-homology embedding is
optionally computed via the project TopologicalFeatureGenerator (if available);
otherwise a lightweight rolling quantile summary is produced as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

try:  # optional, used only if emit_topology and backend available
    from ..config import TopologyConfig
    from ..features.topological import TopologicalFeatureGenerator

    _HAVE_TDA = True
except Exception:  # pragma: no cover
    _HAVE_TDA = False


@dataclass
class CTMCConfig:
    # Simulation controls
    T: int = 1000  # number of events to simulate
    levels: int = 10  # LOB levels L
    tick_size: float = 0.01
    # Optional hints accepted for compatibility with callers/tests; generator derives
    # event-type expansion internally and emits features of size 2*levels by default.
    num_event_types: int | None = None
    feature_dim: int | None = None

    # Baseline intensities (per second) and sensitivities
    mo_base: float = 2.0  # base market order rate per side
    lo_base: float = 5.0  # base limit order rate per side
    cx_per_volume: float = 0.02  # cancel rate proportional to current depth
    mo_imbalance_sensitivity: float = 2.0  # sensitivity of MO to imbalance
    lo_imbalance_sensitivity: float = 1.0  # sensitivity of LO to imbalance

    # Size distributions (log-normal in natural log units)
    mark_mu_mo: float = 0.0
    mark_sigma_mo: float = 0.5
    mark_mu_lo: float = 0.0
    mark_sigma_lo: float = 0.5
    mark_mu_cx: float = -0.2
    mark_sigma_cx: float = 0.5

    # Initial depth distribution parameters (gamma-like via sum of exponentials)
    init_depth_mean: float = 100.0
    init_depth_cv: float = 0.5

    # Level selection for LO/CX (geometric decay)
    level_geom_p: float = 0.35

    # Topology options
    emit_topology: bool = True
    topo_window: int = 32
    topo_stride: int = 1
    topo_representation: Literal["landscape", "image"] = "landscape"
    topo_levels: int = 3
    image_resolution: int = 64
    image_bandwidth: float = 0.05

    # Event type expansion
    expand_types_by_level: bool = False
    mo_expand_by_level: bool = False


def _sample_lognormal(rng: np.random.Generator, mu: float, sigma: float) -> float:
    return float(np.exp(rng.normal(mu, sigma)))


def generate_ctmc_sequence(
    cfg: CTMCConfig, rng: np.random.Generator | None = None
) -> dict[str, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    L = int(cfg.levels)
    Tn = int(cfg.T)

    # Initialize volumes at bid/ask levels from a gamma-ish distribution
    mean = float(cfg.init_depth_mean)
    cv = float(max(cfg.init_depth_cv, 1e-6))
    shape = 1.0 / (cv * cv)
    scale = mean / shape

    def _init_depths():
        return rng.gamma(shape, scale, size=L).astype(np.float64)

    Vb = _init_depths()
    Va = _init_depths()
    price = 0  # price in ticks; relative mid proxy

    # Helper: imbalance over first K levels
    def imbalance(k: int = min(5, L)) -> float:
        Db = float(np.sum(Vb[:k]))
        Da = float(np.sum(Va[:k]))
        return (Db - Da) / (Db + Da + 1e-9)

    # Level selection distribution for LO/CX (geometric)
    p = float(min(max(cfg.level_geom_p, 1e-6), 0.99))
    level_probs = (1 - p) * (p ** np.arange(L))
    level_probs = level_probs / level_probs.sum()

    # Storage
    etypes = np.zeros((Tn,), dtype=np.int64)
    dt = np.zeros((Tn,), dtype=np.float32)
    sizes = np.zeros((Tn,), dtype=np.float32)
    features = np.zeros((Tn, 2 * L), dtype=np.float32)
    side = np.empty((Tn,), dtype=object)
    level = np.zeros((Tn,), dtype=np.int64)

    # Event type ids: 0:MO+, 1:MO-, 2:LO+, 3:LO-, 4:CX+, 5:CX-
    def intensities() -> np.ndarray:
        I = imbalance()
        mo_plus = max(cfg.mo_base * max(0.0, 1.0 + cfg.mo_imbalance_sensitivity * I), 1e-12)
        mo_minus = max(cfg.mo_base * max(0.0, 1.0 - cfg.mo_imbalance_sensitivity * I), 1e-12)
        lo_plus = max(cfg.lo_base * max(0.0, 1.0 - cfg.lo_imbalance_sensitivity * I), 1e-12)
        lo_minus = max(cfg.lo_base * max(0.0, 1.0 + cfg.lo_imbalance_sensitivity * I), 1e-12)
        cx_plus = float(cfg.cx_per_volume) * float(np.sum(Vb))
        cx_minus = float(cfg.cx_per_volume) * float(np.sum(Va))
        return np.array([mo_plus, mo_minus, lo_plus, lo_minus, cx_plus, cx_minus], dtype=float)

    def _shift_ask():
        nonlocal Va, price
        # price up by 1 tick; level-2 becomes level-1, append new depth at L
        Va = np.concatenate([Va[1:], rng.gamma(shape, scale, size=1)])
        price += 1

    def _shift_bid():
        nonlocal Vb, price
        # price down by 1 tick; level-2 becomes level-1, append new depth at L
        Vb = np.concatenate([Vb[1:], rng.gamma(shape, scale, size=1)])
        price -= 1

    # Event-type id mapping when expanding by level
    def _etype_id(m: int, lvl: int | None) -> int:
        # Optional MO per-level expansion
        if bool(cfg.mo_expand_by_level):
            if m == 0:  # MO+
                l = int(0 if lvl is None else max(0, min(L - 1, lvl)))
                return l
            if m == 1:  # MO-
                l = int(0 if lvl is None else max(0, min(L - 1, lvl)))
                return L + l
            base = 2 * L
        else:
            if m == 0:
                return 0
            if m == 1:
                return 1
            base = 2
        # LO/CX per-level expansion (optional)
        if not bool(cfg.expand_types_by_level):
            return int(m)
        l = int(0 if lvl is None else max(0, min(L - 1, lvl)))
        if m == 2:  # LO+
            return base + l
        if m == 3:  # LO-
            return base + L + l
        if m == 4:  # CX+
            return base + 2 * L + l
        # CX-
        return base + 3 * L + l

    for t in range(Tn):
        lam = intensities()
        Lam = float(np.sum(lam))
        if Lam <= 0:
            Lam = 1e-6
        dt[t] = float(rng.exponential(1.0 / Lam))
        m = int(rng.choice(6, p=lam / Lam))

        if m == 0:  # MO+
            s = _sample_lognormal(rng, cfg.mark_mu_mo, cfg.mark_sigma_mo)
            # Sample intended penetration level for labeling when MO per-level enabled
            l = int(rng.choice(L, p=level_probs)) if bool(cfg.mo_expand_by_level) else 0
            # Consume ask queues; walk through levels if depleted
            rem = s
            while rem > 0 and l < L:
                take = min(rem, Va[l])
                Va[l] -= take
                rem -= take
                if Va[l] <= 1e-9:
                    if l == 0:
                        _shift_ask()
                        # After shift, continue on new level-1
                        l = 0
                        continue
                    else:
                        l += 1
                else:
                    break
            side[t] = "ask"
            level[t] = 1
            etypes[t] = _etype_id(m, l if bool(cfg.mo_expand_by_level) else None)
        elif m == 1:  # MO-
            s = _sample_lognormal(rng, cfg.mark_mu_mo, cfg.mark_sigma_mo)
            l = int(rng.choice(L, p=level_probs)) if bool(cfg.mo_expand_by_level) else 0
            rem = s
            while rem > 0 and l < L:
                take = min(rem, Vb[l])
                Vb[l] -= take
                rem -= take
                if Vb[l] <= 1e-9:
                    if l == 0:
                        _shift_bid()
                        l = 0
                        continue
                    else:
                        l += 1
                else:
                    break
            side[t] = "bid"
            level[t] = 1
            etypes[t] = _etype_id(m, l if bool(cfg.mo_expand_by_level) else None)
        elif m == 2:  # LO+
            s = _sample_lognormal(rng, cfg.mark_mu_lo, cfg.mark_sigma_lo)
            l = int(rng.choice(L, p=level_probs))
            Vb[l] += s
            side[t] = "bid"
            level[t] = l + 1
            etypes[t] = _etype_id(m, l)
        elif m == 3:  # LO-
            s = _sample_lognormal(rng, cfg.mark_mu_lo, cfg.mark_sigma_lo)
            l = int(rng.choice(L, p=level_probs))
            Va[l] += s
            side[t] = "ask"
            level[t] = l + 1
            etypes[t] = _etype_id(m, l)
        elif m == 4:  # CX+
            s = _sample_lognormal(rng, cfg.mark_mu_cx, cfg.mark_sigma_cx)
            l = int(rng.choice(L, p=level_probs))
            d = min(s, Vb[l])
            Vb[l] -= d
            side[t] = "bid"
            level[t] = l + 1
            etypes[t] = _etype_id(m, l)
        else:  # CX-
            s = _sample_lognormal(rng, cfg.mark_mu_cx, cfg.mark_sigma_cx)
            l = int(rng.choice(L, p=level_probs))
            d = min(s, Va[l])
            Va[l] -= d
            side[t] = "ask"
            level[t] = l + 1
            etypes[t] = _etype_id(m, l)

        sizes[t] = float(max(s, 1e-8))
        # Feature vector at event time: concatenated depths
        features[t, :L] = Vb.astype(np.float32)
        features[t, L:] = Va.astype(np.float32)

    out = {
        "event_type_ids": etypes,
        "delta_t": dt,
        "sizes": sizes,
        "features": features,
        "level": level,
        "side": np.array([s if isinstance(s, str) else str(s) for s in side], dtype=object),
    }

    # Optional topology embedding from liquidity surface via project TDA
    if bool(cfg.emit_topology) and cfg.topo_window > 1:
        try:
            if _HAVE_TDA:
                ts = (np.cumsum(dt) * 1e9).astype("int64").astype("datetime64[ns]")
                if cfg.topo_representation == "image":
                    topo_cfg = TopologyConfig(
                        window_sizes_s=[1],
                        complex_type="cubical",
                        max_homology_dimension=1,
                        persistence_representation="image",
                        image_resolution=int(cfg.image_resolution),
                        image_bandwidth=float(cfg.image_bandwidth),
                        use_liquidity_surface=True,
                    )
                else:
                    topo_cfg = TopologyConfig(
                        window_sizes_s=[1],
                        complex_type="cubical",
                        max_homology_dimension=1,
                        persistence_representation="landscape",
                        landscape_levels=int(cfg.topo_levels),
                        use_liquidity_surface=True,
                    )
                gen = TopologicalFeatureGenerator(topo_cfg)
                Z = gen.rolling_transform(ts, features, stride=max(1, int(cfg.topo_stride)))
                out["topology"] = Z.astype(np.float32)
            else:
                # Fallback rolling quantile summary
                W = int(cfg.topo_window)
                K = 3
                topo = np.zeros((Tn, K), dtype=np.float32)
                qs = np.linspace(0.2, 1.0, K)
                for i in range(Tn):
                    j0 = max(0, i - W + 1)
                    slab = features[j0 : i + 1]
                    # imbalance surface proxy over [time, levels]
                    bids = slab[:, :L]
                    asks = slab[:, L:]
                    eps = 1e-6
                    surf = (bids - asks) / (np.abs(bids) + np.abs(asks) + eps)
                    q_feat = np.quantile(surf, qs, axis=None)
                    topo[i] = q_feat.astype(np.float32)
                out["topology"] = topo
        except Exception:
            pass

    # Optional remapping of event types to requested cardinality for compatibility
    try:
        if getattr(cfg, "num_event_types", None) is not None:
            M = int(getattr(cfg, "num_event_types"))
            if M <= 0:
                M = 1
            e = out["event_type_ids"].copy()
            if M == 6:
                pass  # already in 0..5
            elif M == 4:
                # Collapse cancellations into corresponding LO side
                # 0:MO+, 1:MO-, 2:LO+, 3:LO-
                mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 3}
                e = np.vectorize(lambda x: mapping.get(int(x), int(x) % 4))(e).astype(np.int64)
            elif M == 2:
                # Collapse by side (buys vs sells): {MO+,LO+,CX+}->0, {MO-,LO-,CX-}->1
                mapping = {0: 0, 2: 0, 4: 0, 1: 1, 3: 1, 5: 1}
                e = np.vectorize(lambda x: mapping.get(int(x), int(x) & 1))(e).astype(np.int64)
            else:
                # Generic fallback: modulo
                e = (e % M).astype(np.int64)
            out["event_type_ids"] = e
    except Exception:
        pass

    return out


__all__ = ["CTMCConfig", "generate_ctmc_sequence"]
