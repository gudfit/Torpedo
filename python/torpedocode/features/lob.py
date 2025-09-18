"""Conventional microstructure feature engineering.

Implements the feature families specified in the methodology, including:
- Depths and cumulative depths at multiple levels
- Multi-scale imbalance
- Prices, spreads, mid-price and short-horizon log-returns
- Queue ages and order-flow counts over causal windows
- Temporal covariates like inter-event times and time-of-day

This module assumes snapshots are aligned to event times with columns:
- bid_price_{l}, ask_price_{l}, bid_size_{l}, ask_size_{l} for l=1..L
- timestamp (ns or ISO8601), event_type (categorical string)
Optional columns such as `last_update_bid_{l}`, `last_update_ask_{l}` encode
the timestamp of the last update to that queue; if absent, queue ages are
computed approximately by detecting size/price changes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional
    import torpedocode_tda as _tda_native  # type: ignore
except Exception:  # pragma: no cover
    _tda_native = None


def build_lob_feature_matrix(
    frame: pd.DataFrame,
    levels: int,
    *,
    imbalance_ks: Iterable[int] = (1, 3, 5, 10),
    ret_horizons: Iterable[int] = (1, 5, 10),
    count_windows: Iterable[pd.Timedelta] = (pd.Timedelta(seconds=1), pd.Timedelta(seconds=5)),
    ewma_halflives: Iterable[float] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Construct level-wise and aggregated features from snapshots."""

    if frame.empty:
        return np.zeros((0, levels * 2), dtype=np.float32), {
            "spreads": np.zeros((0,), dtype=np.float32),
            "mid": np.zeros((0,), dtype=np.float32),
            "imbalance@k": np.zeros((0, len(tuple(imbalance_ks))), dtype=np.float32),
            "cum_depth_b": np.zeros((0, len(tuple(imbalance_ks))), dtype=np.float32),
            "cum_depth_a": np.zeros((0, len(tuple(imbalance_ks))), dtype=np.float32),
            "ret": np.zeros((0, len(tuple(ret_horizons))), dtype=np.float32),
            "delta_t": np.zeros((0,), dtype=np.float32),
        }

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    bid_sizes = [
        frame.get(f"bid_size_{l}", pd.Series(index=frame.index, dtype=float).fillna(0.0))
        for l in range(1, levels + 1)
    ]
    ask_sizes = [
        frame.get(f"ask_size_{l}", pd.Series(index=frame.index, dtype=float).fillna(0.0))
        for l in range(1, levels + 1)
    ]
    bid_prices = [
        frame.get(f"bid_price_{l}", pd.Series(index=frame.index, dtype=float).fillna(np.nan))
        for l in range(1, levels + 1)
    ]
    ask_prices = [
        frame.get(f"ask_price_{l}", pd.Series(index=frame.index, dtype=float).fillna(np.nan))
        for l in range(1, levels + 1)
    ]

    bid_sizes_mat = np.stack([s.to_numpy(dtype=np.float32) for s in bid_sizes], axis=1)
    ask_sizes_mat = np.stack([s.to_numpy(dtype=np.float32) for s in ask_sizes], axis=1)
    base_features = np.concatenate([bid_sizes_mat, ask_sizes_mat], axis=1)
    best_bid = np.nan_to_num(bid_prices[0].to_numpy(dtype=np.float64), nan=0.0)
    best_ask = np.nan_to_num(ask_prices[0].to_numpy(dtype=np.float64), nan=0.0)
    spreads = (best_ask - best_bid).astype(np.float32)
    mid = ((best_ask + best_bid) / 2.0).astype(np.float32)
    ks = [k for k in imbalance_ks if k >= 1]
    ks = [min(k, levels) for k in ks]
    cum_b = np.cumsum(bid_sizes_mat, axis=1)
    cum_a = np.cumsum(ask_sizes_mat, axis=1)
    cum_b_sel = np.stack([cum_b[:, k - 1] for k in ks], axis=1)
    cum_a_sel = np.stack([cum_a[:, k - 1] for k in ks], axis=1)
    eps = 1e-6
    imbalance_k = (cum_b_sel - cum_a_sel) / (cum_b_sel + cum_a_sel + eps)
    ret_h = list(ret_horizons)
    rets = []
    mid_safe = np.clip(mid, a_min=1e-6, a_max=None).astype(np.float64)
    log_mid = np.log(mid_safe)
    for h in ret_h:
        # Use past values only: r_t(h) = log_mid[t] - log_mid[t-h]
        # For the first h samples, fall back to zero difference (no history).
        shifted = np.roll(log_mid, h)
        shifted[:h] = log_mid[:h]
        rets.append((log_mid - shifted).astype(np.float32))
    ret_mat = np.stack(rets, axis=1) if rets else np.zeros((len(frame), 0), dtype=np.float32)

    # Inter-event time
    ts = frame["timestamp"].astype("int64").to_numpy()
    delta_t = np.diff(ts, prepend=ts[0]) / 1e9  # seconds
    delta_t = delta_t.astype(np.float32)

    # Event-type counts over causal windows (per type + total) — vectorized
    counts_mat = np.zeros((len(frame), 0), dtype=np.float32)
    if "event_type" in frame.columns and len(frame) > 0:
        idx = (
            frame["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        ).to_numpy()
        et = frame["event_type"].astype(str).to_numpy()
        uniq_types = sorted(set(et.tolist()))
        K = len(uniq_types)
        Tn = len(frame)
        type_index = {t: j for j, t in enumerate(uniq_types)}
        oh = np.zeros((Tn, K), dtype=np.int32)
        for i, t in enumerate(et):
            oh[i, type_index[t]] = 1
        csum = np.cumsum(oh, axis=0)
        idx_ns = idx.astype("datetime64[ns]").astype("int64")
        per_win_blocks: List[np.ndarray] = []
        for w in count_windows:
            w_ns = np.int64(w.value)
            left_ns = idx_ns - w_ns
            j0 = np.searchsorted(idx_ns, left_ns, side="right")
            block = np.empty((Tn, K + 1), dtype=np.float32)
            block[:, :K] = csum
            mask = j0 > 0
            if np.any(mask):
                block[mask, :K] = csum[mask] - csum[j0[mask] - 1]
            block[:, K] = (np.arange(Tn) - j0 + 1).astype(np.float32)
            per_win_blocks.append(block)
        counts_mat = np.concatenate(per_win_blocks, axis=1) if per_win_blocks else counts_mat

        taus = np.array(
            list(ewma_halflives) if ewma_halflives is not None else [1.0, 5.0], dtype=float
        )
        if delta_t.size == Tn and Tn > 0:
            ew_blocks = []
            decay_factors = np.exp(-delta_t[:, None] / taus[None, :])
            for j in range(K):
                x = oh[:, j].astype(np.float32)
                ew = np.zeros((Tn, len(taus)), dtype=np.float32)
                for t in range(Tn):
                    if t == 0:
                        ew[t] = x[t]
                    else:
                        ew[t] = ew[t - 1] * decay_factors[t] + x[t]
                ew_blocks.append(ew)
            x = oh.sum(axis=1).astype(np.float32)
            ew_total = np.zeros((Tn, len(taus)), dtype=np.float32)
            for t in range(Tn):
                if t == 0:
                    ew_total[t] = x[t]
                else:
                    ew_total[t] = ew_total[t - 1] * decay_factors[t] + x[t]
            ew_full = np.concatenate(ew_blocks + [ew_total], axis=1)
            counts_mat = np.concatenate([counts_mat, ew_full], axis=1)

    time_utc = frame["timestamp"].dt.tz_convert("UTC")
    sod = (
        time_utc.dt.hour.to_numpy() * 3600
        + time_utc.dt.minute.to_numpy() * 60
        + time_utc.dt.second.to_numpy()
    ).astype(np.float32)
    angle = 2.0 * np.pi * sod / 86400.0
    tod_sin = np.sin(angle).astype(np.float32)
    tod_cos = np.cos(angle).astype(np.float32)
    tod_progress = (sod / 86400.0).astype(np.float32)
    dow = time_utc.dt.dayofweek.to_numpy().astype(np.int32)
    dow_angle = 2.0 * np.pi * (dow.astype(np.float32) / 7.0)
    dow_sin = np.sin(dow_angle).astype(np.float32)
    dow_cos = np.cos(dow_angle).astype(np.float32)

    def _queue_age_series(
        sz: np.ndarray, pr: np.ndarray, *, reset_mask: np.ndarray | None = None
    ) -> np.ndarray:
        if _tda_native is not None:
            try:
                return np.asarray(
                    _tda_native.queue_age_series(
                        sz.astype(float), pr.astype(float), delta_t.astype(np.float32)
                    ),
                    dtype=np.float32,
                )
            except Exception:
                pass
        age = np.zeros((len(frame),), dtype=np.float32)
        for i in range(1, len(frame)):
            changed = (sz[i] != sz[i - 1]) or (
                not np.isfinite(pr[i]) or not np.isfinite(pr[i - 1]) or pr[i] != pr[i - 1]
            )
            hard_reset = bool(reset_mask[i]) if reset_mask is not None else False
            age[i] = 0.0 if (changed or hard_reset) else age[i - 1] + float(delta_t[i])
        return age

    qage_b = np.zeros((len(frame), levels), dtype=np.float32)
    qage_a = np.zeros((len(frame), levels), dtype=np.float32)
    ts_ns = frame["timestamp"].astype("int64").to_numpy()

    use_rust_levels = _tda_native is not None and hasattr(_tda_native, "queue_age_levels")
    rust_b_fallback = None
    rust_a_fallback = None
    if use_rust_levels:
        try:
            bid_price_mat = np.stack([p.to_numpy(dtype=float) for p in bid_prices], axis=1)
            ask_price_mat = np.stack([p.to_numpy(dtype=float) for p in ask_prices], axis=1)
            rust_b_fallback = np.asarray(
                _tda_native.queue_age_levels(
                    bid_sizes_mat.astype(float),
                    bid_price_mat.astype(float),
                    delta_t.astype(np.float32),
                    None,
                ),
                dtype=np.float32,
            ).reshape(len(frame), levels)
            rust_a_fallback = np.asarray(
                _tda_native.queue_age_levels(
                    ask_sizes_mat.astype(float),
                    ask_price_mat.astype(float),
                    delta_t.astype(np.float32),
                    None,
                ),
                dtype=np.float32,
            ).reshape(len(frame), levels)
        except Exception:
            rust_b_fallback = None
            rust_a_fallback = None
    has_meta = "event_type" in frame.columns
    side_col = frame.get("side", pd.Series([None] * len(frame)))
    level_col = (
        pd.to_numeric(frame.get("level", pd.Series([-1] * len(frame))), errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    etype_col = frame.get("event_type", pd.Series([None] * len(frame))).astype(str)

    for l in range(1, levels + 1):
        bs = frame.get(
            f"bid_size_{l}", pd.Series(index=frame.index, dtype=float).fillna(0.0)
        ).to_numpy()
        bp = frame.get(f"bid_price_{l}", pd.Series(index=frame.index, dtype=float)).to_numpy()
        as_ = frame.get(
            f"ask_size_{l}", pd.Series(index=frame.index, dtype=float).fillna(0.0)
        ).to_numpy()
        ap = frame.get(f"ask_price_{l}", pd.Series(index=frame.index, dtype=float)).to_numpy()
        reset_b = np.zeros((len(frame),), dtype=bool)
        reset_a = np.zeros((len(frame),), dtype=bool)
        if has_meta:
            mask_lo_cx = etype_col.isin(["LO+", "LO-", "CX+", "CX-"])
            mask_lvl = level_col.values == l
            if side_col is not None:
                sb = (side_col.astype(str).str.lower() == "bid").to_numpy()
                sa = (side_col.astype(str).str.lower() == "ask").to_numpy()
                reset_b |= mask_lo_cx.to_numpy() & mask_lvl & sb
                reset_a |= mask_lo_cx.to_numpy() & mask_lvl & sa
            else:
                m = mask_lo_cx.to_numpy() & mask_lvl
                reset_b |= m
                reset_a |= m
            is_mo_buy = (etype_col == "MO+").to_numpy()
            is_mo_sell = (etype_col == "MO-").to_numpy()
            if l == 1:
                reset_a |= is_mo_buy  # buy MOs hit asks
                reset_b |= is_mo_sell  # sell MOs hit bids

        if rust_b_fallback is not None:
            age_b_fallback = rust_b_fallback[:, l - 1]
        else:
            age_b_fallback = _queue_age_series(bs, bp, reset_mask=reset_b)
        if rust_a_fallback is not None:
            age_a_fallback = rust_a_fallback[:, l - 1]
        else:
            age_a_fallback = _queue_age_series(as_, ap, reset_mask=reset_a)

        lb = frame.get(f"last_update_bid_{l}")
        la = frame.get(f"last_update_ask_{l}")
        if lb is not None:
            try:
                lb_ns = pd.to_datetime(lb, utc=True, errors="coerce").astype("int64").to_numpy()
                age_b = np.clip((ts_ns - lb_ns) / 1e9, a_min=0.0, a_max=None).astype(np.float32)
                if np.isnan(age_b).any():
                    mask = ~np.isfinite(age_b)
                    age_b[mask] = age_b_fallback[mask]
            except Exception:
                age_b = age_b_fallback
        else:
            age_b = age_b_fallback
        if la is not None:
            try:
                la_ns = pd.to_datetime(la, utc=True, errors="coerce").astype("int64").to_numpy()
                age_a = np.clip((ts_ns - la_ns) / 1e9, a_min=0.0, a_max=None).astype(np.float32)
                if np.isnan(age_a).any():
                    mask = ~np.isfinite(age_a)
                    age_a[mask] = age_a_fallback[mask]
            except Exception:
                age_a = age_a_fallback
        else:
            age_a = age_a_fallback

        qage_b[:, l - 1] = age_b
        qage_a[:, l - 1] = age_a
    queue_age_b1 = qage_b[:, 0]
    queue_age_a1 = qage_a[:, 0]

    aux = {
        "spreads": spreads,
        "mid": mid,
        "imbalance@k": imbalance_k.astype(np.float32),
        "cum_depth_b": cum_b_sel.astype(np.float32),
        "cum_depth_a": cum_a_sel.astype(np.float32),
        "ret": ret_mat,
        "delta_t": delta_t,
        "evt_counts": counts_mat.astype(np.float32),
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
        "tod_progress": tod_progress,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "queue_age_b": qage_b,
        "queue_age_a": qage_a,
        "queue_age_b1": queue_age_b1,
        "queue_age_a1": queue_age_a1,
    }

    return base_features.astype(np.float32), aux


__all__ = ["build_lob_feature_matrix"]
