"""Data preprocessing and harmonisation utilities.

Implements canonical NDJSON harmonisation, session filtering (auctions/halts),
instability label construction for clock/event horizons, and corporate actions
helpers used by the ingest and panel CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import numpy as np
import pandas as pd


# ------------------------
# Canonical harmonisation
# ------------------------


@dataclass(slots=True)
class HarmoniseConfig:
    time_zone: str = "UTC"
    drop_auctions: bool = True
    session_time_zone: str = "America/New_York"
    tick_size: Optional[float] = None
    price_scale: Optional[float] = None
    symbol: Optional[str] = None
    # Quality checks
    quality_checks: bool = True
    min_size: float = 0.0
    min_price: float = 0.0
    drop_duplicates: bool = True
    enforce_monotonic_timestamps: bool = True
    drop_backwards_timestamps: bool = True


def _to_utc(ts: pd.Series, tz: str) -> pd.Series:
    """Robust UTC conversion for mixed timestamp types.

    Accepts ISO8601 strings or numeric epoch times. For numeric inputs, auto-detects
    likely unit by magnitude (ns/us/ms/s) and converts to UTC.
    """
    ser = pd.Series(ts)
    # Fast path for already-datetime-like
    t = pd.to_datetime(ser, utc=True, errors="coerce")
    # If a good portion converted, use it
    if t.notna().mean() > 0.5:
        return t
    # Try numeric epoch with unit inference
    v = pd.to_numeric(ser, errors="coerce")
    vals = v[~v.isna()].astype(float).to_numpy()
    if vals.size == 0:
        return t  # keep whatever we could parse
    med = float(np.nanmedian(np.abs(vals)))
    # Heuristic thresholds (epoch around 1.7e9 s, 1.7e12 ms, 1.7e15 us, 1.7e18 ns)
    if med > 1e17:
        unit = "ns"
    elif med > 1e14:
        unit = "us"
    elif med > 1e11:
        unit = "ms"
    elif med > 1e8:
        unit = "s"
    else:
        unit = None
    if unit is not None:
        try:
            return pd.to_datetime(v, unit=unit, utc=True, errors="coerce")
        except Exception:
            pass
    return t


def _normalise_price(v: pd.Series, *, tick_size: Optional[float], price_scale: Optional[float]) -> pd.Series:
    x = pd.to_numeric(v, errors="coerce")
    if price_scale is not None and price_scale != 0:
        x = x * float(price_scale)
    if tick_size is not None and tick_size > 0:
        x = np.round(np.asarray(x, dtype=float) / float(tick_size)) * float(tick_size)
        x = pd.Series(x)
    return x


def _canonicalize_types(s: pd.Series) -> pd.Series:
    # Accept already-canonical codes; map common raw tokens to canonical ones
    m = s.astype(str).str.upper()
    mapping = {
        "TRADE": "MO+",
        "BUY": "MO+",
        "SELL": "MO-",
        "ADD": "LO+",
        "NEW": "LO+",
        "CANCEL": "CX-",
        "DELETE": "CX-",
        "EXECUTE": "MO+",
    }
    mc = m.map(mapping).fillna(m)
    return mc


def harmonise_ndjson(path: Path | str, *, cfg: HarmoniseConfig) -> pd.DataFrame:
    """Read an NDJSON/JSONL file and return the canonical event table.

    Expected fields per line: timestamp, event_type, price, size, optional level, side,
    symbol, venue. Unknown fields are preserved if noted below.
    """
    p = Path(path)
    rows: List[dict] = []
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["timestamp", "event_type", "price", "size"])  # minimal schema
    df = pd.DataFrame(rows)

    # Timestamp → UTC
    df["timestamp"] = _to_utc(df.get("timestamp"), cfg.time_zone)

    # Event types → canonical
    df["event_type"] = _canonicalize_types(df.get("event_type", pd.Series([], dtype=str)))

    # Price/size
    if "price" in df.columns:
        df["price"] = _normalise_price(
            df["price"], tick_size=cfg.tick_size, price_scale=cfg.price_scale
        )
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")

    # Optional fields
    if cfg.symbol is not None and "symbol" not in df.columns:
        df["symbol"] = cfg.symbol

    # Optionally drop auctions/halts via helper (caller typically handles this)
    if bool(cfg.drop_auctions):
        try:
            df = drop_auction_and_halt_intervals(
                df, session_start="09:30", session_end="16:00", local_tz=cfg.session_time_zone
            )
        except Exception:
            pass

    # Quality checks: drop corrupt/inconsistent events
    if bool(getattr(cfg, "quality_checks", True)) and not df.empty:
        try:
            # Coerce timestamp and drop NaT
            tsq = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
            df = df.loc[tsq.notna()].copy()
            df["timestamp"] = tsq.loc[tsq.notna()]
            # Optionally enforce non-decreasing timestamps by dropping backwards rows
            if bool(getattr(cfg, "enforce_monotonic_timestamps", True)) and not df.empty:
                df = df.sort_values("timestamp").reset_index(drop=True)
                if bool(getattr(cfg, "drop_backwards_timestamps", True)):
                    # After sort, sequence is non-decreasing by construction; no further action needed
                    pass
            # Ensure event_type exists
            if "event_type" not in df.columns:
                df["event_type"] = "UNK"
            # Numeric sanitization for price/size
            if "price" in df.columns:
                pr = pd.to_numeric(df["price"], errors="coerce")
                if cfg.min_price is not None:
                    pr = pr.where(pr >= float(cfg.min_price))
                df = df.loc[pr.notna()].copy()
                df["price"] = pr.loc[pr.notna()]
            if "size" in df.columns:
                sz = pd.to_numeric(df["size"], errors="coerce")
                if cfg.min_size is not None:
                    sz = sz.where(sz >= float(cfg.min_size))
                df = df.loc[sz.notna()].copy()
                df["size"] = sz.loc[sz.notna()]
            # Optional duplicate drop on common keys
            if bool(getattr(cfg, "drop_duplicates", True)):
                subset = [c for c in ("timestamp", "event_type", "price", "size") if c in df.columns]
                if subset:
                    df = df.drop_duplicates(subset=subset)
        except Exception:
            pass

    return df.reset_index(drop=True)


# ------------------------
# Auctions/Halts filtering
# ------------------------


def drop_auction_and_halt_intervals(
    frame: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    local_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Drop rows outside regular hours and around explicit/implicit halts.

    - Keeps only rows between [session_start, session_end] in the specified local time zone.
    - Drops rows where trading_state indicates a halt (case-insensitive 'H').
    - Drops the edges around long intra-day gaps (>120s): the row before and after the gap.
    """
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Session filter in local time zone
    try:
        local = ts.dt.tz_convert(local_tz)
    except Exception:
        local = ts  # fallback to UTC windowing
    start_h, start_m = [int(x) for x in session_start.split(":")]
    end_h, end_m = [int(x) for x in session_end.split(":")]
    within = (
        (local.dt.hour > start_h) | ((local.dt.hour == start_h) & (local.dt.minute >= start_m))
    ) & ((local.dt.hour < end_h) | ((local.dt.hour == end_h) & (local.dt.minute <= end_m)))
    df = df.loc[within].copy()

    # Explicit halt flags
    if "trading_state" in df.columns:
        mask_halt = df["trading_state"].astype(str).str.upper().str.contains("H", na=False)
        df = df.loc[~mask_halt].copy()

    # Implicit halts: long gaps within same day
    ts2 = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").reset_index(drop=True)
    if len(ts2) >= 3:
        # Use astype instead of view to avoid deprecation warnings on Pandas >=2.0
        tns = ts2.astype("int64").to_numpy()
        delta = (tns[1:] - tns[:-1]) / 1e9
        big = np.where(delta > 120.0)[0]  # indices where gap between i and i+1 is large
        drop_idx = set()
        for i in big:
            # Drop i (before) and i+1 (after) if same local day
            try:
                d0 = ts2.iloc[i].tz_convert(local_tz).date()
                d1 = ts2.iloc[i + 1].tz_convert(local_tz).date()
            except Exception:
                d0 = ts2.iloc[i].date(); d1 = ts2.iloc[i + 1].date()
            if d0 == d1:
                drop_idx.add(i)
                drop_idx.add(i + 1)
        if drop_idx:
            keep = [k for k in range(len(ts2)) if k not in drop_idx]
            df = df.iloc[keep].copy()
    return df.reset_index(drop=True)


# ------------------------
# Labels
# ------------------------


def label_instability(
    mid: pd.Series,
    timestamps: pd.Series,
    *,
    horizons_s: Iterable[int] | None = None,
    horizons_events: Iterable[int] | None = None,
    threshold_eta: float = 0.0,
) -> Dict[str, pd.Series]:
    """Build binary labels for clock- and event-horizon instability.

    For each time t, label 1 if |mid(t+h) - mid(t)| > eta, otherwise 0.
    """
    y: Dict[str, pd.Series] = {}
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    m = pd.to_numeric(pd.Series(mid), errors="coerce").to_numpy()
    T = len(ts)
    # Clock-time horizons
    if horizons_s:
        t_ns = ts.astype("int64").to_numpy()
        for s in horizons_s:
            dt_ns = int(s) * 1_000_000_000
            idx = np.searchsorted(t_ns, t_ns + dt_ns, side="left")
            idx = np.minimum(idx, T - 1)
            diff = np.abs(m[idx] - m)
            y[f"instability_s_{int(s)}"] = pd.Series((diff > float(threshold_eta)).astype(int))
    # Event-time horizons
    if horizons_events:
        for k in horizons_events:
            j = np.arange(T) + int(k)
            j = np.clip(j, 0, T - 1)
            diff = np.abs(m[j] - m)
            y[f"instability_e_{int(k)}"] = pd.Series((diff > float(threshold_eta)).astype(int))
    return y


# ------------------------
# Corporate actions helpers
# ------------------------


def _load_corporate_actions_csv(
    path: Path | str,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    factor_col: str = "adj_factor",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={date_col: "date", symbol_col: "symbol", factor_col: "adj_factor"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
    return df[["date", "symbol", "adj_factor"]]


def adjust_corporate_actions(
    frame: pd.DataFrame, ca_df: pd.DataFrame, *, mode: str = "divide"
) -> pd.DataFrame:
    if frame.empty or ca_df is None or ca_df.empty:
        return frame
    df = frame.copy()
    ts_date = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.date
    sym = df.get("symbol", pd.Series([None] * len(df)))
    ca = ca_df.copy()
    # Merge by symbol (if present) and date
    key = ["date", "symbol"] if "symbol" in ca.columns and sym.notna().any() else ["date"]
    left = pd.DataFrame({"date": ts_date})
    if "symbol" in key:
        left["symbol"] = sym
    merged = left.merge(ca, how="left", on=key)
    factor = merged.get("adj_factor").fillna(1.0).to_numpy()
    price_cols = [c for c in df.columns if c == "price" or c.startswith("bid_price_") or c.startswith("ask_price_")]
    for c in price_cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if mode == "multiply":
            adj = vals * factor
        else:
            adj = vals / np.where(factor == 0, 1.0, factor)
        df[c] = adj
    return df


def round_prices_to_tick(frame: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    df = frame.copy()
    price_cols = [c for c in df.columns if c == "price" or c.startswith("bid_price_") or c.startswith("ask_price_")]
    ts = float(tick_size)
    for c in price_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy()
        df[c] = (np.round(x / ts) * ts).astype(float)
    return df


# ------------------------
# Liquidity panel helpers
# ------------------------


def compute_liquidity_panel(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    if "median_daily_notional" in df.columns:
        try:
            q = pd.qcut(pd.to_numeric(df["median_daily_notional"], errors="coerce"), 10, labels=False, duplicates="drop")
            df["liq_decile"] = (q.fillna(0).astype(int) + 1).clip(lower=1)
        except Exception:
            df["liq_decile"] = 1
    else:
        df["liq_decile"] = 1
    return df


def match_instruments_across_markets(panel: pd.DataFrame, by: Iterable[str] | None = None) -> pd.DataFrame:
    by = list(by) if by else ["liq_decile", "tick_size"]
    df = panel.copy()
    df["match_group"] = df.groupby(by, dropna=False).ngroup()
    return df


__all__ = [
    "HarmoniseConfig",
    "harmonise_ndjson",
    "drop_auction_and_halt_intervals",
    "label_instability",
    "_load_corporate_actions_csv",
    "adjust_corporate_actions",
    "round_prices_to_tick",
    "compute_liquidity_panel",
    "match_instruments_across_markets",
]
