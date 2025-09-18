"""LOBSTER CSV parser to canonical events.

LOBSTER provides a pair of CSV files per instrument-day and level depth L:
  - messages: time,type,order_id,size,price,direction
  - orderbook: ask_price_1, ask_size_1, ..., bid_price_L, bid_size_L

This parser merges the two streams aligned by row index and maps messages to
canonical event types used in the methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import io


@dataclass(slots=True)
class LOBSTERParseConfig:
    tick_size: Optional[float] = None
    time_zone: str = "UTC"
    symbol: Optional[str] = None
    venue: Optional[str] = "LOBSTER"
    side_aware: bool = False


def _map_lobster_type(t: pd.Series) -> pd.Series:
    mapping = {
        1: "LO+",
        2: "LO+",
        3: "CX+",
        4: "MO+",
        5: "MO+",
        6: "CX+",
        7: "MO+",
    }
    return t.map(mapping).fillna("LO+")


def parse_lobster_pair(
    message_csv: Path | str, orderbook_csv: Path | str, *, cfg: LOBSTERParseConfig
) -> pd.DataFrame:
    import os

    fast_bin = os.environ.get("LOBSTER_FAST_BIN")
    if fast_bin and Path(fast_bin).exists():
        try:
            import subprocess

            tick = str(cfg.tick_size) if (cfg.tick_size and cfg.tick_size > 0) else "0.0"
            sym = cfg.symbol or ""
            cmd = [fast_bin, str(message_csv), str(orderbook_csv), sym, tick]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            df = pd.read_csv(io.StringIO(out.decode("utf-8")))
            return df
        except Exception:
            pass

    mdf = pd.read_csv(message_csv, header=None)
    bdf = pd.read_csv(orderbook_csv, header=None)
    if mdf.shape[1] < 6:
        raise ValueError("LOBSTER messages CSV must have at least 6 columns")
    mdf = mdf.iloc[:, :6]
    mdf.columns = ["time", "lob_type", "order_id", "size", "price", "direction"]

    ts = pd.to_datetime((mdf["time"].astype(float) * 1e9).astype("int64"), unit="ns", utc=True)
    side = mdf["direction"].map({1: "B", -1: "S"}).astype(object)
    base_evt = _map_lobster_type(mdf["lob_type"]).astype(object)
    if bool(cfg.side_aware):
        # Apply sign to MO/CX when side is known; LO uses + for bid, - for ask
        evt = base_evt.copy()
        mask_mo = base_evt.str.startswith("MO") & side.notna()
        evt = evt.where(~mask_mo, "MO" + side.map({"B": "+", "S": "-"}))
        mask_cx = base_evt.str.startswith("CX") & side.notna()
        evt = evt.where(~mask_cx, "CX" + side.map({"B": "+", "S": "-"}))
        mask_lo = base_evt.str.startswith("LO") & side.notna()
        evt = evt.where(~mask_lo, "LO" + side.map({"B": "+", "S": "-"}))
    else:
        evt = base_evt
    size = pd.to_numeric(mdf["size"], errors="coerce").astype(float).clip(lower=0.0)
    price = pd.to_numeric(mdf["price"], errors="coerce").astype(float)
    if cfg.tick_size and cfg.tick_size > 0:
        price = np.round(price / cfg.tick_size) * cfg.tick_size
    L = bdf.shape[1] // 4
    cols = []
    for l in range(1, L + 1):
        cols.extend([f"ask_price_{l}", f"ask_size_{l}"])
    for l in range(1, L + 1):
        cols.extend([f"bid_price_{l}", f"bid_size_{l}"])
    bdf.columns = cols

    # Infer level by matching event price to nearest level on the same row
    level = np.full(len(mdf), np.nan, dtype=float)
    try:
        L = int(L)
        ask_price_mat = np.stack([pd.to_numeric(bdf[f"ask_price_{l}"], errors="coerce").to_numpy() for l in range(1, L + 1)], axis=1)
        bid_price_mat = np.stack([pd.to_numeric(bdf[f"bid_price_{l}"], errors="coerce").to_numpy() for l in range(1, L + 1)], axis=1)
        pr = price.to_numpy()
        side_arr = side.to_numpy()
        # Match exact equality after tick rounding; fallback: nearest match within 0.5*tick
        for i in range(len(mdf)):
            s = side_arr[i]
            pi = pr[i]
            if not np.isfinite(pi):
                continue
            if s == "S":  # ask side
                idx = np.where(np.isclose(ask_price_mat[i], pi, rtol=0.0, atol=(cfg.tick_size or 0.0) * 0.5))[0]
            else:  # B or unknown -> bid side
                idx = np.where(np.isclose(bid_price_mat[i], pi, rtol=0.0, atol=(cfg.tick_size or 0.0) * 0.5))[0]
            if idx.size > 0:
                level[i] = float(int(idx[0] + 1))
    except Exception:
        pass

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": evt,
            "price": price,
            "size": size,
            "level": level,
            "side": side,
            "symbol": cfg.symbol,
            "venue": cfg.venue,
        }
    )
    out = pd.concat([out, bdf], axis=1)
    return out


__all__ = ["LOBSTERParseConfig", "parse_lobster_pair"]
