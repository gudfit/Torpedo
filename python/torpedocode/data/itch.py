"""Minimal ITCH parser (subset) to canonical events.

This parser supports a practical subset of NASDAQ TotalView-ITCH-like
messages sufficient for research prototyping and tests. It expects a
binary stream where each message is:

    - 8 bytes: timestamp in nanoseconds since epoch (unsigned, little-endian)
    - 1 byte:  message type (ASCII)
    - payload: depends on message type

Supported message types and payloads (all little-endian):

    'A' (Add Order):
        - 8 bytes: order_id (uint64)
        - 1 byte:  side ('B' or 'S')
        - 4 bytes: shares (uint32)
        - 8 bytes: price in 1e-4 units (uint64)
        - 8 bytes: stock id/symbol hash (uint64) [ignored]

    'E' (Execute):
        - 8 bytes: order_id (uint64)
        - 4 bytes: executed_shares (uint32)

    'X' (Cancel):
        - 8 bytes: order_id (uint64)
        - 4 bytes: canceled_shares (uint32)

    'P' (Trade, non-cross):
        - 8 bytes: trade_id (uint64)
        - 1 byte:  side ('B' or 'S')
        - 4 bytes: shares (uint32)
        - 8 bytes: price in 1e-4 units (uint64)

This is not a complete ITCH 5.0 implementation; instead it provides a
stable, well-documented binary schema we can generate for tests and small
examples, and a mapping to the paper's canonical event schema.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ITCHParseConfig:
    tick_size: Optional[float] = None
    price_scale: float = 1e-4  # incoming integer price units
    symbol: Optional[str] = None
    venue: Optional[str] = "ITCH"


def _read_exact(b: io.BufferedReader, n: int) -> bytes:
    data = b.read(n)
    if data is None or len(data) != n:
        raise EOFError
    return data


def parse_itch_minimal(path: Path | str, *, cfg: ITCHParseConfig) -> pd.DataFrame:
    path = Path(path)
    records: List[dict] = []
    with path.open("rb") as f:
        bio = f
        while True:
            try:
                ts_ns = struct.unpack("<Q", _read_exact(bio, 8))[0]
                mtype = _read_exact(bio, 1)
            except EOFError:
                break
            if not mtype:
                break
            m = mtype.decode("ascii", errors="ignore")
            if m == "A":
                order_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                side = _read_exact(bio, 1).decode("ascii")
                shares = struct.unpack("<I", _read_exact(bio, 4))[0]
                price_i = struct.unpack("<Q", _read_exact(bio, 8))[0]
                _ = _read_exact(bio, 8)  # stock id/hash, ignored
                price = float(price_i) * cfg.price_scale
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": f"LO{('+' if side == 'B' else '-')}",
                        "size": float(shares),
                        "price": price,
                        "level": np.nan,
                        "side": side,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "F":  # Add Order with attribution (treat as add)
                _order_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                side = _read_exact(bio, 1).decode("ascii")
                shares = struct.unpack("<I", _read_exact(bio, 4))[0]
                price_i = struct.unpack("<Q", _read_exact(bio, 8))[0]
                _ = _read_exact(bio, 4)  # stock locate/MPID placeholder
                price = float(price_i) * cfg.price_scale
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": f"LO{('+' if side == 'B' else '-')}",
                        "size": float(shares),
                        "price": price,
                        "level": np.nan,
                        "side": side,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "E":
                _order_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                executed = struct.unpack("<I", _read_exact(bio, 4))[0]
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "MO+",
                        "size": float(executed),
                        "price": np.nan,
                        "level": np.nan,
                        "side": None,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "X":
                _order_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                canceled = struct.unpack("<I", _read_exact(bio, 4))[0]
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "CX+",
                        "size": float(canceled),
                        "price": np.nan,
                        "level": np.nan,
                        "side": None,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "P":
                _trade_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                side = _read_exact(bio, 1).decode("ascii")
                shares = struct.unpack("<I", _read_exact(bio, 4))[0]
                price_i = struct.unpack("<Q", _read_exact(bio, 8))[0]
                price = float(price_i) * cfg.price_scale
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "MO+",
                        "size": float(shares),
                        "price": price,
                        "level": np.nan,
                        "side": side,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "D":  # Delete order (cancel)
                _order_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "CX+",
                        "size": float(0),
                        "price": np.nan,
                        "level": np.nan,
                        "side": None,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "C":  # Execute with price (extended)
                _order_id = struct.unpack("<Q", _read_exact(bio, 8))[0]
                executed = struct.unpack("<I", _read_exact(bio, 4))[0]
                price_i = struct.unpack("<Q", _read_exact(bio, 8))[0]
                price = float(price_i) * cfg.price_scale
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "MO+",
                        "size": float(executed),
                        "price": price,
                        "level": np.nan,
                        "side": None,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            elif m == "U":  # Replace (simple mapping)
                _orig = struct.unpack("<Q", _read_exact(bio, 8))[0]
                _new = struct.unpack("<Q", _read_exact(bio, 8))[0]
                new_shares = struct.unpack("<I", _read_exact(bio, 4))[0]
                price_i = struct.unpack("<Q", _read_exact(bio, 8))[0]
                price = float(price_i) * cfg.price_scale
                records.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "LO+",
                        "size": float(new_shares),
                        "price": price,
                        "level": np.nan,
                        "side": None,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            else:
                # Unknown type; attempt to skip a reasonable payload or stop
                break

    if not records:
        return pd.DataFrame(
            columns=["timestamp", "event_type", "size", "price", "level", "side", "symbol", "venue"]
        )

    df = pd.DataFrame.from_records(records)
    if cfg.tick_size and cfg.tick_size > 0:
        df["price"] = (np.round(df["price"].astype(float) / cfg.tick_size) * cfg.tick_size).astype(
            float
        )
    return df


__all__ = ["ITCHParseConfig", "parse_itch_minimal"]
