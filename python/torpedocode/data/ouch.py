"""Minimal OUCH parser (subset) to canonical events.

Implements a compact, test-friendly binary format for a subset of OUCH-like
messages. Each message is encoded as:

    - 8 bytes: timestamp in ns (uint64 LE)
    - 1 byte:  message type (ASCII)
    - payload per type, little-endian

Supported message types:

  'O' (Enter Order):
      - 8 bytes: client_order_id (uint64)
      - 1 byte:  side ('B' or 'S')
      - 4 bytes: shares (uint32)
      - 8 bytes: price in 1e-4 units (uint64)

  'U' (Replace Order):
      - 8 bytes: orig_client_order_id (uint64)
      - 8 bytes: new_client_order_id (uint64)
      - 4 bytes: new_shares (uint32)
      - 8 bytes: new_price in 1e-4 units (uint64)

  'X' (Cancel Order):
      - 8 bytes: client_order_id (uint64)
      - 4 bytes: canceled_shares (uint32)

  'E' (Order Executed):
      - 8 bytes: client_order_id (uint64)
      - 4 bytes: executed_shares (uint32)

This is intentionally minimal to enable deterministic tests and to map to
the canonical schema in this project.
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
class OUCHParseConfig:
    tick_size: Optional[float] = None
    price_scale: float = 1e-4
    symbol: Optional[str] = None
    venue: Optional[str] = "OUCH"


def _read_exact(b: io.BufferedReader, n: int) -> bytes:
    data = b.read(n)
    if data is None or len(data) != n:
        raise EOFError
    return data


def parse_ouch_minimal(path: Path | str, *, cfg: OUCHParseConfig) -> pd.DataFrame:
    path = Path(path)
    recs: List[dict] = []
    with path.open("rb") as f:
        while True:
            try:
                ts_ns = struct.unpack("<Q", _read_exact(f, 8))[0]
                mtype = _read_exact(f, 1)
            except EOFError:
                break
            if not mtype:
                break
            m = mtype.decode("ascii", errors="ignore")
            if m == "O":
                _cid = struct.unpack("<Q", _read_exact(f, 8))[0]
                side = _read_exact(f, 1).decode("ascii")
                shares = struct.unpack("<I", _read_exact(f, 4))[0]
                price_i = struct.unpack("<Q", _read_exact(f, 8))[0]
                price = float(price_i) * cfg.price_scale
                recs.append(
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
            elif m == "U":
                _orig = struct.unpack("<Q", _read_exact(f, 8))[0]
                _new = struct.unpack("<Q", _read_exact(f, 8))[0]
                new_shares = struct.unpack("<I", _read_exact(f, 4))[0]
                new_price_i = struct.unpack("<Q", _read_exact(f, 8))[0]
                price = float(new_price_i) * cfg.price_scale
                recs.append(
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
            elif m == "X":
                _cid = struct.unpack("<Q", _read_exact(f, 8))[0]
                canceled = struct.unpack("<I", _read_exact(f, 4))[0]
                recs.append(
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
            elif m == "E":
                _cid = struct.unpack("<Q", _read_exact(f, 8))[0]
                executed = struct.unpack("<I", _read_exact(f, 4))[0]
                recs.append(
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
            elif m == "D":  # Delete order (treat as cancel)
                _cid = struct.unpack("<Q", _read_exact(f, 8))[0]
                recs.append(
                    {
                        "timestamp": pd.to_datetime(ts_ns, unit="ns", utc=True),
                        "event_type": "CX+",
                        "size": 0.0,
                        "price": np.nan,
                        "level": np.nan,
                        "side": None,
                        "symbol": cfg.symbol,
                        "venue": cfg.venue,
                    }
                )
            else:
                break

    if not recs:
        return pd.DataFrame(
            columns=["timestamp", "event_type", "size", "price", "level", "side", "symbol", "venue"]
        )
    df = pd.DataFrame.from_records(recs)
    if cfg.tick_size and cfg.tick_size > 0:
        df["price"] = (np.round(df["price"].astype(float) / cfg.tick_size) * cfg.tick_size).astype(
            float
        )
    return df


__all__ = ["OUCHParseConfig", "parse_ouch_minimal"]
