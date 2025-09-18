#!/usr/bin/env python3
from __future__ import annotations

"""
Convert FI-2010 style LOB snapshots to canonical NDJSON.

Expected input: CSV with at least best-level columns:
  - ask_price_1, ask_size_1, bid_price_1, bid_size_1
Optionally multiple levels ask_price_{l}, ask_size_{l}, bid_price_{l}, bid_size_{l}.

Output: NDJSON with one or two LO events per row for best bid/ask, using synthetic timestamps
(incrementing by 1ms by default). This provides a free, reproducible path to evaluate the
pipeline without paid equity feeds, acknowledging that FI-2010 is not an event stream.
"""

import argparse
import json
from pathlib import Path


def convert(csv_path: Path, out_path: Path, *, symbol: str, dt_ns: int = 1_000_000) -> int:
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Missing required columns: {sorted(required - set(df.columns))}")
    t0 = 0
    lines: list[str] = []
    for _, row in df.iterrows():
        ts = t0
        t0 += dt_ns
        try:
            ap = float(row["ask_price_1"]); asz = float(row["ask_size_1"])  # noqa: E741
            bp = float(row["bid_price_1"]); bsz = float(row["bid_size_1"])
        except Exception:
            continue
        rec_b = {
            "timestamp": ts,
            "event_type": "LO+",
            "price": bp,
            "size": bsz,
            "level": 1,
            "side": "B",
            "symbol": symbol,
            "venue": "FI2010",
        }
        rec_a = {
            "timestamp": ts,
            "event_type": "LO-",
            "price": ap,
            "size": asz,
            "level": 1,
            "side": "S",
            "symbol": symbol,
            "venue": "FI2010",
        }
        lines.append(json.dumps(rec_b))
        lines.append(json.dumps(rec_a))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert FI-2010 LOB snapshots to canonical NDJSON")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--symbol", type=str, required=True)
    ap.add_argument("--dt-ns", type=int, default=1_000_000, help="Synthetic timestep in ns")
    args = ap.parse_args()
    n = convert(args.input, args.output, symbol=args.symbol, dt_ns=args.dt_ns)
    print(f"Wrote {n} lines to {args.output}")


if __name__ == "__main__":
    main()

