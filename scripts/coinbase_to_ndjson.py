#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def convert_lines(lines: Iterable[str], default_symbol: str | None = None, venue: str = "COINBASE") -> list[str]:
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        t = obj.get("type")
        # Coinbase channels examples:
        # - match: {"type":"match","time":"2025-01-01T00:00:00Z","product_id":"BTC-USD","price":"29000.1","size":"0.123"}
        # - l2update: {"type":"l2update","time":"...","product_id":"BTC-USD","changes":[["buy","28999.9","1.2"],["sell","29000.3","0.8"]]}
        ts = obj.get("time") or obj.get("E") or obj.get("timestamp")
        sym = obj.get("product_id") or obj.get("symbol") or default_symbol
        if t == "match":
            price = float(obj.get("price") or 0.0)
            size = float(obj.get("size") or 0.0)
            rec = {
                "timestamp": ts,
                "event_type": "MO+",
                "price": price,
                "size": size,
                "level": None,
                "side": None,
                "symbol": sym,
                "venue": venue,
            }
            out.append(json.dumps(rec))
        elif t == "l2update":
            changes = obj.get("changes") or []
            for side, price_s, size_s in changes:
                try:
                    price = float(price_s)
                    size = float(size_s)
                except Exception:
                    continue
                rec = {
                    "timestamp": ts,
                    "event_type": "LO+" if side == "buy" else "LO-",
                    "price": price,
                    "size": size,
                    "level": None,
                    "side": "B" if side == "buy" else "S",
                    "symbol": sym,
                    "venue": venue,
                }
                out.append(json.dumps(rec))
        else:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Coinbase JSON lines to canonical NDJSON")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--symbol", type=str, default=None)
    args = ap.parse_args()

    lines = args.input.read_text().splitlines()
    out_lines = convert_lines(lines, default_symbol=args.symbol)
    args.output.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"Wrote {len(out_lines)} lines to {args.output}")


if __name__ == "__main__":
    main()

