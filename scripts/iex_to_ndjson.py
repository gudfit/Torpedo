#!/usr/bin/env python3
from __future__ import annotations

"""
Convert IEX DEEP/TOPS-like JSON lines to canonical NDJSON.

This is a best-effort mapping for research without paid feeds. Expected input lines are raw IEX
messages as JSON objects. We map:
- book updates (bids/asks arrays) to LO+/LO- per level update
- trade messages to MO+
"""

import argparse
import json
from pathlib import Path


def convert_lines(lines: list[str], default_symbol: str | None = None, venue: str = "IEX") -> list[str]:
    out: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        msg_type = obj.get("type") or obj.get("msgType")
        ts = obj.get("timestamp") or obj.get("ts") or obj.get("T")
        sym = obj.get("symbol") or obj.get("S") or default_symbol
        if msg_type in {"book", "Book"}:
            # Expect bids/asks arrays: [[price,size], ...]
            for side_key, flag in (("bids", "B"), ("asks", "S")):
                arr = obj.get(side_key) or []
                for lvl in arr:
                    try:
                        price = float(lvl[0]); size = float(lvl[1])
                    except Exception:
                        continue
                    rec = {
                        "timestamp": ts,
                        "event_type": "LO+" if flag == "B" else "LO-",
                        "price": price,
                        "size": size,
                        "level": None,
                        "side": flag,
                        "symbol": sym,
                        "venue": venue,
                    }
                    out.append(json.dumps(rec))
        elif msg_type in {"trade", "Trade"}:
            try:
                price = float(obj.get("price"))
                size = float(obj.get("size"))
            except Exception:
                continue
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
        else:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert IEX DEEP JSON lines to canonical NDJSON")
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

