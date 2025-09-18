#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def convert_lines(lines: Iterable[str], default_symbol: str | None = None, venue: str = "BINANCE") -> list[str]:
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        # Binance WebSocket examples:
        # - AggTrade: {"e":"aggTrade","E":1690000000000,"s":"BTCUSDT","p":"29000.1","q":"0.123"}
        # - Book Ticker: {"e":"bookTicker","E":1690000000000,"s":"BTCUSDT","b":"28999.9","B":"1.2","a":"29000.3","A":"0.8"}
        # - Depth Update: {"e":"depthUpdate","E":1690000000000,"s":"BTCUSDT","b":[[price,qty],...],"a":[[price,qty],...]}
        etype = (obj.get("e") or obj.get("event"))
        ts = obj.get("E") or obj.get("T") or obj.get("timestamp")
        sym = obj.get("s") or obj.get("symbol") or default_symbol
        if etype == "aggTrade" or etype == "trade":
            price = float(obj.get("p") or obj.get("price") or 0.0)
            qty = float(obj.get("q") or obj.get("qty") or 0.0)
            rec = {
                "timestamp": ts,
                "event_type": "MO+",
                "price": price,
                "size": qty,
                "level": None,
                "side": None,
                "symbol": sym,
                "venue": venue,
            }
            out.append(json.dumps(rec))
        elif etype == "bookTicker":
            # Emit two limit updates for best bid/ask levels
            try:
                bid_p = float(obj.get("b")); bid_q = float(obj.get("B"))
                ask_p = float(obj.get("a")); ask_q = float(obj.get("A"))
            except Exception:
                continue
            rec_b = {
                "timestamp": ts,
                "event_type": "LO+",
                "price": bid_p,
                "size": bid_q,
                "level": 1,
                "side": "B",
                "symbol": sym,
                "venue": venue,
            }
            rec_a = {
                "timestamp": ts,
                "event_type": "LO-",
                "price": ask_p,
                "size": ask_q,
                "level": 1,
                "side": "S",
                "symbol": sym,
                "venue": venue,
            }
            out.append(json.dumps(rec_b))
            out.append(json.dumps(rec_a))
        elif etype == "depthUpdate":
            # Emits multiple price level updates
            for side_key, side_flag in (("b", "B"), ("a", "S")):
                arr = obj.get(side_key) or []
                for elem in arr:
                    try:
                        price = float(elem[0])
                        qty = float(elem[1])
                    except Exception:
                        continue
                    rec = {
                        "timestamp": ts,
                        "event_type": "LO+" if side_flag == "B" else "LO-",
                        "price": price,
                        "size": qty,
                        "level": None,
                        "side": side_flag,
                        "symbol": sym,
                        "venue": venue,
                    }
                    out.append(json.dumps(rec))
        else:
            # Unknown event; ignore
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Binance JSON lines to canonical NDJSON")
    ap.add_argument("--input", type=Path, required=True, help="Path to Binance JSON lines file")
    ap.add_argument("--output", type=Path, required=True, help="Path to write canonical NDJSON")
    ap.add_argument("--symbol", type=str, default=None, help="Default symbol if missing in records")
    args = ap.parse_args()

    lines = args.input.read_text().splitlines()
    out_lines = convert_lines(lines, default_symbol=args.symbol)
    args.output.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"Wrote {len(out_lines)} lines to {args.output}")


if __name__ == "__main__":
    main()

