#!/usr/bin/env python3
from __future__ import annotations

"""
Coinbase L2/trade WebSocket streamer to JSONL.

Requires `websockets` package installed. Writes raw Coinbase messages (one per line)
for later conversion via `scripts/coinbase_to_ndjson.py`.
"""

import argparse
import asyncio
import json
from pathlib import Path


async def stream(product: str, output: Path, channels: list[str], rotate_seconds: int | None = None) -> None:
    url = "wss://ws-feed.exchange.coinbase.com"
    try:
        import websockets  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise SystemExit("Please install 'websockets' to use this script") from e

    output.parent.mkdir(parents=True, exist_ok=True)
    sub = {
        "type": "subscribe",
        "product_ids": [product],
        "channels": channels,
    }
    async with websockets.connect(url, max_size=10_000_000) as ws:
        await ws.send(json.dumps(sub))
        start_ts = None
        f = None
        try:
            while True:
                msg = await ws.recv()
                now = asyncio.get_event_loop().time()
                if rotate_seconds and (start_ts is None or (now - start_ts) >= rotate_seconds or f is None):
                    if f:
                        f.close()
                    start_ts = now
                    rotated = output.with_name(output.stem + f"_{int(now)}" + output.suffix)
                    rotated.parent.mkdir(parents=True, exist_ok=True)
                    f = rotated.open("a")
                if f is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    f = output.open("a")
                    start_ts = now
                f.write(msg + "\n")
                f.flush()
        finally:
            if f:
                f.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream Coinbase WS to JSONL")
    ap.add_argument("--product", type=str, required=True, help="e.g., BTC-USD")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--channels",
        type=str,
        default="matches,l2update",
        help="Comma-separated channels (matches,l2update,heartbeat)",
    )
    ap.add_argument("--rotate-seconds", type=int, default=None, help="Rotate output files every N seconds")
    args = ap.parse_args()
    chs = args.channels.split(",")
    asyncio.run(stream(args.product, args.output, chs, rotate_seconds=args.rotate_seconds))


if __name__ == "__main__":
    main()
