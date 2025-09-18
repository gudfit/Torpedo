#!/usr/bin/env python3
from __future__ import annotations

"""
Binance L2/trade WebSocket streamer to JSONL.

Requires `websockets` (or `websocket-client`) package installed.
Writes one JSON object per line (raw Binance payload) for later conversion via
`scripts/binance_to_ndjson.py`.
"""

import argparse
import asyncio
import json
from pathlib import Path


async def stream(symbol: str, output: Path, streams: list[str], rotate_seconds: int | None = None) -> None:
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
    try:
        import websockets  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise SystemExit("Please install 'websockets' to use this script") from e

    output.parent.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(url, max_size=10_000_000) as ws:  # 10MB messages
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
                # Flatten combined stream wrapper {"stream":..., "data":{...}}
                try:
                    obj = json.loads(msg)
                    data = obj.get("data", obj)
                except Exception:
                    data = {"raw": msg}
                f.write(json.dumps(data) + "\n")
                f.flush()
        finally:
            if f:
                f.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream Binance L2/trade WS to JSONL")
    ap.add_argument("--symbol", type=str, required=True, help="e.g., BTCUSDT")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    ap.add_argument(
        "--channels",
        type=str,
        default="aggTrade,bookTicker",  # alternatively: depth@100ms etc.
        help="Comma-separated channels (e.g., aggTrade,bookTicker,depth@100ms)",
    )
    ap.add_argument("--rotate-seconds", type=int, default=None, help="Rotate output files every N seconds")
    args = ap.parse_args()

    syml = args.symbol.lower()
    chs = [f"{syml}@{c}" if "@" not in c else c for c in args.channels.split(",")]
    asyncio.run(stream(args.symbol, args.output, chs, rotate_seconds=args.rotate_seconds))


if __name__ == "__main__":
    main()
