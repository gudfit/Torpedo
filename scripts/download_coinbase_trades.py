#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None  # type: ignore


BASE = "https://api.exchange.coinbase.com"


def _parse_iso(ts: str) -> datetime:
    try:
        # Allow Z suffix
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts[:-1]).replace(tzinfo=timezone.utc)
        d = datetime.fromisoformat(ts)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        raise SystemExit(f"Invalid ISO timestamp: {ts}")


def fetch_trades(product: str, start: datetime, end: datetime, *, limit: int = 100) -> list[dict]:
    if requests is None:
        raise SystemExit("requests not installed. pip install requests")
    url = f"{BASE}/products/{product}/trades"
    results: list[dict] = []
    after: Optional[int] = None
    # Coinbase returns most recent first; we page with before/after
    # We'll walk backward from end time using 'before' parameter (trade_id)
    # strategy: fetch batches, keep those within [start,end], stop when older than start
    before: Optional[int] = None
    # Prime by finding the first page and set 'before' high
    params = {"limit": limit}
    try:
        r0 = requests.get(url, params=params, timeout=30)
        r0.raise_for_status()
    except Exception as e:
        raise SystemExit(f"HTTP error: {e}")
    try:
        latest = r0.json()
    except Exception:
        latest = []
    if latest:
        before = int(latest[0].get("trade_id", 0)) + 1

    # Pagination loop
    while True:
        q = {"limit": limit}
        if before is not None:
            q["before"] = before
        try:
            r = requests.get(url, params=q, timeout=30)
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            print(f"[warn] request failed: {e}")
            time.sleep(1.0)
            continue
        if not batch:
            break
        # Update 'before' to the smallest id in this batch (older trades)
        before = int(batch[-1].get("trade_id", 0))
        # Keep within window
        keep = []
        for tr in batch:
            try:
                t = datetime.fromisoformat(tr.get("time").replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                continue
            if t > end:
                # Skip too new
                continue
            if t < start:
                # We crossed the start bound; stop
                batch = []
                keep = []
                break
            keep.append(tr)
        results.extend(keep)
        if not batch:
            break
        # Be kind to the API
        time.sleep(0.2)
    return results


def trades_to_canonical_ndjson(trades: list[dict], product: str) -> list[str]:
    out: list[str] = []
    for tr in trades:
        try:
            price = float(tr.get("price"))
            size = float(tr.get("size"))
            time_iso = tr.get("time")
            ts = time_iso
        except Exception:
            continue
        side = tr.get("side")
        rec = {
            "timestamp": ts,
            "event_type": "MO+" if side == "buy" else "MO-",
            "price": price,
            "size": size,
            "level": None,
            "side": None,
            "symbol": product,
            "venue": "COINBASE",
        }
        out.append(json.dumps(rec))
    return out


def main():
    ap = argparse.ArgumentParser(description="Download Coinbase public trades (REST) and write canonical NDJSON")
    ap.add_argument("--product", required=True, help="Product id, e.g., BTC-USD")
    ap.add_argument("--start", required=True, help="Start ISO (UTC), e.g., 2024-06-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="End ISO (UTC), e.g., 2024-08-31T23:59:59Z")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    start = _parse_iso(args.start)
    end = _parse_iso(args.end)
    if end <= start:
        raise SystemExit("end must be after start")

    print(f"Fetching trades for {args.product} from {start} to {end} (UTC)")
    trades = fetch_trades(args.product, start, end, limit=max(1, min(100, int(args.limit))))
    ndjson_lines = trades_to_canonical_ndjson(trades, args.product)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ndjson_lines) + ("\n" if ndjson_lines else ""))
    print(f"Wrote {len(ndjson_lines)} lines to {args.output}")


if __name__ == "__main__":
    main()

