#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Iterable

import zipfile

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None  # type: ignore


BASE = "https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip"


def _dl(url: str) -> bytes:
    if requests is None:
        raise SystemExit("requests not installed. pip install requests")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _aggtrades_zip_to_jsonl(buf: bytes, symbol: str) -> list[str]:
    zf = zipfile.ZipFile(io.BytesIO(buf))
    members = [n for n in zf.namelist() if n.endswith('.csv')]
    if not members:
        return []
    name = members[0]
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding='utf-8')
        rdr = csv.reader(text)
        out = []
        for row in rdr:
            # Binance aggTrades CSV columns: a (aggId), p (price), q (qty), f, l, T (timestamp), m (is buyer maker), M (ignore)
            try:
                price = float(row[1]); qty = float(row[2]); ts = int(row[5])
            except Exception:
                continue
            # synthesize websocket-like JSON for converter
            obj = {"e": "aggTrade", "E": ts, "s": symbol, "p": price, "q": qty}
            out.append(obj)
        return [__import__('json').dumps(o) for o in out]


def convert_months(symbol: str, year: int, months: Iterable[int]) -> list[str]:
    all_lines: list[str] = []
    for m in months:
        url = BASE.format(symbol=symbol, year=year, month=int(m))
        try:
            data = _dl(url)
        except Exception as e:
            print(f"[warn] failed to download {url}: {e}", file=sys.stderr)
            continue
        lines = _aggtrades_zip_to_jsonl(data, symbol=symbol)
        all_lines.extend(lines)
    return all_lines


def main():
    ap = argparse.ArgumentParser(description="Download Binance monthly aggTrades and convert to NDJSON")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--months", type=int, nargs="+", required=True, help="List of months, e.g., 6 7 8")
    ap.add_argument("--output", type=Path, required=True, help="NDJSON output path")
    args = ap.parse_args()

    json_lines = convert_months(args.symbol.upper(), int(args.year), [int(x) for x in args.months])
    # Use existing converter to canonical NDJSON
    from binance_to_ndjson import convert_lines  # type: ignore
    ndjson = convert_lines(json_lines, default_symbol=args.symbol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ndjson) + ("\n" if ndjson else ""))
    print(f"Wrote {len(ndjson)} canonical lines to {args.output}")


if __name__ == "__main__":
    main()

