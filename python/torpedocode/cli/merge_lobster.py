"""CLI: Merge multiple LOBSTER day folders into a single cached instrument.

Example:
  python -m torpedocode.cli.merge_lobster \
    --instrument AAPL --tick-size 0.01 --cache-root ./cache \
    --days ./2024-06-01 ./2024-06-02 ./2024-06-03
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ..config import DataConfig
from ..data.preprocessing import LOBPreprocessor


def main():
    ap = argparse.ArgumentParser(description="Merge LOBSTER day folders into one cache")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--tick-size", type=float, default=None)
    ap.add_argument("--price-scale", type=float, default=None)
    ap.add_argument("--days", type=Path, nargs="+", required=True, help="One or more day directories")
    args = ap.parse_args()

    cfg = DataConfig(raw_data_root=args.days[0], cache_root=args.cache_root, instruments=[args.instrument])
    pp = LOBPreprocessor(cfg)
    df = pp.harmonise(args.days, instrument=args.instrument, tick_size=args.tick_size, price_scale=args.price_scale)
    if df.empty:
        raise SystemExit("No events parsed from provided day folders")
    path = pp.cache(df, instrument=args.instrument)
    print(path)


if __name__ == "__main__":
    main()

