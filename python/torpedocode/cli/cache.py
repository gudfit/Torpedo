"""CLI: Batch convert NDJSON/JSONL feeds to canonical parquet caches.

Examples:
  - Single file to instrument based on stem:
      python -m torpedocode.cli.cache --input data/AAPL.ndjson --cache-root caches/

  - Many files into one instrument (merged):
      python -m torpedocode.cli.cache --input data/ --instrument XNAS --cache-root caches/

  - One cache per file (instrument = stem):
      python -m torpedocode.cli.cache --input data/ --per-file --cache-root caches/
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

from ..config import DataConfig
from ..data.preprocessing import LOBPreprocessor


def _find_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files = list(root.rglob("*.ndjson")) + list(root.rglob("*.jsonl"))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="NDJSON->Parquet cache builder")
    ap.add_argument("--input", type=Path, required=True, help="NDJSON file or directory")
    ap.add_argument(
        "--cache-root", type=Path, required=True, help="Destination directory for parquet caches"
    )
    ap.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Instrument name; if omitted with single file, uses stem",
    )
    ap.add_argument(
        "--per-file", action="store_true", help="Write one cache per file; instrument = file stem"
    )
    ap.add_argument("--time-zone", type=str, default="UTC")
    ap.add_argument("--drop-auctions", action="store_true", default=False)
    ap.add_argument(
        "--session-tz",
        type=str,
        default="America/New_York",
        help="Market local time zone for session hours (e.g., America/New_York, Europe/London)",
    )
    ap.add_argument("--tick-size", type=float, default=None)
    ap.add_argument("--price-scale", type=float, default=None)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--horizons-s", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--horizons-events", type=int, nargs="*", default=[100, 500])
    ap.add_argument(
        "--eta", type=float, default=None, help="Instability threshold |Δmid| > eta to store labels"
    )

    args = ap.parse_args()

    files = _find_files(args.input)
    if not files:
        raise SystemExit(f"No NDJSON/JSONL files under {args.input}")

    cfg = DataConfig(
        raw_data_root=args.input if args.input.is_dir() else args.input.parent,
        cache_root=args.cache_root,
        instruments=[],
        levels=args.levels,
        forecast_horizons_s=args.horizons_s,
        forecast_horizons_events=args.horizons_events,
        time_zone=args.time_zone,
        drop_auctions=bool(args.drop_auctions),
        session_time_zone=args.session_tz,
    )
    pp = LOBPreprocessor(cfg)

    if args.per_file:
        for f in files:
            inst = f.stem
            frame = pp.harmonise(
                [f], instrument=inst, tick_size=args.tick_size, price_scale=args.price_scale
            )
            frame = pp.add_instability_labels(
                frame, eta=float(args.eta) if args.eta is not None else 0.0
            )
            path = pp.cache(frame, instrument=inst)
            print(str(path))
    else:
        if args.instrument is None:
            if len(files) == 1:
                inst = files[0].stem
            else:
                raise SystemExit(
                    "--instrument is required when merging multiple files into one cache; or pass --per-file"
                )
        else:
            inst = args.instrument
        frame = pp.harmonise(
            files, instrument=inst, tick_size=args.tick_size, price_scale=args.price_scale
        )
        if frame.empty and args.drop_auctions:
            cfg.drop_auctions = False  # type: ignore[attr-defined]
            pp2 = LOBPreprocessor(cfg)
            frame = pp2.harmonise(
                files, instrument=inst, tick_size=args.tick_size, price_scale=args.price_scale
            )
        frame = pp.add_instability_labels(
            frame, eta=float(args.eta) if args.eta is not None else 0.0
        )
        path = pp.cache(frame, instrument=inst)
        print(str(path))


if __name__ == "__main__":
    main()
