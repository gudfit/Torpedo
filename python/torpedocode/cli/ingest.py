"""CLI: Harmonize and cache raw feeds in a folder.

Supports NDJSON/JSONL, ITCH/OUCH minimal binary, and LOBSTER CSV pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import DataConfig
from ..data.pipeline import LOBPreprocessor


def main():
    ap = argparse.ArgumentParser(description="Harmonize + cache raw feeds in a folder")
    ap.add_argument("--raw-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--instrument", type=str, required=True)
    ap.add_argument("--tick-size", type=float, default=None)
    ap.add_argument("--price-scale", type=float, default=None)
    ap.add_argument("--drop-auctions", dest="drop_auctions", action="store_true")
    ap.add_argument("--no-drop-auctions", dest="drop_auctions", action="store_false")
    ap.set_defaults(drop_auctions=True)
    ap.add_argument(
        "--itch-spec", type=str, default=None, help="Vendor spec hint, e.g., nasdaq-itch-5.0"
    )
    ap.add_argument(
        "--ouch-spec", type=str, default=None, help="Vendor spec hint, e.g., nasdaq-ouch-4.2"
    )
    ap.add_argument(
        "--session-tz",
        type=str,
        default="America/New_York",
        help="Market local time zone for session hours (e.g., America/New_York, Europe/London)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs for native paths and fuse selection",
    )
    ap.add_argument(
        "--check", action="store_true", help="Print available native components and exit"
    )
    ap.add_argument(
        "--eta", type=float, default=None, help="Instability threshold |Δmid| > eta to store labels"
    )
    ap.add_argument(
        "--actions-csv",
        type=Path,
        default=None,
        help="Corporate actions CSV for preview/adjustment validation",
    )
    ap.add_argument(
        "--validate-actions",
        action="store_true",
        help="Emit a small summary of corporate actions adjustments",
    )
    ap.add_argument(
        "--apply-actions",
        action="store_true",
        help="Apply corporate actions to prices prior to caching (divide by adj_factor by default)",
    )
    ap.add_argument(
        "--actions-date-col",
        type=str,
        default="date",
        help="Date column name in corporate actions CSV",
    )
    ap.add_argument(
        "--actions-symbol-col",
        type=str,
        default="symbol",
        help="Symbol column name in corporate actions CSV",
    )
    ap.add_argument(
        "--actions-factor-col",
        type=str,
        default="adj_factor",
        help="Adjustment factor column name in corporate actions CSV",
    )
    ap.add_argument(
        "--actions-mode",
        type=str,
        choices=["divide", "multiply"],
        default="divide",
        help="Whether to divide or multiply price-like columns by adj_factor",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON path to write actions summary",
    )
    ap.add_argument(
        "--sample-csv", type=Path, default=None, help="Optional CSV sample of adjusted rows"
    )
    args = ap.parse_args()

    if args.verbose:
        import os

        os.environ["TORPEDOCODE_VERBOSE"] = "1"

    if args.check:
        import importlib

        try:
            from ..utils.ops import has_torpedocode_op

            ok_op = has_torpedocode_op()
        except Exception:
            ok_op = False
        try:
            mod = importlib.import_module("torpedocode_ingest")
            ok_native = True
        except Exception:
            ok_native = False
        print(
            json.dumps(
                {
                    "torch_op_available": ok_op,
                    "native_parser_available": ok_native,
                    "itch_spec": args.itch_spec,
                    "ouch_spec": args.ouch_spec,
                },
                indent=2,
            )
        )
        return

    cfg = DataConfig(
        raw_data_root=args.raw_dir[0],
        cache_root=args.cache_root,
        instruments=[args.instrument],
        drop_auctions=bool(args.drop_auctions),
        session_time_zone=args.session_tz,
        itch_spec=args.itch_spec,
        ouch_spec=args.ouch_spec,
    )
    pp = LOBPreprocessor(cfg)
    sources = [Path(p) for p in args.raw_dir]
    df = pp.harmonise(
        sources, instrument=args.instrument, tick_size=args.tick_size, price_scale=args.price_scale
    )
    if df.empty and bool(args.drop_auctions):
        cfg2 = DataConfig(
            raw_data_root=args.raw_dir[0],
            cache_root=args.cache_root,
            instruments=[args.instrument],
            drop_auctions=False,
            session_time_zone=args.session_tz,
            itch_spec=args.itch_spec,
            ouch_spec=args.ouch_spec,
        )
        pp2 = LOBPreprocessor(cfg2)
        df = pp2.harmonise(
            sources,
            instrument=args.instrument,
            tick_size=args.tick_size,
            price_scale=args.price_scale,
        )
    if args.actions_csv is not None and df is not None and not df.empty:
        try:
            from ..data.preprocess import (
                _load_corporate_actions_csv,
                adjust_corporate_actions,
                round_prices_to_tick,
            )
            import numpy as _np
            import pandas as _pd

            ca = _load_corporate_actions_csv(
                args.actions_csv,
                date_col=str(args.actions_date_col),
                symbol_col=str(args.actions_symbol_col),
                factor_col=str(args.actions_factor_col),
            )
            before = df.copy()
            after = adjust_corporate_actions(before, ca, mode=str(args.actions_mode))
            if args.tick_size is not None and args.tick_size > 0 and True:
                after = round_prices_to_tick(after, float(args.tick_size))
            price_cols = [
                c
                for c in after.columns
                if c == "price" or c.startswith("bid_price_") or c.startswith("ask_price_")
            ]
            sample = None
            if price_cols:
                comp = _pd.DataFrame({"timestamp": _pd.to_datetime(after["timestamp"]).astype(str)})
                for c in price_cols:
                    comp[f"{c}_before"] = _pd.to_numeric(before[c], errors="coerce")
                    comp[f"{c}_after"] = _pd.to_numeric(after[c], errors="coerce")
                    comp[f"{c}_delta"] = comp[f"{c}_after"] - comp[f"{c}_before"]
                sample = comp.head(50)
            summary = {
                "rows": int(len(df)),
                "price_cols": price_cols,
                "sample_rows": 0 if sample is None else int(len(sample)),
            }
            if bool(args.validate_actions) and args.summary_json is not None:
                args.summary_json.parent.mkdir(parents=True, exist_ok=True)
                with open(args.summary_json, "w") as f:
                    json.dump(summary, f, indent=2)
            if bool(args.validate_actions) and args.sample_csv is not None and sample is not None:
                args.sample_csv.parent.mkdir(parents=True, exist_ok=True)
                sample.to_csv(args.sample_csv, index=False)
            if bool(args.apply_actions):
                df = after
        except Exception:
            pass
    if args.eta is not None:
        df = pp.add_instability_labels(df, eta=float(args.eta))
    if df.empty:
        raise SystemExit("No events parsed. Check input directory and file patterns.")
    try:
        import pyarrow  # noqa: F401
    except Exception:
        print("Parsed events; skipping cache (pyarrow not installed)")
        return
    path = pp.cache(df, instrument=args.instrument)
    try:
        import time as _time

        cfg_art = {
            "created_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            "instrument": args.instrument,
            "drop_auctions": bool(args.drop_auctions),
            "session_time_zone": args.session_tz,
            "tick_size": None if args.tick_size is None else float(args.tick_size),
            "price_scale": None if args.price_scale is None else float(args.price_scale),
            "itch_spec": args.itch_spec,
            "ouch_spec": args.ouch_spec,
            "actions_csv": str(args.actions_csv) if args.actions_csv is not None else None,
            "actions_mode": str(args.actions_mode) if args.actions_csv is not None else None,
            "actions_cols": (
                {
                    "date": str(args.actions_date_col),
                    "symbol": str(args.actions_symbol_col),
                    "factor": str(args.actions_factor_col),
                }
                if args.actions_csv is not None
                else None
            ),
        }
        meta_path = path.with_suffix(".ingest.json")
        with open(meta_path, "w") as f:
            json.dump(cfg_art, f, indent=2)
    except Exception:
        pass
    print(f"Cached harmonized events to {path}")


if __name__ == "__main__":
    main()
