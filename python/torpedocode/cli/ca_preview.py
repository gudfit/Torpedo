"""CLI: Corporate actions preview and price adjustment sanity-check.

Loads a canonical event frame (Parquet produced by LOBPreprocessor.cache or NDJSON),
loads a corporate actions CSV with per-symbol per-date adjustment factors, applies
adjustments over a selected date range, and optionally re-rounds to a tick grid.
Outputs a JSON summary and optionally writes a small CSV sample of adjusted rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..data.preprocess import _load_corporate_actions_csv, adjust_corporate_actions


def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".ndjson", ".jsonl"}:
        recs: List[dict] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            recs.append(obj)
        return pd.DataFrame.from_records(recs)
    raise ValueError(f"Unsupported input: {path}")


def _round_to_tick(df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    out = df.copy()
    cols = [
        c
        for c in out.columns
        if c == "price" or c.startswith("bid_price_") or c.startswith("ask_price_")
    ]
    if tick_size is None or tick_size <= 0:
        return out
    for c in cols:
        try:
            p = pd.to_numeric(out[c], errors="coerce")
            out[c] = (np.round(p / tick_size) * tick_size).astype(float)
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description="Preview corporate actions price adjustments")
    ap.add_argument(
        "--input", type=Path, required=True, help="Parquet (preferred) or NDJSON canonical events"
    )
    ap.add_argument(
        "--actions-csv", type=Path, required=True, help="CSV with symbol,date,adj_factor"
    )
    ap.add_argument("--symbol", type=str, default=None, help="Optional symbol filter")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--tick-size", type=float, default=None, help="Tick size; used when --round-ticks is set"
    )
    ap.add_argument(
        "--round-ticks", action="store_true", help="Re-round adjusted prices to tick grid"
    )
    ap.add_argument(
        "--sample-csv", type=Path, default=None, help="Optional CSV output with before/after sample"
    )
    ap.add_argument("--limit", type=int, default=50, help="Number of sample rows to emit")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON summary output")
    ap.add_argument("--date-col", type=str, default="date")
    ap.add_argument("--symbol-col", type=str, default="symbol")
    ap.add_argument("--factor-col", type=str, default="adj_factor")
    args = ap.parse_args()

    df = _load_input(args.input)
    if df.empty:
        raise SystemExit("No data in input frame")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if args.symbol is not None and "symbol" in df.columns:
        df = df.loc[df["symbol"].astype(str) == str(args.symbol)]
    if args.start:
        df = df.loc[df["timestamp"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df.loc[df["timestamp"] <= pd.Timestamp(args.end, tz="UTC")]

    if df.empty:
        raise SystemExit("No data after filters")

    ca = _load_corporate_actions_csv(
        args.actions_csv,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
        factor_col=args.factor_col,
    )
    before = df.copy()
    after = adjust_corporate_actions(before, ca, mode="divide")
    if args.round_ticks and args.tick_size is not None:
        after = _round_to_tick(after, float(args.tick_size))

    price_cols = [
        c
        for c in after.columns
        if c == "price" or c.startswith("bid_price_") or c.startswith("ask_price_")
    ]
    sample = None
    if price_cols:
        comp = pd.DataFrame({"timestamp": after["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")})
        for c in price_cols:
            comp[f"{c}_before"] = pd.to_numeric(before[c], errors="coerce")
            comp[f"{c}_after"] = pd.to_numeric(after[c], errors="coerce")
            comp[f"{c}_delta"] = comp[f"{c}_after"] - comp[f"{c}_before"]
        sample = comp.head(int(args.limit))
        if args.sample_csv is not None:
            args.sample_csv.parent.mkdir(parents=True, exist_ok=True)
            sample.to_csv(args.sample_csv, index=False)

    tmp = adjust_corporate_actions(before, ca, mode="divide")
    factors = []
    if "price" in price_cols:
        b = pd.to_numeric(before["price"], errors="coerce")
        a = pd.to_numeric(tmp["price"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.isfinite(b) & (b != 0), b / a, 1.0)
        factors = f[np.isfinite(f)]
    adj_count = int(np.sum((np.asarray(factors) != 1.0))) if len(factors) else 0

    out = {
        "rows": int(len(df)),
        "price_cols": price_cols,
        "adjusted_rows_estimate": adj_count,
        "factor_min": float(np.nanmin(factors)) if len(factors) else 1.0,
        "factor_max": float(np.nanmax(factors)) if len(factors) else 1.0,
        "sample_rows": int(len(sample)) if sample is not None else 0,
        "sample_csv": str(args.sample_csv) if args.sample_csv is not None else None,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
