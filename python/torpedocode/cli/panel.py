"""CLI: Build liquidity panel and match instruments across markets.

Reads a CSV with columns including at least: market,symbol,median_daily_notional[,tick_size].
Outputs a JSON/CSV with added columns liq_decile and match_group for cross-market matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..data.preprocess import compute_liquidity_panel, match_instruments_across_markets


def main():
    ap = argparse.ArgumentParser(description="Compute liquidity deciles and match groups")
    ap.add_argument("--input", type=Path, required=True, help="CSV with per-instrument stats")
    ap.add_argument(
        "--by", type=str, nargs="*", default=["liq_decile", "tick_size"], help="Columns to match on"
    )
    ap.add_argument("--output", type=Path, required=False, help="Output JSON/CSV path")
    args = ap.parse_args()

    import shutil

    rust_bin = shutil.which("torpedocode-panel")
    if rust_bin is not None:
        import subprocess

        cmd = [rust_bin, "--input", str(args.input), "--by", *args.by]
        if args.output is not None:
            cmd += ["--output", str(args.output)]
            subprocess.run(cmd, check=True)
            return
        out = subprocess.check_output(cmd, text=True)
        print(out)
        return

    df = pd.read_csv(args.input)
    panel = compute_liquidity_panel(df)
    matched = match_instruments_across_markets(panel, by=args.by)

    if args.output is None:
        print(matched.to_json(orient="records"))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            with open(args.output, "w") as f:
                json.dump(json.loads(matched.to_json(orient="records")), f, indent=2)
        else:
            matched.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
