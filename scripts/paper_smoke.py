#!/usr/bin/env python3
from __future__ import annotations

"""Paper smoke: 1–2 minute end-to-end CPU check (no network).

Steps:
  1) Generate two tiny instrument caches (synthetic)
  2) Train multi-horizon hybrid (epochs=1) per instrument on CPU
  3) Aggregate pooled metrics across both instruments for one label
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def make_cache(root: Path, inst: str, T: int = 120) -> None:
    ts = pd.date_range("2025-01-01", periods=T, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * (T // 4),
            "size": np.abs(np.random.randn(T)).astype(float),
            "price": 100 + np.cumsum(np.random.randn(T)).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(T).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(T).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=T),
            "ask_size_1": np.random.randint(1, 100, size=T),
        }
    )
    import pyarrow  # noqa: F401

    (root / f"{inst}.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / f"{inst}.parquet", index=False)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Paper smoke: quick end-to-end CPU/GPU check")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    here = Path.cwd()
    cache = here / "cache_smoke"
    art = here / "artifacts_smoke"
    cache.mkdir(exist_ok=True)
    art.mkdir(exist_ok=True)

    # 1) Two tiny caches
    make_cache(cache, "INST_A")
    make_cache(cache, "INST_B")

    # 2) Train multi-horizon (epochs=1) on CPU for s_1 only (fast)
    import subprocess, sys

    cmd = [
        sys.executable,
        "-m",
        "torpedocode.cli.batch_train",
        "--cache-root",
        str(cache),
        "--artifact-root",
        str(art),
        "--instruments",
        "INST_A",
        "INST_B",
        "--label-keys",
        "instability_s_1",
        "--epochs",
        "1",
        "--device",
        str(args.device),
    ]
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)

    # 3) Aggregate pooled metrics
    agg_out = art / "aggregate_instability_s_1.json"
    cmd2 = [
        sys.executable,
        "-m",
        "torpedocode.cli.aggregate",
        "--pred-root",
        str(art),
        "--pred-pattern",
        "*/instability_s_1/predictions_test.csv",
        "--output",
        str(agg_out),
    ]
    print("$", " ".join(cmd2))
    subprocess.check_call(cmd2)
    obj = json.loads(agg_out.read_text())
    print("aggregate:", json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
