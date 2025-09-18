#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Append synthetic returns to predictions CSV")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n = len(df)
    rng = np.random.default_rng(int(args.seed))
    # create light autocorrelated noise to induce RV variation
    e = rng.normal(0, 0.01, size=n)
    r = np.zeros(n, dtype=float)
    alpha = 0.8
    for i in range(1, n):
        r[i] = alpha * r[i-1] + e[i]
    out = df.copy()
    out["ret"] = r
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(str(args.output))


if __name__ == "__main__":
    main()

