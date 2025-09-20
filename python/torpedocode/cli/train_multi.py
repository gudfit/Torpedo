"""CLI: One-command multi-horizon training/eval for the hybrid model.

Wraps batch_train with a default label key set:
  instability_s_{1,5,10} and instability_e_{100,500}

Usage:
  python -m torpedocode.cli.train_multi --cache-root ./cache --artifact-root ./artifacts \
    [--instruments AAPL MSFT] [--epochs 3] [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Multi-horizon train/eval wrapper")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--instruments", type=str, nargs="*")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bptt", type=int, default=64)
    ap.add_argument("--topo-stride", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--temperature-scale", action="store_true")
    ap.add_argument("--tpp-diagnostics", action="store_true")
    ap.add_argument("--strict-tda", action="store_true")
    ap.add_argument("--include-market-embedding", action="store_true")
    ap.add_argument("--topology-json", type=Path, default=None)
    ap.add_argument("--use-topo-selected", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--log-splits", action="store_true")
    ap.add_argument("--warm-start", type=Path, default=None, help="Checkpoint to warm-start all runs")
    args = ap.parse_args()

    labels = ["instability_s_1", "instability_s_5", "instability_s_10", "instability_e_100", "instability_e_500"]

    from . import batch_train as btrain

    argv = [
        "prog",
        "--cache-root",
        str(args.cache_root),
        "--artifact-root",
        str(args.artifact_root),
        "--label-keys",
        *labels,
        "--epochs",
        str(int(args.epochs)),
        "--batch",
        str(int(args.batch)),
        "--bptt",
        str(int(args.bptt)),
        "--topo-stride",
        str(int(args.topo_stride)),
        "--hidden",
        str(int(args.hidden)),
        "--layers",
        str(int(args.layers)),
        "--lr",
        str(float(args.lr)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--beta",
        str(float(args.beta)),
    ]
    if args.progress:
        argv.append("--progress")
    if args.log_splits:
        argv.append("--log-splits")
    if args.topology_json is not None:
        argv += ["--topology-json", str(args.topology_json)]
    if args.use_topo_selected:
        argv.append("--use-topo-selected")
    if args.warm_start is not None:
        argv += ["--warm-start", str(args.warm_start)]
    if args.instruments:
        argv += ["--instruments", *args.instruments]
    if args.temperature_scale:
        argv.append("--temperature-scale")
    if args.tpp_diagnostics:
        argv.append("--tpp-diagnostics")
    if args.strict_tda:
        argv.append("--strict-tda")
    if args.include_market_embedding:
        argv.append("--include-market-embedding")

    old = sys.argv
    try:
        sys.argv = argv
        btrain.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
