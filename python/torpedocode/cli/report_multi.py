"""CLI: Multi-horizon reporting for clock- and event-time labels in one go.

Trains a lightweight logistic baseline per horizon (with TDA features) on the
cached splits and reports AUROC/AUPRC/Brier/ECE for each of:
- instability_s_{1,5,10}
- instability_e_{100,500}

Usage:
  python -m torpedocode.cli.report_multi --cache-root ./cache --instrument AAPL \
    [--horizons-s 1 5 10] [--horizons-events 100 500] [--strict-tda] [--output out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import DataConfig, TopologyConfig
from ..data.loader import LOBDatasetBuilder
from ..evaluation.metrics import compute_classification_metrics
from .baselines import _fit_predict_logistic


def _as_list(x: Iterable[int] | None, default: list[int]) -> list[int]:
    if x is None:
        return default
    return [int(v) for v in x]


def main():
    ap = argparse.ArgumentParser(description="Report metrics for multiple horizons (s/events)")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--instrument", type=str, required=True)
    ap.add_argument("--horizons-s", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--horizons-events", type=int, nargs="*", default=[100, 500])
    ap.add_argument("--strict-tda", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    data = DataConfig(
        raw_data_root=args.cache_root, cache_root=args.cache_root, instruments=[args.instrument]
    )
    builder = LOBDatasetBuilder(data)

    horizons_s = _as_list(args.horizons_s, [1, 5, 10])
    horizons_e = _as_list(args.horizons_events, [100, 500])
    topo = TopologyConfig(strict_tda=bool(args.strict_tda))

    results: dict[str, dict] = {}

    for h in horizons_s:
        label_key = f"instability_s_{int(h)}"
        tr, va, te, _ = builder.build_splits(
            args.instrument, label_key=label_key, topology=topo, topo_stride=5, artifact_dir=None
        )
        Xtr = np.hstack([tr["features"], tr["topology"]])
        Xte = np.hstack([te["features"], te["topology"]])
        p = _fit_predict_logistic(Xtr, tr["labels"].astype(int), Xte)
        m = compute_classification_metrics(p, te["labels"].astype(int))
        results[label_key] = {
            "auroc": float(m.auroc),
            "auprc": float(m.auprc),
            "brier": float(m.brier),
            "ece": float(m.ece),
        }

    for k in horizons_e:
        label_key = f"instability_e_{int(k)}"
        tr, va, te, _ = builder.build_splits(
            args.instrument, label_key=label_key, topology=topo, topo_stride=5, artifact_dir=None
        )
        Xtr = np.hstack([tr["features"], tr["topology"]])
        Xte = np.hstack([te["features"], te["topology"]])
        p = _fit_predict_logistic(Xtr, tr["labels"].astype(int), Xte)
        m = compute_classification_metrics(p, te["labels"].astype(int))
        results[label_key] = {
            "auroc": float(m.auroc),
            "auprc": float(m.auprc),
            "brier": float(m.brier),
            "ece": float(m.ece),
        }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

