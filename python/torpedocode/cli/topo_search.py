"""CLI: Light topology grid search on a validation slice using logistic baseline.

Explores a small grid of TopologyConfig settings and selects the best by validation AUROC.
Writes the chosen topology config JSON into artifacts for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

from ..config import DataConfig, TopologyConfig
from ..data.loader import LOBDatasetBuilder
from .baselines import _fit_predict_logistic
from ..evaluation.metrics import compute_classification_metrics


def main():
    ap = argparse.ArgumentParser(description="Topology grid search via logistic baseline")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--instrument", type=str, required=True)
    ap.add_argument("--label-key", type=str, required=True)
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--strict-tda", action="store_true", help="Fail if TDA backends missing")
    args = ap.parse_args()

    data = DataConfig(
        raw_data_root=args.cache_root, cache_root=args.cache_root, instruments=[args.instrument]
    )
    builder = LOBDatasetBuilder(data)
    import os as _os
    env_strict = _os.environ.get("PAPER_TORPEDO_STRICT_TDA", "0").lower() in {"1", "true"}

    window_sizes = [[1], [5], [10]]
    reps = ["landscape", "image"]
    Ks = [3, 5]
    img_res = [32, 64, 128]
    img_bw = [0.02, 0.05]

    best = None
    best_cfg = None
    for ws in window_sizes:
        for rep in reps:
            if rep == "landscape":
                for K in Ks:
                    topo = TopologyConfig(
                        window_sizes_s=ws,
                        persistence_representation="landscape",
                        landscape_levels=K,
                        strict_tda=(bool(args.strict_tda) or env_strict),
                    )
                    tr, va, _, _ = builder.build_splits(
                        args.instrument,
                        label_key=args.label_key,
                        topology=topo,
                        topo_stride=5,
                        artifact_dir=None,
                    )
                    Xtr = np.hstack([tr["features"], tr["topology"]])
                    Xva = np.hstack([va["features"], va["topology"]])
                    p = _fit_predict_logistic(Xtr, tr["labels"].astype(int), Xva)
                    m = compute_classification_metrics(p, va["labels"].astype(int))
                    score = m.auroc
                    if best is None or score > best:
                        best, best_cfg = score, topo
            else:
                for res in img_res:
                    for bw in img_bw:
                        topo = TopologyConfig(
                            window_sizes_s=ws,
                            persistence_representation="image",
                            image_resolution=res,
                            image_bandwidth=bw,
                            strict_tda=(bool(args.strict_tda) or env_strict),
                        )
                        tr, va, _, _ = builder.build_splits(
                            args.instrument,
                            label_key=args.label_key,
                            topology=topo,
                            topo_stride=5,
                            artifact_dir=None,
                        )
                        Xtr = np.hstack([tr["features"], tr["topology"]])
                        Xva = np.hstack([va["features"], va["topology"]])
                        p = _fit_predict_logistic(Xtr, tr["labels"].astype(int), Xva)
                        m = compute_classification_metrics(p, va["labels"].astype(int))
                        score = m.auroc
                        if best is None or score > best:
                            best, best_cfg = score, topo

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(args.artifact_dir / "topology_selected.json", "w") as f:
        json.dump(asdict(best_cfg if best_cfg is not None else TopologyConfig()), f, indent=2)
    print(
        json.dumps(
            {
                "val_auroc": best,
                "topology": asdict(best_cfg) if best_cfg is not None else asdict(TopologyConfig()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
