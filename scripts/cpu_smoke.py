#!/usr/bin/env python3
"""CPU-only smoke tests: broad coverage of core components.

Run with:
    uv run python scripts/cpu_smoke.py
or:
    .venv/bin/python scripts/cpu_smoke.py
"""

from __future__ import annotations

import numpy as np

from torpedocode.evaluation.metrics import (
    compute_calibration_report,
    compute_classification_metrics,
    compute_point_process_diagnostics,
)
from torpedocode.evaluation.tpp import (
    TPPArrays,
    nll_per_event_from_arrays,
    rescaled_times,
    rescaled_times_per_type,
)
from torpedocode.evaluation.economic import (
    var_es,
    kupiec_pof_test,
    christoffersen_independence_test,
    block_bootstrap_var_es,
)
from torpedocode.features.topological import TopologicalFeatureGenerator
import warnings
from torpedocode.config import TopologyConfig
from torpedocode.data.preprocess import harmonise_ndjson, HarmoniseConfig
from torpedocode.features.lob import build_lob_feature_matrix


def run_metrics_smoke() -> None:
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(int)
    cm = compute_classification_metrics(p, y)
    cal = compute_calibration_report(p, y, num_bins=10)
    xi = np.random.exponential(1.0, size=256)
    pp = compute_point_process_diagnostics(
        xi, empirical_frequencies=np.array([0.6, 0.4]), model_frequencies=np.array([0.55, 0.45])
    )
    # TPP arrays path
    lam = np.vstack([np.full(128, 0.8), np.full(128, 0.2)]).T
    et = (np.arange(128) % 2).astype(int)
    dt = np.full(128, 0.5)
    arr = TPPArrays(intensities=lam, event_type_ids=et, delta_t=dt)
    _ = nll_per_event_from_arrays(lam, et, dt)
    _ = rescaled_times(arr)
    _ = rescaled_times_per_type(arr)
    # Economic quick checks
    r = np.random.normal(0, 0.01, size=500)
    _ = var_es(r, alpha=0.99)
    exc = (r < -np.quantile(r, 0.99)).astype(int)
    _ = kupiec_pof_test(exc, alpha=0.99)
    _ = christoffersen_independence_test(exc)
    _ = block_bootstrap_var_es(r, alpha=0.99, expected_block_length=25.0, n_boot=50)
    print("classification:", cm)
    print("calibration bins:", cal.bin_confidence.shape, cal.bin_accuracy.shape)
    print("point-process:", pp)


def run_model_smoke() -> None:
    try:
        import torch
        from torpedocode.config import ModelConfig
        from torpedocode.models.hybrid import HybridLOBModel
        from torpedocode.training.losses import HybridLossComputer
        from torpedocode.data.loader import LOBDatasetBuilder
        from torpedocode.config import DataConfig
    except Exception as e:
        print("[skip] model smoke (torch unavailable):", e)
        return

    feature_dim, topo_dim, num_event_types = 8, 4, 3
    cfg = ModelConfig(hidden_size=32, num_layers=1, include_market_embedding=False)
    model = HybridLOBModel(feature_dim, topo_dim, num_event_types, cfg)
    builder = LOBDatasetBuilder(
        DataConfig(raw_data_root=__file__, cache_root=__file__, instruments=["X"])
    )
    np_batch = builder.build_synthetic_batch(
        batch_size=2,
        T=8,
        num_event_types=num_event_types,
        feature_dim=feature_dim,
        topo_dim=topo_dim,
    )
    batch = {k: torch.tensor(v) for k, v in np_batch.items()}
    out = model(batch["features"], batch["topology"])
    loss = HybridLossComputer(
        lambda_cls=1.0, beta=1e-4, gamma=0.0, cls_loss_type="focal", focal_gamma=2.0
    )
    loss_out = loss(out, batch, list(model.parameters()))
    print("TPP+Mark loss:", float(loss_out.tpp_mark.detach().cpu()))
    print("Classification loss:", float(loss_out.classification.detach().cpu()))


def run_tda_smoke() -> None:
    T = 64
    F = 8
    ts = (np.arange(T).astype("int64") * 1_000_000).astype("datetime64[ns]")
    X = np.random.normal(size=(T, F)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        topo_vr = TopologicalFeatureGenerator(
            TopologyConfig(
                window_sizes_s=[1],
                complex_type="vietoris_rips",
                max_homology_dimension=1,
                persistence_representation="landscape",
                landscape_levels=3,
            )
        )
        Zv = topo_vr.rolling_transform(ts, X, stride=4)
        topo_vi = TopologicalFeatureGenerator(
            TopologyConfig(
                window_sizes_s=[1],
                complex_type="vietoris_rips",
                max_homology_dimension=1,
                persistence_representation="image",
                image_resolution=16,
                image_bandwidth=0.05,
            )
        )
        Zi = topo_vi.rolling_transform(ts, X, stride=4)
    topo_cu = TopologicalFeatureGenerator(
        TopologyConfig(
            window_sizes_s=[1],
            complex_type="cubical",
            max_homology_dimension=1,
            persistence_representation="landscape",
            landscape_levels=3,
        )
    )
    Zc = topo_cu.rolling_transform(ts, X, stride=4)
    print("TDA shapes:", Zv.shape, Zi.shape, Zc.shape)


def run_preprocess_and_lob_smoke(tmpdir: str) -> None:
    import os, json

    nd = os.path.join(tmpdir, "mini.ndjson")
    records = [
        {"timestamp": "2020-01-01T12:00:00Z", "event_type": "TRADE", "price": 100.0, "size": 1.0},
        {
            "timestamp": "2020-01-01T12:00:00.500Z",
            "event_type": "ADD",
            "price": 99.5,
            "size": 2.0,
            "level": 1,
            "side": "B",
        },
        {
            "timestamp": "2020-01-01T12:00:01Z",
            "event_type": "ADD",
            "price": 100.5,
            "size": 2.5,
            "level": 1,
            "side": "S",
        },
    ]
    with open(nd, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    df = harmonise_ndjson(nd, cfg=HarmoniseConfig(time_zone="UTC", drop_auctions=False))
    base, aux = build_lob_feature_matrix(df, levels=1)
    print("Harmonised rows:", len(df), "Features shape:", base.shape)


def main() -> None:
    run_metrics_smoke()
    run_model_smoke()
    run_tda_smoke()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        run_preprocess_and_lob_smoke(td)

    # Optional: exercise training CLI end-to-end when pyarrow and torch are present
    try:
        import pyarrow  # noqa: F401
        import torch  # noqa: F401
        import os
        import sys
        from pathlib import Path
        import pandas as pd

        from torpedocode.cli import train as train_cli

        with tempfile.TemporaryDirectory() as td2:
            cwd = os.getcwd()
            os.chdir(td2)
            try:
                import numpy as _np

                ts = pd.date_range("2025-01-01", periods=120, freq="s", tz="UTC")
                df = pd.DataFrame(
                    {
                        "timestamp": ts,
                        "event_type": ["MO+", "MO-", "LO+", "LO-"] * 30,
                        "size": _np.abs(_np.random.randn(120)).astype(float),
                        "price": 100 + _np.cumsum(_np.random.randn(120)).astype(float) * 0.01,
                        "bid_price_1": 100 + _np.random.randn(120).astype(float) * 0.01,
                        "ask_price_1": 100.1 + _np.random.randn(120).astype(float) * 0.01,
                        "bid_size_1": _np.random.randint(1, 100, size=120),
                        "ask_size_1": _np.random.randint(1, 100, size=120),
                    }
                )
                inst = "SMOKE"
                df.to_parquet(Path(f"{inst}.parquet"), index=False)
                art = Path("artifacts_smoke")
                sys.argv = [
                    "prog",
                    "--instrument",
                    inst,
                    "--label-key",
                    "instability_s_1",
                    "--artifact-dir",
                    str(art),
                    "--epochs",
                    "1",
                    "--batch",
                    "8",
                    "--bptt",
                    "16",
                    "--topo-stride",
                    "4",
                    "--hidden",
                    "16",
                    "--layers",
                    "1",
                    "--device",
                    "cpu",
                    "--beta",
                    "1e-4",
                    "--temperature-scale",
                    "--tpp-diagnostics",
                    "--include-market-embedding",
                    "--market-vocab-size",
                    "2",
                    "--market-id",
                    "0",
                ]
                train_cli.main()
                print("[smoke] train artifacts:", sorted(os.listdir(art)))
            finally:
                os.chdir(cwd)
    except Exception as e:
        print("[skip] training CLI smoke:", e)


if __name__ == "__main__":
    main()
