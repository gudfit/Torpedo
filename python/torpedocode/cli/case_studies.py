"""CLI: Case studies of persistence diagrams and model outputs at high |Δmid| times.

Selects timestamps in the test split with the largest absolute mid-price
changes over a clock horizon, computes persistence diagrams via the current
topology config, and extracts model outputs at those times. Writes a JSON
report with diagrams (H0/H1) and probabilities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..config import DataConfig, TopologyConfig, ModelConfig
from ..data.loader import MarketDataLoader, LOBDatasetBuilder
from ..features.topological import TopologicalFeatureGenerator
from ..models.hybrid import HybridLOBModel


def _compute_mid(df: pd.DataFrame) -> np.ndarray:
    if {"bid_price_1", "ask_price_1"}.issubset(df.columns):
        bid = pd.to_numeric(df["bid_price_1"], errors="coerce").ffill().bfill()
        ask = pd.to_numeric(df["ask_price_1"], errors="coerce").ffill().bfill()
        mid = (bid + ask) / 2.0
    else:
        mid = pd.to_numeric(df.get("price", 0.0), errors="coerce").ffill().bfill()
    return mid.to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser(
        description="Case studies: persistence diagrams and outputs at high |Δmid|"
    )
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--instrument", type=str, required=True)
    ap.add_argument("--horizon-s", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--topo-window-s", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--strict-tda", action="store_true", help="Fail if TDA backends missing")
    ap.add_argument("--state-dict", type=Path, default=None)
    args = ap.parse_args()

    data = DataConfig(
        raw_data_root=args.cache_root, cache_root=args.cache_root, instruments=[args.instrument]
    )
    mdl = MarketDataLoader(data)
    df = mdl.load_events(args.instrument)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mid = _compute_mid(df)

    fut = ts + pd.to_timedelta(int(args.horizon_s), unit="s")
    ts_ns = ts.astype("int64").to_numpy()
    fut_ns = fut.astype("int64").to_numpy()
    idx = np.searchsorted(ts_ns, fut_ns, side="left")
    idx = np.clip(idx, 0, len(mid) - 1)
    abs_dmid = np.abs(mid[idx] - mid)

    builder = LOBDatasetBuilder(data)
    tr, va, te, scaler = builder.build_splits(
        args.instrument,
        label_key=f"instability_s_{int(args.horizon_s)}",
        topology=TopologyConfig(
            window_sizes_s=[int(args.topo_window_s)],
            strict_tda=(bool(args.strict_tda) or (
                __import__('os').environ.get('PAPER_TORPEDO_STRICT_TDA', '0').lower() in {'1','true'}
            )),
        ),
        topo_stride=1,
        artifact_dir=None,
    )
    T = len(ts)
    t0 = int(0.6 * T)
    v0 = int(0.8 * T)
    test_mask = np.zeros(T, dtype=bool)
    test_mask[v0:] = True

    ranks = np.argsort(-abs_dmid)
    sel: List[int] = []
    for i in ranks:
        if test_mask[i]:
            sel.append(i)
        if args.top_k is not None and len(sel) >= int(args.top_k):
            break
    if args.top_k is None:
        thresh = (
            float(np.quantile(abs_dmid[test_mask], float(args.quantile)))
            if np.any(test_mask)
            else float("inf")
        )
        sel = [i for i in range(T) if test_mask[i] and abs_dmid[i] >= thresh]

    topo_cfg = TopologyConfig(window_sizes_s=[int(args.topo_window_s)], strict_tda=bool(args.strict_tda))
    topo = TopologicalFeatureGenerator(topo_cfg)
    X_scaled = scaler.transform(builder.build_sequence(args.instrument)["features_raw"])
    Z_all = topo.rolling_transform(ts.to_numpy(), X_scaled, stride=1)

    probs = None
    if torch is not None:
        try:
            Fdim = te["features"].shape[1]
            Zdim = te["topology"].shape[1]
            model = HybridLOBModel(
                Fdim,
                Zdim,
                num_event_types=6,
                config=ModelConfig(hidden_size=64, num_layers=1, include_market_embedding=False),
            )
            if args.state_dict and Path(args.state_dict).exists():
                state = torch.load(args.state_dict, map_location="cpu")
                sd = state.get("state_dict", state)
                model.load_state_dict(sd, strict=False)
            model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(te["features"]).unsqueeze(0)
                zb = torch.from_numpy(te["topology"]).unsqueeze(0)
                out = model(xb, zb)
                probs = torch.sigmoid(out.instability_logits[0, :, 0]).cpu().numpy()
        except Exception:
            probs = None

    cases: List[Dict] = []
    topo_gen = TopologicalFeatureGenerator(topo_cfg)
    series = X_scaled
    ts_np = ts.to_numpy()
    w_ns = int(args.topo_window_s) * 1_000_000_000
    for i in sel:
        left = ts.iloc[i].value - w_ns
        j0 = int(np.searchsorted(ts.astype("int64").to_numpy(), left, side="right"))
        slab = series[j0 : i + 1]
        try:
            if topo_cfg.complex_type == "cubical" and getattr(
                topo_cfg, "use_liquidity_surface", True
            ):
                field = topo_gen._liquidity_surface_field(slab)
                H0, H1 = topo_gen._compute_diagram(field)
            else:
                H0, H1 = topo_gen._compute_diagram(slab)
            h0 = H0.tolist() if isinstance(H0, np.ndarray) else []
            h1 = H1.tolist() if isinstance(H1, np.ndarray) else []
        except Exception:
            h0, h1 = [], []
        rec = {
            "index": int(i - v0),
            "timestamp": str(ts.iloc[i].to_pydatetime()),
            "abs_dmid": float(abs_dmid[i]),
            "topo_vector": Z_all[i].tolist() if i < len(Z_all) else [],
            "diagram_H0": h0,
            "diagram_H1": h1,
        }
        if probs is not None and i >= v0:
            rec["prob"] = float(probs[i - v0])
        cases.append(rec)

    out = {
        "instrument": args.instrument,
        "horizon_s": int(args.horizon_s),
        "topo_window_s": int(args.topo_window_s),
        "num_cases": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"num_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
