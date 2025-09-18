"""CLI: Cross-market orchestration (pooled vs LOMO) with the hybrid model.

Loads a matched panel (CSV or JSON records) with at least columns: market, symbol.
Builds train/val/test splits via LOBDatasetBuilder and trains a compact HybridLOBModel
on windowed sequences (TBPTT-style) pooled across markets. Reports per-market metrics
and global micro/macro aggregates.

Modes:
- pooled: train on all markets (train+val windows), evaluate per-market (test splits).
- lomo: for each market M*, train on all other markets, evaluate on M* (test splits).

Example:
  python -m torpedocode.cli.cross_market \
    --panel artifacts/panel_matched.csv --cache-root ./cache --label-key instability_s_5 \
    --mode pooled --with-tda --epochs 1 --device cpu --output artifacts/cross_market_pooled.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import DataConfig, TopologyConfig
from ..data.loader import LOBDatasetBuilder
from ..evaluation.metrics import compute_classification_metrics, delong_ci_auroc
from ..models.hybrid import HybridLOBModel
from ..training.losses import HybridLossComputer
from ..config import ModelConfig
from . import train as train_cli
import torch


def _load_panel(path: Path) -> List[Dict]:
    txt = Path(path).read_text()
    if path.suffix.lower() == ".json":
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [dict(r) for r in obj]
        raise ValueError("JSON panel must be a list of records with market,symbol")
    # CSV
    import csv

    out: List[Dict] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(dict(row))
    return out


def _xz(
    builder: LOBDatasetBuilder, instrument: str, label_key: str, with_tda: bool, strict_tda: bool
) -> Tuple[Dict, Dict, Dict]:
    topo = TopologyConfig(strict_tda=bool(strict_tda))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key=label_key, topology=topo, topo_stride=5, artifact_dir=None
    )

    def featz(split: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = split["features"]
        Z = split["topology"]
        y = split["labels"].astype(int)
        return (np.hstack([X, Z]) if with_tda else X), Z, y

    Xt, Zt, yt = featz(tr)
    Xv, Zv, yv = featz(va)
    Xe, Ze, ye = featz(te)
    return (
        {"X": Xt, "Z": Zt, "y": yt},
        {"X": Xv, "Z": Zv, "y": yv},
        {"X": Xe, "Z": Ze, "y": ye},
    )


def _metrics(p: np.ndarray, y: np.ndarray) -> Dict:
    m = compute_classification_metrics(p, y)
    auc, lo, hi = delong_ci_auroc(p, y)
    return {"auroc": auc, "auroc_ci": [lo, hi], "auprc": m.auprc, "brier": m.brier, "ece": m.ece}


def main():
    ap = argparse.ArgumentParser(
        description="Cross-market pooled/LOMO evaluation (logistic baseline)"
    )
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--label-key", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["pooled", "lomo"], default="pooled")
    ap.add_argument("--with-tda", action="store_true")
    ap.add_argument("--strict-tda", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bptt", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1e-4, help="Smoothness penalty weight")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    panel = _load_panel(args.panel)
    # Normalize column names
    recs = []
    for r in panel:
        mk = r.get("market") or r.get("Market") or r.get("mkt")
        sy = r.get("symbol") or r.get("Symbol") or r.get("instr")
        if mk is None or sy is None:
            continue
        recs.append({"market": str(mk), "symbol": str(sy)})
    if not recs:
        raise SystemExit("Panel must include columns: market,symbol")
    markets = sorted({r["market"] for r in recs})

    out: Dict = {"mode": args.mode, "label_key": args.label_key, "markets": {}}

    def build_builder() -> LOBDatasetBuilder:
        data = DataConfig(raw_data_root=args.cache_root, cache_root=args.cache_root, instruments=[])
        return LOBDatasetBuilder(data)

    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    def _train_model(
        loaders: List[Dict[str, torch.Tensor]], Fdim: int, Zdim: int
    ) -> HybridLOBModel:
        model = HybridLOBModel(
            Fdim,
            Zdim,
            num_event_types=6,
            config=ModelConfig(
                hidden_size=args.hidden, num_layers=args.layers, include_market_embedding=False
            ),
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
        loss_fn = HybridLossComputer(lambda_cls=0.0, beta=float(args.beta), gamma=0.0)
        model.train()
        for _ in range(int(args.epochs)):
            for batch in loaders:
                xb = batch["features"].to(dev)
                zb = batch["topology"].to(dev)
                yb = batch["instability_labels"].float().to(dev)
                out = model(xb, zb)
                lo = loss_fn(out, {**batch, "instability_labels": yb}, list(model.parameters()))
                opt.zero_grad()
                lo.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return model

    if args.mode == "pooled":
        builder = build_builder()
        # Pool training windows across all markets; evaluate per market tests
        loaders: List[Dict[str, torch.Tensor]] = []
        per_market_tests: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        Fdim = Zdim = None  # infer from first split
        for m in markets:
            insts = [r["symbol"] for r in recs if r["market"] == m]
            for sym in insts:
                tr, va, te = _xz(builder, sym, args.label_key, args.with_tda, args.strict_tda)
                if Fdim is None:
                    Fdim = tr["X"].shape[1] if not args.with_tda else (tr["X"].shape[1])
                    Zdim = 0  # topology is embedded into X when with_tda=True
                # Build loaders from train+val windows
                Ztr = (
                    np.zeros((tr["X"].shape[0], 0), dtype=np.float32) if args.with_tda else tr["Z"]
                )  # embed topology into X when with_tda
                Zva = (
                    np.zeros((va["X"].shape[0], 0), dtype=np.float32) if args.with_tda else va["Z"]
                )  # ditto
                tr_loader = train_cli._window_batches(
                    tr["X"],
                    Ztr,
                    tr["y"],
                    bptt=int(args.bptt),
                    batch_size=int(args.batch),
                    balanced=True,
                )
                va_loader = train_cli._window_batches(
                    va["X"],
                    Zva,
                    va["y"],
                    bptt=int(args.bptt),
                    batch_size=int(args.batch),
                    balanced=True,
                )
                loaders.extend(list(tr_loader))
                loaders.extend(list(va_loader))
                per_market_tests.setdefault(
                    m,
                    (
                        np.empty((0, tr["X"].shape[1]), dtype=np.float32),
                        np.empty((0,), dtype=int),
                        np.empty((0, 0), dtype=np.float32),
                    ),
                )
                Xe, ye, Ze = per_market_tests[m]
                Ze_te = (
                    np.zeros((te["X"].shape[0], 0), dtype=np.float32) if args.with_tda else te["Z"]
                )  # zeros if embedded
                per_market_tests[m] = (
                    np.vstack([Xe, te["X"]]),
                    np.concatenate([ye, te["y"]]),
                    np.vstack([Ze, Ze_te]),
                )
        if not loaders:
            raise SystemExit("No training data constructed from panel")
        model = _train_model(loaders, int(Fdim), int(Zdim))
        # Evaluate per market and global
        pm = {}
        micro_p = []
        micro_y = []
        for m, (Xe, ye, Ze) in per_market_tests.items():
            logits, _lam = train_cli._predict_sequence(model, Xe, Ze, dev)
            p = 1.0 / (1.0 + np.exp(-logits))
            pm[m] = _metrics(p, ye)
            micro_p.append(p)
            micro_y.append(ye)
        out["markets"] = pm
        if micro_p:
            P = np.concatenate(micro_p)
            Y = np.concatenate(micro_y)
            out["micro"] = _metrics(P, Y)
            out["macro"] = {
                k: (float(np.mean([pm[m][k] for m in pm])) if k != "auroc_ci" else None)
                for k in pm[markets[0]].keys()
            }
    else:  # LOMO
        pm = {}
        for held in markets:
            builder = build_builder()
            loaders: List[Dict[str, torch.Tensor]] = []
            test_X: List[np.ndarray] = []
            test_y: List[np.ndarray] = []
            test_Z: List[np.ndarray] = []
            Fdim = Zdim = None
            for m in markets:
                insts = [r["symbol"] for r in recs if r["market"] == m]
                for sym in insts:
                    tr, va, te = _xz(builder, sym, args.label_key, args.with_tda, args.strict_tda)
                    if m == held:
                        Ze_te = (
                            np.zeros((te["X"].shape[0], 0), dtype=np.float32)
                            if args.with_tda
                            else te["Z"]
                        )  # zeros if embedded
                        test_X.append(te["X"])
                        test_y.append(te["y"])
                        test_Z.append(Ze_te)
                    else:
                        if Fdim is None:
                            Fdim = tr["X"].shape[1]
                            Zdim = 0
                        Ztr = (
                            np.zeros((tr["X"].shape[0], 0), dtype=np.float32)
                            if args.with_tda
                            else tr["Z"]
                        )  # zeros if embedded
                        Zva = (
                            np.zeros((va["X"].shape[0], 0), dtype=np.float32)
                            if args.with_tda
                            else va["Z"]
                        )  # zeros if embedded
                        loaders.extend(
                            list(
                                train_cli._window_batches(
                                    tr["X"],
                                    Ztr,
                                    tr["y"],
                                    bptt=int(args.bptt),
                                    batch_size=int(args.batch),
                                    balanced=True,
                                )
                            )
                        )
                        loaders.extend(
                            list(
                                train_cli._window_batches(
                                    va["X"],
                                    Zva,
                                    va["y"],
                                    bptt=int(args.bptt),
                                    batch_size=int(args.batch),
                                    balanced=True,
                                )
                            )
                        )
            if not loaders or not test_X:
                continue
            model = _train_model(loaders, int(Fdim), int(Zdim))
            Xe = np.vstack(test_X)
            ye = np.concatenate(test_y)
            Ze = np.vstack(test_Z)
            logits, _lam = train_cli._predict_sequence(model, Xe, Ze, dev)
            p = 1.0 / (1.0 + np.exp(-logits))
            pm[held] = _metrics(p, ye)
        out["markets"] = pm
        if pm:
            # Macro average across held-out markets
            anyk = list(next(iter(pm.values())).keys())
            out["macro"] = {
                k: (float(np.mean([pm[m][k] for m in pm])) if k != "auroc_ci" else None)
                for k in anyk
            }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
