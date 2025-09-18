"""Simple training CLI: reads artifacts schema, trains with TBPTT and balanced windows,
and emits prediction files ready for the eval CLI.

Example:
  python -m torpedocode.cli.train --instrument AAPL --label-key instability_s_5 \
    --artifact-dir artifacts/aapl --epochs 5 --batch 256 --bptt 64 --topo-stride 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # allow import; check in main

from ..config import ModelConfig, TrainingConfig, TopologyConfig, ExperimentConfig, DataConfig
from ..training.samplers import BalancedBatchSampler
from ..data.loader import LOBDatasetBuilder
from ..utils.scaler import SplitSafeStandardScaler
from ..evaluation.calibration import TemperatureScaler
from ..evaluation.tpp import (
    TPPArrays,
    rescaled_times,
    model_and_empirical_frequencies,
    nll_per_event_from_arrays,
)


def _window_batches(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    *,
    bptt: int,
    batch_size: int,
    balanced: bool,
    event_type_ids: np.ndarray | None = None,
    delta_t: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    market_ids: np.ndarray | None = None,
) -> Iterable[Dict[str, torch.Tensor]]:
    T = len(y)

    def to_tensor(arr):
        return torch.from_numpy(arr) if (torch is not None) else arr

    if bptt <= 0 or bptt > T:
        bptt = T
    ends = np.arange(bptt - 1, T)
    y_end = y[ends]
    if balanced:
        sampler = BalancedBatchSampler(labels=y_end.tolist(), batch_size=batch_size)
        for idx in sampler:
            idx = np.asarray(idx)
            idx = np.clip(idx, 0, len(ends) - 1)
            feats = np.stack([X[e - bptt + 1 : e + 1] for e in ends[idx]], axis=0)
            topo = np.stack([Z[e - bptt + 1 : e + 1] for e in ends[idx]], axis=0)
            labels = np.stack([y[e - bptt + 1 : e + 1] for e in ends[idx]], axis=0)
            batch: Dict[str, "torch.Tensor | np.ndarray"] = {
                "features": to_tensor(feats),
                "topology": to_tensor(topo),
                "instability_labels": (
                    to_tensor(labels).float() if torch is not None else labels.astype(np.float32)
                ),
            }
            if event_type_ids is not None:
                et = np.stack([event_type_ids[e - bptt + 1 : e + 1] for e in ends[idx]], axis=0)
                batch["event_type_ids"] = to_tensor(et.astype(np.int64))
            if delta_t is not None:
                dt = np.stack([delta_t[e - bptt + 1 : e + 1] for e in ends[idx]], axis=0)
                batch["delta_t"] = to_tensor(dt.astype(np.float32))
            if sizes is not None:
                sz = np.stack([sizes[e - bptt + 1 : e + 1] for e in ends[idx]], axis=0)
                batch["sizes"] = to_tensor(sz.astype(np.float32))
            if market_ids is not None:
                mids = np.asarray([market_ids[e] for e in ends[idx]], dtype=np.int64)
                batch["market_ids"] = to_tensor(mids)
            yield batch
    else:
        for i in range(0, len(ends), batch_size):
            idx = ends[i : i + batch_size]
            feats = np.stack([X[e - bptt + 1 : e + 1] for e in idx], axis=0)
            topo = np.stack([Z[e - bptt + 1 : e + 1] for e in idx], axis=0)
            labels = np.stack([y[e - bptt + 1 : e + 1] for e in idx], axis=0)
            batch: Dict[str, "torch.Tensor | np.ndarray"] = {
                "features": to_tensor(feats),
                "topology": to_tensor(topo),
                "instability_labels": (
                    to_tensor(labels).float() if torch is not None else labels.astype(np.float32)
                ),
            }
            if event_type_ids is not None:
                et = np.stack([event_type_ids[e - bptt + 1 : e + 1] for e in idx], axis=0)
                batch["event_type_ids"] = to_tensor(et.astype(np.int64))
            if delta_t is not None:
                dt = np.stack([delta_t[e - bptt + 1 : e + 1] for e in idx], axis=0)
                batch["delta_t"] = to_tensor(dt.astype(np.float32))
            if sizes is not None:
                sz = np.stack([sizes[e - bptt + 1 : e + 1] for e in idx], axis=0)
                batch["sizes"] = to_tensor(sz.astype(np.float32))
            if market_ids is not None:
                mids = np.asarray([market_ids[e] for e in idx], dtype=np.int64)
                batch["market_ids"] = to_tensor(mids)
            yield batch


def _predict_sequence(
    model,
    X: np.ndarray,
    Z: np.ndarray,
    device: torch.device,
    market_id: int | None = None,
    temperature: float | None = None,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        xb = torch.from_numpy(X)[None, ...].to(dev)
        zb = torch.from_numpy(Z)[None, ...].to(dev)
        extra = {}
        if market_id is not None:
            extra["market_ids"] = torch.tensor([market_id], dtype=torch.long, device=dev)
        out = model(xb, zb, **extra)
        logits = out.instability_logits[0].detach().cpu().numpy().reshape(-1)
        if temperature is not None and float(temperature) > 0:
            logits = logits / float(temperature)
        return 1.0 / (1.0 + np.exp(-logits))


def _predict_logits_and_intensities(
    model, X: np.ndarray, Z: np.ndarray, device: torch.device, market_id: int | None = None
):
    """Return (logits[T], intensities[T,M]) for one sequence."""
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        xb = torch.from_numpy(X)[None, ...].to(dev)
        zb = torch.from_numpy(Z)[None, ...].to(dev)
        extra = {}
        if market_id is not None:
            extra["market_ids"] = torch.tensor([market_id], dtype=torch.long, device=dev)
        out = model(xb, zb, **extra)
        logits = out.instability_logits[0].detach().cpu().numpy().reshape(-1)
        heads = [out.intensities[f"event_{i}"] for i in range(len(out.intensities))]
        lam = (
            torch.cat(heads, dim=-1)[0].detach().cpu().numpy()
            if heads
            else np.zeros((len(logits), 0), dtype=np.float32)
        )
        return logits, lam


def main():
    if torch is None:
        raise SystemExit("PyTorch is required for training CLI: pip install torch")
    ap = argparse.ArgumentParser(description="Train hybrid model with TBPTT and balanced windows.")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--label-key", required=True)
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bptt", type=int, default=64)
    ap.add_argument("--topo-stride", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--folds", type=int, default=1, help="Walk-forward folds (k-step)")
    ap.add_argument("--strict-tda", action="store_true", help="Fail if TDA backends missing")
    ap.add_argument(
        "--topology-json",
        type=Path,
        default=None,
        help="Optional TopologyConfig JSON (e.g., output of topo_search: topology_selected.json)",
    )
    ap.add_argument(
        "--use-topo-selected",
        action="store_true",
        help="Load topology_selected.json from artifact-dir (or ./artifacts/topo/<instrument>/)",
    )
    ap.add_argument(
        "--write-topology-schema",
        action="store_true",
        help="Force-embed the active topology into feature_schema.json even if it exists",
    )
    ap.add_argument(
        "--beta", type=float, default=1e-4, help="Smoothness penalty weight for intensities"
    )
    ap.add_argument(
        "--smoothness-norm",
        type=str,
        choices=["none", "global", "per_seq"],
        default="global",
        help="Normalization for intensity smoothness penalty",
    )
    ap.add_argument(
        "--temperature-scale",
        action="store_true",
        help="Fit temperature on validation and apply to test predictions",
    )
    ap.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    ap.add_argument(
        "--tpp-diagnostics", action="store_true", help="Save TPP arrays and diagnostics for test"
    )
    ap.add_argument(
        "--include-market-embedding",
        action="store_true",
        help="Include learned market embedding (single-market id supported)",
    )
    ap.add_argument(
        "--market-vocab-size",
        type=int,
        default=None,
        help="Market vocabulary size when using market embeddings",
    )
    ap.add_argument("--market-id", type=int, default=0, help="Market id for this instrument")
    ap.add_argument(
        "--warm-start", type=Path, default=None, help="Optional checkpoint (.pt) to warm-start"
    )
    ap.add_argument(
        "--expand-types-by-level",
        action="store_true",
        help="Expand LO/CX event types by level when a 'level' column exists",
    )
    ap.add_argument(
        "--print-types-info", action="store_true", help="Print inferred num_event_types and exit"
    )
    args = ap.parse_args()

    data = DataConfig(
        raw_data_root=Path("."),
        cache_root=Path("."),
        instruments=[args.instrument],
        expand_event_types_by_level=bool(args.expand_types_by_level),
    )
    try:
        import random as _random

        _random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        if torch is not None:
            torch.manual_seed(int(args.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    builder = LOBDatasetBuilder(data)
    scaler_path = args.artifact_dir / "scaler_schema.json"
    feat_schema_path = args.artifact_dir / "feature_schema.json"
    topo_cfg = None
    if feat_schema_path.exists():
        schema = json.loads(feat_schema_path.read_text())
        try:
            topo_cfg = TopologyConfig(**schema.get("topology", {}))
        except Exception:
            topo_cfg = TopologyConfig()

    # Allow strict TDA toggle on config
    if topo_cfg is None:
        topo_cfg = TopologyConfig()
    # Override via topology JSON selection if requested
    if args.topology_json is not None and Path(args.topology_json).exists():
        try:
            obj = json.loads(Path(args.topology_json).read_text())
            topo_cfg = TopologyConfig(**obj)
        except Exception as e:
            print(f"[warn] failed to load topology-json: {e}")
    elif bool(args.use_topo_selected):
        cand = args.artifact_dir / "topology_selected.json"
        if not cand.exists():
            alt = Path("artifacts") / "topo" / args.instrument / "topology_selected.json"
            cand = alt if alt.exists() else cand
        if cand.exists():
            try:
                obj = json.loads(cand.read_text())
                topo_cfg = TopologyConfig(**obj)
                print(f"[info] loaded topology from {cand}")
            except Exception as e:
                print(f"[warn] failed to load topology from {cand}: {e}")
    if bool(args.strict_tda):
        try:
            topo_cfg.strict_tda = True  # type: ignore[attr-defined]
        except Exception:
            pass

    folds = max(1, int(args.folds))
    if folds == 1:
        train, val, test, scaler = builder.build_splits(
            args.instrument,
            label_key=args.label_key,
            topology=topo_cfg,
            topo_stride=args.topo_stride,
            artifact_dir=args.artifact_dir,
        )
        datasets = [(train, val, test)]
    else:
        wf = builder.build_walkforward_splits(
            args.instrument,
            label_key=args.label_key,
            topology=topo_cfg,
            topo_stride=args.topo_stride,
            folds=folds,
            artifact_dir=args.artifact_dir,
        )
        datasets = [(tr, va, te) for (tr, va, te, _sc) in wf]

    if args.print_types_info:

        def _infer_num_event_types(*splits):
            mx = 0
            found = False
            for s in splits:
                et = s.get("event_type_ids")
                if et is not None and len(et) > 0:
                    mx = max(mx, int(np.max(et)))
                    found = True
            return (mx + 1) if found else 6

        num_event_types = _infer_num_event_types(train, val, test)
        info = {"num_event_types": int(num_event_types)}
        print(json.dumps(info))
        return

    if scaler_path.exists():
        scaler = SplitSafeStandardScaler.load_schema(str(scaler_path))
    try:
        from dataclasses import asdict
        if not feat_schema_path.exists():
            schema = {
                "instrument": args.instrument,
                "topology": asdict(topo_cfg if topo_cfg is not None else TopologyConfig()),
            }
            with open(feat_schema_path, "w") as f:
                json.dump(schema, f, indent=2)
        else:
            obj = json.loads(feat_schema_path.read_text())
            if args.write_topology_schema or ("topology" not in obj):
                obj["topology"] = asdict(topo_cfg if topo_cfg is not None else TopologyConfig())
                with open(feat_schema_path, "w") as f:
                    json.dump(obj, f, indent=2)
    except Exception:
        pass

    from ..models.hybrid import HybridLOBModel
    from ..training.losses import HybridLossComputer
    from ..training.pipeline import TrainingPipeline

    F = datasets[0][0]["features"].shape[1]
    Z = datasets[0][0]["topology"].shape[1]

    def _infer_num_event_types(*splits):
        mx = 0
        found = False
        for s in splits:
            et = s.get("event_type_ids")
            if et is not None and len(et) > 0:
                mx = max(mx, int(np.max(et)))
                found = True
        return (mx + 1) if found else 6

    num_event_types = _infer_num_event_types(train, val, test)
    cfg = ModelConfig(
        hidden_size=args.hidden,
        num_layers=args.layers,
        include_market_embedding=bool(args.include_market_embedding),
        market_vocab_size=(int(args.market_vocab_size) if args.market_vocab_size is not None else None),
    )
    model = HybridLOBModel(F, Z, num_event_types=int(num_event_types), config=cfg)
    if args.warm_start:
        try:
            state = torch.load(args.warm_start, map_location=args.device)
            sd = state.get("state_dict", state)
            model.load_state_dict(sd, strict=False)
        except Exception as e:  # pragma: no cover
            print(f"[warn] failed to load warm-start checkpoint: {e}")
    tcfg = TrainingConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        gradient_clipping=1.0,
        max_epochs=args.epochs,
        patience=max(2, args.epochs),
        mixed_precision=False,
        apply_temperature_scaling=bool(args.temperature_scale),
        bptt_steps=args.bptt,
    )
    exp = ExperimentConfig(data=data, model=cfg, training=tcfg)
    loss = HybridLossComputer(
        lambda_cls=1.0,
        beta=float(args.beta),
        gamma=1e-4,
        cls_loss_type=("focal" if args.focal else "bce"),
        pos_weight=args.pos_weight,
        smoothness_norm=str(args.smoothness_norm),
    )
    pipe = TrainingPipeline(exp, model, loss)

    def to_loader(
        split: Dict[str, np.ndarray], balanced: bool
    ) -> Iterable[Dict[str, torch.Tensor]]:
        mids = None
        if bool(args.include_market_embedding):
            mids = np.full((len(split["labels"]),), int(args.market_id), dtype=np.int64)
        return _window_batches(
            split["features"],
            split["topology"],
            split["labels"],
            bptt=args.bptt,
            batch_size=args.batch,
            balanced=balanced,
            event_type_ids=split.get("event_type_ids"),
            delta_t=split.get("delta_t"),
            sizes=split.get("sizes"),
            market_ids=(split.get("market_ids") if split.get("market_ids") is not None else mids),
        )

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    model.to(device)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for i, (train, val, test) in enumerate(datasets, start=1):
        fold_dir = args.artifact_dir if folds == 1 else (args.artifact_dir / f"fold_{i}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        metrics = pipe.fit(to_loader(train, balanced=True), to_loader(val, balanced=False))
        # Per-fold prediction writing (with optional temperature scaling)
        Tcal = None
        if bool(args.temperature_scale):
            with torch.no_grad():
                xb = torch.from_numpy(val["features"]).unsqueeze(0).to(device)
                zb = torch.from_numpy(val["topology"]).unsqueeze(0).to(device)
                out = model(xb, zb)
                z_val = out.instability_logits[0].detach().cpu().numpy().reshape(-1)
            y_val = val["labels"].astype(int).reshape(-1)
            scaler = TemperatureScaler()
            try:
                Tcal = float(scaler.fit(z_val, y_val))
                if folds == 1:
                    with open(args.artifact_dir / "temperature.json", "w") as f:
                        json.dump({"temperature": Tcal}, f, indent=2)
            except Exception:
                Tcal = None
        for name, split in ("val", val), ("test", test):
            p = _predict_sequence(
                model,
                split["features"],
                split["topology"],
                device,
                temperature=(Tcal if (name == "test" and Tcal is not None) else None),
            )
            y = split["labels"].astype(int)
            out = np.vstack([np.arange(len(y)), p, y]).T
            np.savetxt(
                (fold_dir / f"predictions_{name}.csv"),
                out,
                delimiter=",",
                header="idx,pred,label",
                comments="",
            )

    try:
        meta = {
            "beta": float(args.beta),
            "seed": int(args.seed),
        }
        if isinstance(metrics, dict) and "temperature" in metrics:
            meta["temperature"] = float(metrics["temperature"])
        with open(args.artifact_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    if bool(args.tpp_diagnostics):
        try:
            logits_te, lam_te = _predict_logits_and_intensities(
                model, test["features"], test["topology"], device
            )
            et = test.get("event_type_ids")
            dt = test.get("delta_t")
            if et is not None and dt is not None and lam_te.size:
                np.savez_compressed(
                    args.artifact_dir / "tpp_test_arrays.npz",
                    intensities=lam_te.astype(np.float32),
                    event_type_ids=et.astype(np.int64),
                    delta_t=dt.astype(np.float32),
                )
                from ..evaluation.metrics import compute_point_process_diagnostics
                arr = TPPArrays(intensities=lam_te, event_type_ids=et, delta_t=dt)
                xi = rescaled_times(arr)
                emp, mod = model_and_empirical_frequencies(arr)
                nll_evt = nll_per_event_from_arrays(lam_te, et, dt)
                diag_pp = compute_point_process_diagnostics(xi, empirical_frequencies=emp, model_frequencies=mod)
                diag = {
                    "nll_per_event": float(nll_evt),
                    "ks_p_value": float(diag_pp.ks_p_value),
                    "coverage_error": float(diag_pp.coverage_error),
                }
                with open(args.artifact_dir / "tpp_test_diagnostics.json", "w") as f:
                    json.dump(diag, f, indent=2)
        except Exception:
            pass

    # Write TDA backend info for appendix/repro
    try:
        def _check_mod(name: str):
            try:
                mod = __import__(name)
                ver = None
                for key in ("__version__", "version", "__VERSION__"):
                    if hasattr(mod, key):
                        ver = getattr(mod, key)
                        break
                return {"available": True, "version": (ver if isinstance(ver, (str, float, int)) else str(ver))}
            except Exception:
                return {"available": False, "version": None}

        tda = {"ripser": _check_mod("ripser"), "gudhi": _check_mod("gudhi"), "persim": _check_mod("persim")}
        with open(args.artifact_dir / "tda_backends.json", "w") as f:
            json.dump(tda, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
