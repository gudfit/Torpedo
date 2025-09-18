"""Batch training CLI: iterate over instruments, train, and store eval JSONs.

Discovers instruments from cache-root (*.parquet) by default, or accepts explicit list.
Stores artifacts per instrument under --artifact-root/<instrument>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

from ..config import DataConfig, ModelConfig, TrainingConfig, TopologyConfig, ExperimentConfig
from ..data.loader import LOBDatasetBuilder
from . import train as train_cli
from ..evaluation.metrics import compute_classification_metrics, delong_ci_auroc
from dataclasses import asdict
from ..config import TopologyConfig


def _discover_instruments(cache_root: Path) -> List[str]:
    return [p.stem for p in sorted(cache_root.glob("*.parquet"))]


def _predict_probs(
    model: HybridLOBModel, X: np.ndarray, Z: np.ndarray, device: torch.device
) -> np.ndarray:
    return train_cli._predict_sequence(model, X, Z, device)


def main():
    if torch is None:
        raise SystemExit("PyTorch is required: pip install torch")
    ap = argparse.ArgumentParser(description="Batch train hybrid model across instruments")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--instruments", type=str, nargs="*")
    g_lbl = ap.add_mutually_exclusive_group(required=True)
    g_lbl.add_argument("--label-key", type=str)
    g_lbl.add_argument(
        "--label-keys",
        type=str,
        nargs="*",
        help="Train/evaluate multiple horizons (e.g., instability_s_1 instability_e_100)",
    )
    ap.add_argument("--topo-stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bptt", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    ap.add_argument("--beta", type=float, default=1e-4, help="Smoothness penalty weight for intensities")
    ap.add_argument(
        "--smoothness-norm",
        type=str,
        choices=["none", "global", "per_seq"],
        default="global",
        help="Normalization for intensity smoothness penalty",
    )
    ap.add_argument("--temperature-scale", action="store_true", help="Fit temperature on validation and apply to test predictions")
    ap.add_argument("--tpp-diagnostics", action="store_true", help="Save TPP arrays and diagnostics for test")
    ap.add_argument("--strict-tda", action="store_true", help="Fail if TDA backends missing")
    ap.add_argument("--topology-json", type=Path, default=None, help="Optional TopologyConfig JSON to apply")
    ap.add_argument("--use-topo-selected", action="store_true", help="Load topology_selected.json per instrument from artifact-root/<inst>/ or ./artifacts/topo/<inst>/")
    ap.add_argument(
        "--warm-start", type=Path, default=None, help="Optional checkpoint (.pt) to warm-start"
    )
    ap.add_argument(
        "--mode", choices=["per_instrument", "pooled", "lomo"], default="per_instrument"
    )
    ap.add_argument(
        "--expand-types-by-level",
        action="store_true",
        help="Expand LO/CX event types by level when a 'level' column exists",
    )
    ap.add_argument(
        "--print-types-info",
        action="store_true",
        help="Print inferred num_event_types per instrument (or pooled) and exit",
    )
    ap.add_argument(
        "--include-market-embedding",
        action="store_true",
        help="Include learned market embedding in per_instrument mode as well",
    )
    args = ap.parse_args()

    try:
        import random as _random

        _random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        try:
            import torch as _t

            _t.manual_seed(int(args.seed))
            if _t.cuda.is_available():
                _t.cuda.manual_seed_all(int(args.seed))
            try:
                _t.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                _t.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    instruments = args.instruments or _discover_instruments(args.cache_root)
    if not instruments:
        raise SystemExit(
            "No instruments found; provide --instruments or ensure *.parquet present under cache-root"
        )

    data = DataConfig(
        raw_data_root=args.cache_root,
        cache_root=args.cache_root,
        instruments=instruments,
        expand_event_types_by_level=bool(args.expand_types_by_level),
    )
    builder = LOBDatasetBuilder(data)

    # Helper to resolve a TopologyConfig for an instrument
    import json as _json_topo

    _topo_global = None
    if args.topology_json is not None and Path(args.topology_json).exists():
        try:
            _topo_global = TopologyConfig(**_json_topo.loads(Path(args.topology_json).read_text()))
        except Exception:
            _topo_global = None

    def _topo_for_inst(inst: str) -> TopologyConfig:
        if _topo_global is not None:
            return _topo_global
        if args.use_topo_selected:
            cand = args.artifact_root / inst / "topology_selected.json"
            if not cand.exists():
                alt = Path("artifacts") / "topo" / inst / "topology_selected.json"
                cand = alt if alt.exists() else cand
            if cand.exists():
                try:
                    return TopologyConfig(**_json_topo.loads(cand.read_text()))
                except Exception:
                    pass
        return TopologyConfig(strict_tda=bool(args.strict_tda))

    if args.print_types_info:

        def infer_num_types_for_instrument(inst: str) -> int:
            rec = builder.build_sequence(inst)
            et = rec.get("event_type_ids")
            if et is None or len(et) == 0:
                return 6
            return int(np.max(et)) + 1

        info = {}
        if args.mode == "per_instrument":
            for inst in instruments:
                info[inst] = infer_num_types_for_instrument(inst)
        elif args.mode == "pooled":
            mx = 0
            for inst in instruments:
                mx = max(mx, infer_num_types_for_instrument(inst))
            info["POOLED"] = mx
        else:  # LOMO
            for inst in instruments:
                mx = 0
                for other in instruments:
                    if other == inst:
                        continue
                    mx = max(mx, infer_num_types_for_instrument(other))
                info[f"LOMO_{inst}"] = mx
        import json

        print(json.dumps(info))
        return

    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    label_keys = args.label_keys if args.label_keys else [args.label_key]

    from ..evaluation.calibration import TemperatureScaler as _Temp
    from ..evaluation.tpp import (
        TPPArrays as _TPPA,
        rescaled_times as _xi,
        model_and_empirical_frequencies as _freqs,
        nll_per_event_from_arrays as _nll_evt,
    )

    def _train_one(
        dataset, inst_name: str, lbl: str, market_vocab_size: int | None, market_id: int | None
    ):
        from ..models.hybrid import HybridLOBModel
        from ..training.losses import HybridLossComputer
        from ..training.pipeline import TrainingPipeline

        train, val, test = dataset
        F = train["features"].shape[1]
        Z = train["topology"].shape[1]

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
            include_market_embedding=(market_vocab_size is not None),
            market_vocab_size=market_vocab_size,
        )
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
        model = HybridLOBModel(F, Z, num_event_types=int(num_event_types), config=cfg).to(dev)
        assert len(model.intensity_heads) == int(num_event_types)
        assert len(model.mark_heads) == int(num_event_types)
        if args.warm_start:
            try:
                state = torch.load(args.warm_start, map_location=dev)
                sd = state.get("state_dict", state)
                model.load_state_dict(sd, strict=False)
            except Exception as e:  # pragma: no cover
                print(f"[warn] failed to load warm-start checkpoint: {e}")
        loss = HybridLossComputer(lambda_cls=1.0, beta=float(args.beta), gamma=1e-4, smoothness_norm=str(args.smoothness_norm))
        pipe = TrainingPipeline(exp, model, loss)

        train_loader = train_cli._window_batches(
            train["features"],
            train["topology"],
            train["labels"],
            bptt=args.bptt,
            batch_size=args.batch,
            balanced=True,
            event_type_ids=train.get("event_type_ids"),
            delta_t=train.get("delta_t"),
            sizes=train.get("sizes"),
            market_ids=train.get("market_ids"),
        )
        val_loader = train_cli._window_batches(
            val["features"],
            val["topology"],
            val["labels"],
            bptt=args.bptt,
            batch_size=args.batch,
            balanced=False,
            event_type_ids=val.get("event_type_ids"),
            delta_t=val.get("delta_t"),
            sizes=val.get("sizes"),
            market_ids=val.get("market_ids"),
        )
        metrics = pipe.fit(train_loader, val_loader)

        # Optional temperature calibration on validation logits
        Tcal = None
        if bool(args.temperature_scale):
            with torch.no_grad():
                xb = torch.from_numpy(val["features"]).unsqueeze(0).to(dev)
                zb = torch.from_numpy(val["topology"]).unsqueeze(0).to(dev)
                out = model(xb, zb)
                z_val = out.instability_logits[0].detach().cpu().numpy().reshape(-1)
            y_val = val["labels"].astype(int).reshape(-1)
            tsc = _Temp()
            try:
                Tcal = float(tsc.fit(z_val, y_val))
            except Exception:
                Tcal = None

        p = train_cli._predict_sequence(
            model,
            test["features"],
            test["topology"],
            dev,
            market_id=market_id,
            temperature=Tcal if Tcal is not None else None,
        )
        y = test["labels"].astype(int)
        art = args.artifact_root / inst_name / lbl
        art.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            art / "predictions_test.csv",
            np.vstack([np.arange(len(y)), p, y]).T,
            delimiter=",",
            header="idx,pred,label",
            comments="",
        )
        m = compute_classification_metrics(p, y)
        auc, lo, hi = delong_ci_auroc(p, y)
        out = {
            "auroc": auc,
            "auroc_ci": [lo, hi],
            "auprc": m.auprc,
            "brier": m.brier,
            "ece": m.ece,
            "instrument": inst_name,
            "label_key": lbl,
        }
        import json as _json

        with open(art / "eval_test.json", "w") as f:
            _json.dump(out, f, indent=2)

        # Persist meta (beta, seed, temperature)
        try:
            meta = {"beta": float(args.beta), "seed": int(args.seed)}
            if Tcal is not None:
                meta["temperature"] = float(Tcal)
            with open(art / "training_meta.json", "w") as f:
                _json.dump(meta, f, indent=2)
            if Tcal is not None:
                with open(art / "temperature.json", "w") as f:
                    _json.dump({"temperature": float(Tcal)}, f, indent=2)
        except Exception:
            pass

        # Optional: TPP diagnostics for test split
        if bool(args.tpp_diagnostics):
            try:
                with torch.no_grad():
                    xb = torch.from_numpy(test["features"]).unsqueeze(0).to(dev)
                    zb = torch.from_numpy(test["topology"]).unsqueeze(0).to(dev)
                    out = model(xb, zb)
                    heads = [out.intensities[f"event_{i}"] for i in range(len(out.intensities))]
                    lam = (
                        torch.cat(heads, dim=-1)[0].detach().cpu().numpy() if heads else np.zeros((len(test["labels"]), 0), dtype=np.float32)
                    )
                et = test.get("event_type_ids")
                dt = test.get("delta_t")
                if et is not None and dt is not None and lam.size:
                    np.savez_compressed(
                        art / "tpp_test_arrays.npz",
                        intensities=lam.astype(np.float32),
                        event_type_ids=et.astype(np.int64),
                        delta_t=dt.astype(np.float32),
                    )
                    from ..evaluation.metrics import compute_point_process_diagnostics as _ppdiag
                    arr = _TPPA(intensities=lam, event_type_ids=et, delta_t=dt)
                    xi = _xi(arr)
                    emp, mod = _freqs(arr)
                    nll_evt = _nll_evt(lam, et, dt)
                    d = _ppdiag(xi, empirical_frequencies=emp, model_frequencies=mod)
                    diag = {
                        "nll_per_event": float(nll_evt),
                        "ks_p_value": float(d.ks_p_value),
                        "coverage_error": float(d.coverage_error),
                    }
                    with open(art / "tpp_test_diagnostics.json", "w") as f:
                        _json.dump(diag, f, indent=2)
            except Exception:
                pass

        # Persist TDA backend info for appendix/repro
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
            with open(art / "tda_backends.json", "w") as f:
                _json.dump(tda, f, indent=2)
        except Exception:
            pass

    if args.mode == "per_instrument":
        for inst in instruments:
            print(f"Training {inst}")
            for lbl in label_keys:
                ds = builder.build_splits(
                    inst,
                    label_key=lbl,
                    topology=_topo_for_inst(inst),
                    topo_stride=args.topo_stride,
                    artifact_dir=args.artifact_root / inst / lbl,
                )
                if args.include_market_embedding:
                    # Attach single-market ids (0) and set vocab size to 1
                    for s in ds[:3]:
                        s["market_ids"] = np.zeros((len(s["labels"]),), dtype=np.int64)
                    _train_one(ds[:3], inst, lbl, market_vocab_size=1, market_id=0)
                else:
                    _train_one(ds[:3], inst, lbl, market_vocab_size=None, market_id=None)
    elif args.mode == "pooled":
        market_vocab = len(instruments)
        # Resolve pooled topology config: prefer global or first instrument's selected
        topo = _topo_global if _topo_global is not None else None
        if topo is None and bool(args.use_topo_selected) and instruments:
            cand = args.artifact_root / instruments[0] / "topology_selected.json"
            if not cand.exists():
                alt = Path("artifacts") / "topo" / instruments[0] / "topology_selected.json"
                cand = alt if alt.exists() else cand
            if cand.exists():
                try:
                    topo = TopologyConfig(**_json_topo.loads(cand.read_text()))
                except Exception:
                    topo = None
        if topo is None:
            topo = TopologyConfig(strict_tda=bool(args.strict_tda))
        for lbl in label_keys:
            print(f"Training pooled for {lbl}")
            seqs = []
            for k, inst in enumerate(instruments):
                rec = builder.build_sequence(inst)
                T = len(rec["timestamps"])
                t0 = int(0.6 * T)
                v0 = int(0.8 * T)
                seqs.append((inst, rec, t0, v0, k))
            from ..utils.scaler import SplitSafeStandardScaler as _Scaler

            scaler = _Scaler()
            Xtrain_all = np.concatenate([s[1]["features_raw"][: s[2]] for s in seqs], axis=0)
            scaler.fit(Xtrain_all, feature_names=seqs[0][1]["feature_names"].tolist())
            from ..features.topological import TopologicalFeatureGenerator as _Topo

            topo_gen = _Topo(topo)

            def _build(rec, t0, v0):
                X = scaler.transform(rec["features_raw"])

                def build_slice(a, b):
                    ts = rec["timestamps"][a:b]
                    Z = topo_gen.rolling_transform(ts, X[a:b], stride=args.topo_stride)
                    out = {
                        "features": X[a:b].astype(np.float32),
                        "topology": Z.astype(np.float32),
                        "labels": rec["labels"][lbl][a:b].astype(np.int64),
                        "delta_t": rec["delta_t"][a:b].astype(np.float32),
                    }
                    if rec["event_type_ids"] is not None:
                        out["event_type_ids"] = rec["event_type_ids"][a:b].astype(np.int64)
                    if rec["sizes"] is not None:
                        out["sizes"] = rec["sizes"][a:b].astype(np.float32)
                    return out

                return build_slice(0, t0), build_slice(t0, v0), build_slice(v0, len(X))

            trains = []
            vals = []
            tests = []
            market_ids = []
            for inst, rec, t0, v0, k in seqs:
                tr, va, te = _build(rec, t0, v0)
                for s in (tr, va, te):
                    s["market_ids"] = np.full((len(s["labels"]),), k, dtype=np.int64)
                trains.append(tr)
                vals.append(va)
                tests.append(te)
                market_ids.append(k)

            def cat(list_of_dicts):
                keys = list(list_of_dicts[0].keys())
                return {k: np.concatenate([d[k] for d in list_of_dicts], axis=0) for k in keys}

            train = cat(trains)
            val = cat(vals)
            test = cat(tests)
            _train_one(
                (train, val, test), "POOLED", lbl, market_vocab_size=market_vocab, market_id=None
            )
            art = args.artifact_root / "POOLED" / lbl
            art.mkdir(parents=True, exist_ok=True)
            topo_schema = {"topology": asdict(TopologyConfig(strict_tda=bool(args.strict_tda)))}
            import json

            with open(art / "feature_schema.json", "w") as f:
                json.dump(topo_schema, f, indent=2)
    else:
        for lbl in label_keys:
            for holdout_idx, inst in enumerate(instruments):
                print(f"LOMO: hold out {inst} for {lbl}")
                topo = _topo_for_inst(inst)
                seqs = []
                for k, other in enumerate(instruments):
                    rec = builder.build_sequence(other)
                    T = len(rec["timestamps"])
                    t0 = int(0.6 * T)
                    v0 = int(0.8 * T)
                    seqs.append((other, rec, t0, v0, k))
                from ..utils.scaler import SplitSafeStandardScaler as _Scaler

                scaler = _Scaler()
                Xtrain_all = np.concatenate(
                    [s[1]["features_raw"][: s[2]] for s in seqs if s[0] != inst], axis=0
                )
                scaler.fit(Xtrain_all, feature_names=seqs[0][1]["feature_names"].tolist())
                from ..features.topological import TopologicalFeatureGenerator as _Topo

                topo_gen = _Topo(topo)

                def _build(rec, t0, v0):
                    X = scaler.transform(rec["features_raw"])

                    def build_slice(a, b):
                        ts = rec["timestamps"][a:b]
                        Z = topo_gen.rolling_transform(ts, X[a:b], stride=args.topo_stride)
                        out = {
                            "features": X[a:b].astype(np.float32),
                            "topology": Z.astype(np.float32),
                            "labels": rec["labels"][lbl][a:b].astype(np.int64),
                            "delta_t": rec["delta_t"][a:b].astype(np.float32),
                        }
                        if rec["event_type_ids"] is not None:
                            out["event_type_ids"] = rec["event_type_ids"][a:b].astype(np.int64)
                        if rec["sizes"] is not None:
                            out["sizes"] = rec["sizes"][a:b].astype(np.float32)
                        return out

                    return build_slice(0, t0), build_slice(t0, v0), build_slice(v0, len(X))

                trains = []
                vals = []
                for other, rec, t0, v0, k in seqs:
                    if other == inst:
                        continue
                    tr, va, _ = _build(rec, t0, v0)
                    tr["market_ids"] = np.full((len(tr["labels"]),), k, dtype=np.int64)
                    va["market_ids"] = np.full((len(va["labels"]),), k, dtype=np.int64)
                    trains.append(tr)
                    vals.append(va)

                def cat(list_of_dicts):
                    keys = list(list_of_dicts[0].keys())
                    return {k: np.concatenate([d[k] for d in list_of_dicts], axis=0) for k in keys}

                train = cat(trains)
                val = cat(vals)
                for other, rec, t0, v0, k in seqs:
                    if other == inst:
                        _tr, _va, test = _build(rec, t0, v0)
                        break
                _train_one(
                    (train, val, test),
                    f"LOMO_{inst}",
                    lbl,
                    market_vocab_size=len(instruments),
                    market_id=holdout_idx,
                )
                art = args.artifact_root / f"LOMO_{inst}" / lbl
                art.mkdir(parents=True, exist_ok=True)
                topo_schema = {"topology": asdict(TopologyConfig(strict_tda=bool(args.strict_tda)))}
                import json

                with open(art / "feature_schema.json", "w") as f:
                    json.dump(topo_schema, f, indent=2)

    print("Batch training complete")


if __name__ == "__main__":
    main()
