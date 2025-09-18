"""CLI: Interpretability utilities (permutation importance, optional SHAP).

Loads a trained hybrid model checkpoint (state dict) and computes permutation
importance for feature groups (conventional vs TDA) by shuffling columns.
If `shap` is installed and `--shap` is passed, attempts KernelExplainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import torch

from ..models.hybrid import HybridLOBModel
from ..config import ModelConfig, DataConfig, TopologyConfig
from ..data.loader import LOBDatasetBuilder
from ..evaluation.metrics import compute_classification_metrics


def _predict(
    model: HybridLOBModel, X: np.ndarray, Z: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X)[None, ...].to(device)
        zb = torch.from_numpy(Z)[None, ...].to(device)
        out = model(xb, zb)
        logits = out.instability_logits[0].detach().cpu().numpy().reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))


def main():
    ap = argparse.ArgumentParser(
        description="Permutation importance and optional SHAP for hybrid model"
    )
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--instrument", type=str, required=True)
    ap.add_argument("--label-key", type=str, required=True)
    ap.add_argument(
        "--state-dict", type=Path, required=False, help="Optional PyTorch state_dict path (.pt)"
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--strict-tda", action="store_true", help="Fail if TDA backends missing")
    ap.add_argument(
        "--shap", action="store_true", help="Compute SHAP KernelExplainer on last-step features"
    )
    ap.add_argument("--shap-samples", type=int, default=32)
    ap.add_argument(
        "--bptt", type=int, default=64, help="Sequence window for SHAP last-step context"
    )
    args = ap.parse_args()

    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    data = DataConfig(
        raw_data_root=args.cache_root, cache_root=args.cache_root, instruments=[args.instrument]
    )
    builder = LOBDatasetBuilder(data)
    topo = TopologyConfig(strict_tda=bool(args.strict_tda))
    tr, va, te, _ = builder.build_splits(
        args.instrument, label_key=args.label_key, topology=topo, topo_stride=5, artifact_dir=None
    )
    F = tr["features"].shape[1]
    Z = tr["topology"].shape[1]
    model = HybridLOBModel(
        F,
        Z,
        num_event_types=6,
        config=ModelConfig(hidden_size=64, num_layers=1, include_market_embedding=False),
    ).to(dev)
    if args.state_dict and args.state_dict.exists():
        state = torch.load(args.state_dict, map_location=dev)
        model.load_state_dict(state)

    p0 = _predict(model, te["features"], te["topology"], dev)
    m0 = compute_classification_metrics(p0, te["labels"].astype(int))

    rng = np.random.default_rng(0)
    Xp = te["features"].copy()
    for j in range(Xp.shape[1]):
        rng.shuffle(Xp[:, j])
    pp = _predict(model, Xp, te["topology"], dev)
    mp = compute_classification_metrics(pp, te["labels"].astype(int))

    Zp = te["topology"].copy()
    for j in range(Zp.shape[1]):
        rng.shuffle(Zp[:, j])
    pz = _predict(model, te["features"], Zp, dev)
    mz = compute_classification_metrics(pz, te["labels"].astype(int))

    out = {
        "baseline": {"auroc": m0.auroc, "auprc": m0.auprc},
        "permute_features": {
            "auroc": mp.auroc,
            "auprc": mp.auprc,
            "delta_auroc": mp.auroc - m0.auroc,
        },
        "permute_topology": {
            "auroc": mz.auroc,
            "auprc": mz.auprc,
            "delta_auroc": mz.auroc - m0.auroc,
        },
    }
    if args.shap:
        try:
            import shap  # type: ignore

            Fdim, Zdim = te["features"].shape[1], te["topology"].shape[1]
            T = te["features"].shape[0]
            W = min(int(args.bptt), T)
            idx = np.linspace(W - 1, T - 1, num=min(int(args.shap_samples), T), dtype=int)

            X_last = te["features"][idx]
            Z_last = te["topology"][idx]
            bg = np.hstack([X_last[: min(8, len(idx))], Z_last[: min(8, len(idx))]])

            def f(xcat: np.ndarray) -> np.ndarray:
                Xc = xcat[:, :Fdim]
                Zc = xcat[:, Fdim : Fdim + Zdim]
                outs = []
                with torch.no_grad():
                    for k in range(Xc.shape[0]):
                        e = int(idx[min(k, len(idx) - 1)])
                        s = e - W + 1
                        Xw = te["features"][s : e + 1].copy()
                        Zw = te["topology"][s : e + 1].copy()
                        Xw[-1] = Xc[k]
                        Zw[-1] = Zc[k]
                        xb = torch.from_numpy(Xw)[None, ...].to(dev)
                        zb = torch.from_numpy(Zw)[None, ...].to(dev)
                        pr = torch.sigmoid(model(xb, zb).instability_logits[0, -1, 0]).item()
                        outs.append(pr)
                return np.array(outs, dtype=float)

            explainer = shap.KernelExplainer(f, bg)
            x_eval = np.hstack([X_last, Z_last])
            shap_vals = explainer.shap_values(x_eval, nsamples=64)
            sv = np.abs(np.array(shap_vals))
            feat_contrib = float(np.mean(np.sum(sv[:, :Fdim], axis=1))) if sv.size else 0.0
            topo_contrib = (
                float(np.mean(np.sum(sv[:, Fdim : Fdim + Zdim], axis=1))) if sv.size else 0.0
            )
            out["shap_summary"] = {
                "mean_abs_sum_features": feat_contrib,
                "mean_abs_sum_topology": topo_contrib,
            }
        except Exception as e:
            out["shap_summary_error"] = str(e)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
