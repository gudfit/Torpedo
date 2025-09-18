"""CLI: Pretrain the hybrid model on synthetic marked CTMC sequences.

Generates batches from a simple CTMC generator to warm-start the hybrid
model's recurrent state and TPP+mark heads. Saves a checkpoint that can be
loaded before fine-tuning on historical data.

Example:
  python -m torpedocode.cli.pretrain_ctmc \
    --epochs 5 --steps 500 --batch 64 --T 128 --num-event-types 6 \
    --hidden 128 --layers 1 --lr 3e-4 --device cpu \
    --output artifacts/pretrained/model.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

from ..models.hybrid import HybridLOBModel
from ..training.losses import HybridLossComputer
from ..config import ModelConfig
from ..utils.checkpoint import save_checkpoint
from ..data.synthetic_ctmc import CTMCConfig, generate_ctmc_sequence


def _make_batch(
    *,
    batch_size: int,
    T: int,
    num_event_types: int,
    feature_dim: int,
    topo_dim: int,
    feature_state_scale: float,
    feature_noise_sigma: float,
    topo_window: int,
    topo_levels: int,
    emit_topology: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    feats = []
    topos = []
    etypes = []
    dt = []
    sizes = []
    for _ in range(batch_size):
        cfg = CTMCConfig(
            T=T,
            num_event_types=num_event_types,
            feature_dim=feature_dim,
            feature_state_scale=float(feature_state_scale),
            feature_noise_sigma=float(feature_noise_sigma),
            topo_window=int(topo_window),
            topo_levels=int(topo_levels),
            emit_topology=bool(emit_topology),
        )
        rec = generate_ctmc_sequence(cfg, rng)
        # rec["features"] is white-noise with shape [T, M] by default; adapt to feature_dim
        X = rec["features"].astype(np.float32)
        if X.shape[1] != feature_dim:
            if X.shape[1] == 0:
                X = rng.normal(size=(T, feature_dim)).astype(np.float32)
            else:
                W = np.eye(X.shape[1], feature_dim, dtype=np.float32)[: X.shape[1]]
                X = (
                    (X @ W)
                    if X.shape[1] >= feature_dim
                    else np.pad(X, ((0, 0), (0, feature_dim - X.shape[1])))
                )
        feats.append(X)
        etypes.append(rec["event_type_ids"].astype(np.int64))
        dt.append(rec["delta_t"].astype(np.float32))
        sizes.append(rec["sizes"].astype(np.float32))
        # Optional synthetic topology from generator
        topo_rec = rec.get("topology", None)
        if topo_rec is None:
            topo_rec = np.zeros((T, max(1, topo_dim)), dtype=np.float32)
        else:
            # match requested topo_dim by padding or projecting
            Zr = topo_rec.shape[1]
            if Zr == topo_dim:
                pass
            elif Zr < topo_dim:
                topo_rec = np.pad(topo_rec, ((0, 0), (0, topo_dim - Zr)))
            else:
                Wp = np.eye(Zr, topo_dim, dtype=np.float32)
                topo_rec = topo_rec @ Wp
        topos.append(topo_rec.astype(np.float32))

    features = np.stack(feats, axis=0)
    if len(topos) > 0:
        topology = np.stack(topos, axis=0)
    else:
        Zdim = max(1, int(topo_dim))
        topology = np.zeros((batch_size, T, Zdim), dtype=np.float32)
    event_type_ids = np.stack(etypes, axis=0)
    delta_t = np.stack(dt, axis=0)
    sizes = np.stack(sizes, axis=0)
    return {
        "features": features,
        "topology": topology,
        "event_type_ids": event_type_ids,
        "delta_t": delta_t,
        "sizes": sizes,
    }


def main() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required: pip install torch")

    ap = argparse.ArgumentParser(description="Pretrain hybrid model on synthetic CTMC sequences")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500, help="Steps per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=128, help="Events per synthetic sequence")
    ap.add_argument("--num-event-types", type=int, default=6)
    ap.add_argument("--feature-dim", type=int, default=None)
    ap.add_argument("--ctmc-feature-scale", type=float, default=1.0, help="State-dependent feature mean scale")
    ap.add_argument("--ctmc-feature-noise", type=float, default=0.5, help="Feature emission noise sigma")
    ap.add_argument("--ctmc-topo-window", type=int, default=16, help="Synthetic topology window length")
    ap.add_argument("--ctmc-topo-levels", type=int, default=3, help="Synthetic topology levels")
    ap.add_argument("--no-ctmc-topo", action="store_true", help="Disable synthetic topology emission")
    ap.add_argument("--topo-dim", type=int, default=0, help="Topology feature dim (zeros)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    F = int(args.feature_dim) if args.feature_dim is not None else int(args.num_event_types)
    Z = max(1, int(args.topo_dim))

    dev = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    model_cfg = ModelConfig(
        hidden_size=args.hidden, num_layers=args.layers, include_market_embedding=False
    )
    model = HybridLOBModel(F, Z, num_event_types=int(args.num_event_types), config=model_cfg).to(
        dev
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    loss_comp = HybridLossComputer(
        lambda_cls=0.0,
        beta=model_cfg.intensity_smoothness_penalty,
        gamma=model_cfg.weight_decay,
        cls_loss_type="bce",
    )

    model.train()
    for epoch in range(int(args.epochs)):
        last_loss = None
        for _ in range(int(args.steps)):
            batch_np = _make_batch(
                batch_size=int(args.batch),
                T=int(args.T),
                num_event_types=int(args.num_event_types),
                feature_dim=F,
                topo_dim=Z,
                feature_state_scale=float(args.ctmc_feature_scale),
                feature_noise_sigma=float(args.ctmc_feature_noise),
                topo_window=int(args.ctmc_topo_window),
                topo_levels=int(args.ctmc_topo_levels),
                emit_topology=not bool(args.no_ctmc_topo),
                rng=rng,
            )
            batch = {k: torch.from_numpy(v).to(dev) for k, v in batch_np.items()}
            out = model(batch["features"], batch["topology"])
            lo = loss_comp(out, batch, list(model.parameters()))
            opt.zero_grad()
            lo.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(lo.total.detach().cpu())
        print({"epoch": epoch + 1, "train_loss": last_loss})

    save_checkpoint(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "pretrain": {
                "epochs": int(args.epochs),
                "steps": int(args.steps),
                "batch": int(args.batch),
                "T": int(args.T),
                "num_event_types": int(args.num_event_types),
                "feature_dim": int(F),
                "topo_dim": int(Z),
                "lr": float(args.lr),
                "seed": int(args.seed),
            },
        },
        args.output,
    )
    print(f"Saved CTMC-pretrained checkpoint to {args.output}")


if __name__ == "__main__":
    main()
