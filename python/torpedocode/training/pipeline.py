"""High-level experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from ..config import ExperimentConfig
from ..models import HybridLOBModel
from .losses import HybridLossComputer
from ..evaluation.calibration import TemperatureScaler
from ..evaluation.metrics import compute_classification_metrics


@dataclass
class TrainingState:
    """Mutable state tracked throughout training."""

    epoch: int = 0
    best_metric: float = float("inf")
    patience_counter: int = 0


class TrainingPipeline:
    """Coordinate data loading, optimisation, and evaluation loops."""

    def __init__(
        self, config: ExperimentConfig, model: HybridLOBModel, loss: HybridLossComputer
    ) -> None:
        self.config = config
        self.model = model
        self.loss = loss

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """Run the full training loop with early stopping."""

        device = (
            next(self.model.parameters()).device
            if any(p.requires_grad for p in self.model.parameters())
            else torch.device("cpu")
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=0.0,
        )

        try:
            if hasattr(self.loss, "gamma"):
                wd_nonzero = any(
                    getattr(g, "weight_decay", 0.0) not in (0, 0.0) for g in optimizer.param_groups
                )
                if wd_nonzero and float(getattr(self.loss, "gamma", 0.0)) > 0:
                    import warnings as _warnings

                    _warnings.warn(
                        "Optimizer has non-zero weight_decay while loss applies L2 via gamma; this will double-count."
                    )
        except Exception:
            pass

        # Optional progress bars via tqdm if available and enabled by env
        _use_bar = str(os.environ.get("TORPEDOCODE_PROGRESS", "0")).lower() in {"1", "true"}
        _tqdm = None
        if _use_bar:
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
            except Exception:
                _tqdm = None

        state = TrainingState()
        epoch_bar = None
        if _tqdm is not None:
            epoch_bar = _tqdm(total=int(self.config.training.max_epochs), desc="train", leave=False)

        while state.epoch < self.config.training.max_epochs:
            self.model.train()
            last_loss = None
            loader = train_loader
            batch_bar = None
            if _tqdm is not None:
                try:
                    total_batches = len(loader)  # type: ignore[arg-type]
                except Exception:
                    total_batches = None
                batch_bar = _tqdm(total=total_batches, desc=f"e{state.epoch+1}", leave=False)
                try:
                    loader = iter(loader)
                except Exception:
                    pass
            for batch in (loader if _tqdm is None else iter(train_loader)):
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                if self.config.training.bptt_steps and batch["features"].ndim == 3:
                    T = batch["features"].shape[1]
                    step = int(self.config.training.bptt_steps)
                    hx = None
                    for t0 in range(0, T, step):
                        t1 = min(T, t0 + step)
                        sl = slice(t0, t1)
                        sub = {
                            k: (v[:, sl] if isinstance(v, torch.Tensor) and v.ndim == 3 else v)
                            for k, v in batch.items()
                        }
                        outputs, hx = self.model.forward_with_state(
                            sub["features"],
                            sub["topology"],
                            sub.get("mask"),
                            market_ids=sub.get("market_ids"),
                            hx=hx,
                        )
                        loss_outputs = self.loss(outputs, sub, list(self.model.parameters()))
                        optimizer.zero_grad()
                        loss_outputs.total.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.gradient_clipping
                        )
                        optimizer.step()
                        if isinstance(hx, tuple):
                            hx = (hx[0].detach(), hx[1].detach())
                else:
                    outputs = self.model(
                        batch["features"],
                        batch["topology"],
                        batch.get("mask"),
                        market_ids=batch.get("market_ids"),
                    )
                    loss_outputs = self.loss(outputs, batch, list(self.model.parameters()))
                    optimizer.zero_grad()
                    loss_outputs.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clipping
                    )
                    optimizer.step()
                last_loss = loss_outputs
                if batch_bar is not None:
                    try:
                        batch_bar.update(1)
                    except Exception:
                        pass
            if last_loss is None:
                # Be tolerant in tiny/demo or edge splits: allow training to proceed
                # without batches so downstream steps (eval/predictions) can run.
                import warnings as _warnings
                _warnings.warn(
                    "Training loader produced no batches; skipping optimisation this epoch."
                )
                return {"train_loss": float("nan")}

            metrics = {"train_loss": float(last_loss.total.detach().cpu())}
            if val_loader is not None:
                metrics.update(self.evaluate(val_loader, device))

            state.epoch += 1
            if epoch_bar is not None:
                try:
                    epoch_bar.update(1)
                except Exception:
                    pass

            if val_loader is None:
                continue
            current_metric = metrics.get("val_loss", float("inf"))
            if current_metric < state.best_metric:
                state.best_metric = current_metric
                state.patience_counter = 0
            else:
                state.patience_counter += 1
                if state.patience_counter >= self.config.training.patience:
                    break

        if epoch_bar is not None:
            try:
                epoch_bar.close()
            except Exception:
                pass
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: Iterable[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate the model on a validation or test loader."""

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        logits_all = []
        labels_all = []
        for batch in data_loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            outputs = self.model(batch["features"], batch["topology"], batch.get("mask"))
            loss_outputs = self.loss(outputs, batch, list(self.model.parameters()))
            total_loss += float(loss_outputs.total.detach().cpu())
            num_batches += 1
            if "instability_labels" in batch:
                logits_all.append(outputs.instability_logits.detach().cpu().numpy().reshape(-1))
                labels_all.append(batch["instability_labels"].detach().cpu().numpy().reshape(-1))

        metrics = {"val_loss": total_loss / max(num_batches, 1)}
        if logits_all:
            z = np.concatenate(logits_all)
            y = np.concatenate(labels_all)
            p = 1.0 / (1.0 + np.exp(-z))
            m = compute_classification_metrics(p, y)
            metrics.update(
                {
                    "val_brier": float(m.brier),
                    "val_ece": float(m.ece),
                    "val_auroc": float(m.auroc),
                    "val_auprc": float(m.auprc),
                }
            )
        if self.config.training.apply_temperature_scaling and logits_all:
            z = np.concatenate(logits_all)
            y = np.concatenate(labels_all)
            scaler = TemperatureScaler()
            T = scaler.fit(z, y)
            z_cal = scaler.transform(z)
            p = 1.0 / (1.0 + np.exp(-z_cal))
            eps = 1e-12
            bce = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            mcal = compute_classification_metrics(p, y)
            metrics.update(
                {
                    "temperature": float(T),
                    "val_bce_calibrated": bce,
                    "val_brier_calibrated": float(mcal.brier),
                    "val_ece_calibrated": float(mcal.ece),
                }
            )
        return metrics


__all__ = ["TrainingPipeline", "TrainingState"]
