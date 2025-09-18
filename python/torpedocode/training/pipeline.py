"""High-level experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from ..config import ExperimentConfig
from ..models import HybridLOBModel
from .losses import HybridLossComputer
from ..evaluation.calibration import TemperatureScaler


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

        # Respect the model's current device; don't force to CUDA automatically.
        device = next(self.model.parameters()).device if any(p.requires_grad for p in self.model.parameters()) else torch.device("cpu")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=0.0,
        )

        state = TrainingState()
        while state.epoch < self.config.training.max_epochs:
            self.model.train()
            last_loss = None
            for batch in train_loader:
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
            if last_loss is None:
                raise RuntimeError(
                    "Training loader produced no batches. Provide at least one batch."
                )

            metrics = {"train_loss": float(last_loss.total.detach().cpu())}
            if val_loader is not None:
                metrics.update(self.evaluate(val_loader, device))

            state.epoch += 1

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
            if self.config.training.apply_temperature_scaling and "instability_labels" in batch:
                logits_all.append(outputs.instability_logits.detach().cpu().numpy().reshape(-1))
                labels_all.append(batch["instability_labels"].detach().cpu().numpy().reshape(-1))

        metrics = {"val_loss": total_loss / max(num_batches, 1)}
        if self.config.training.apply_temperature_scaling and logits_all:
            z = np.concatenate(logits_all)
            y = np.concatenate(labels_all)
            scaler = TemperatureScaler()
            T = scaler.fit(z, y)
            z_cal = scaler.transform(z)
            p = 1.0 / (1.0 + np.exp(-z_cal))
            eps = 1e-12
            bce = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            metrics.update({"temperature": float(T), "val_bce_calibrated": bce})
        return metrics


__all__ = ["TrainingPipeline", "TrainingState"]
