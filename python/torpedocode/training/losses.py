"""Loss functions mirroring the methodology section."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..utils.ops import has_torpedocode_op


def _maybe_native_tpp_loss(
    intensities: torch.Tensor,
    mark_mu: torch.Tensor,
    mark_log_sigma: torch.Tensor,
    event_types: torch.Tensor,
    delta_t: torch.Tensor,
    sizes: torch.Tensor | None,
    mask: torch.Tensor | None,
    smoothness_norm: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        if not has_torpedocode_op():
            return None
    except Exception:
        return None

    mode_map = {"none": 0, "global": 1, "per_seq": 2}
    mode = mode_map.get(str(smoothness_norm).lower(), 1)
    try:
        out = torch.ops.torpedocode.tpp_loss(
            intensities,
            mark_mu,
            mark_log_sigma,
            event_types,
            delta_t,
            sizes if sizes is not None else None,
            mask if mask is not None else None,
            mode,
        )
    except (AttributeError, RuntimeError, TypeError):
        return None

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]
    return None


@dataclass
class LossOutputs:
    """Aggregated loss terms for logging."""

    total: torch.Tensor
    tpp_mark: torch.Tensor
    classification: torch.Tensor
    smoothness: torch.Tensor
    weight_decay: torch.Tensor


class HybridLossComputer:
    """Compose the joint training objective from model outputs."""

    def __init__(
        self,
        lambda_cls: float,
        beta: float,
        gamma: float,
        cls_loss_type: str = "bce",
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
        smoothness_norm: str = "global",
    ) -> None:
        self.lambda_cls = lambda_cls
        self.beta = beta
        self.gamma = gamma
        self.cls_loss_type = cls_loss_type
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        # smoothness_norm: "none" (sum over all), "global" (current default), "per_seq" (divide by per-sequence pairs)
        self.smoothness_norm = smoothness_norm

    def __call__(
        self,
        outputs,
        batch,
        parameters,
    ) -> LossOutputs:
        """Compute all loss components for a training batch."""

        device = batch["features"].device
        tpp_mark_loss = torch.zeros((), device=device)
        cls_loss = torch.zeros((), device=device)
        smoothness = torch.zeros((), device=device)
        wd = torch.zeros((), device=device)

        if "instability_labels" in batch:
            logits = outputs.instability_logits
            target = batch["instability_labels"].unsqueeze(-1).float()
            if self.cls_loss_type == "bce":
                pos_w = (
                    None
                    if self.pos_weight is None
                    else torch.tensor(self.pos_weight, device=device)
                )
                cls_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_w)
            elif self.cls_loss_type == "focal":
                p = torch.sigmoid(logits)
                eps = 1e-12
                ce = -(target * torch.log(p + eps) + (1 - target) * torch.log(1 - p + eps))
                pt = target * p + (1 - target) * (1 - p)
                cls_loss = ((1 - pt).pow(self.focal_gamma) * ce).mean()
            else:
                raise ValueError(f"Unknown classification loss type: {self.cls_loss_type}")

        # FIX: temporal point process + mark likelihood (only if data present)
        # Expected batch keys:
        # - event_type_ids: LongTensor[B, T] in [0, M-1]
        # - delta_t: FloatTensor[B, T] inter-event times (seconds)
        # - sizes: FloatTensor[B, T] positive marks (optional)
        if (
            "event_type_ids" in batch
            and "delta_t" in batch
            and outputs.intensities
            and outputs.mark_params
        ):
            heads = [
                outputs.intensities[f"event_{i}"] for i in range(len(outputs.intensities))
            ]
            lamb = torch.cat(heads, dim=-1).clamp_min(1e-12)
            device = lamb.device
            dtype = lamb.dtype
            etypes = batch["event_type_ids"].to(device=device, dtype=torch.long)
            dt = batch["delta_t"].to(device=device, dtype=dtype)
            event_mask = (etypes >= 0).to(dtype=dtype)
            mus, log_sigs = zip(
                *[outputs.mark_params[f"event_{i}"] for i in range(len(outputs.mark_params))]
            )
            mark_mu = torch.cat(mus, dim=-1).to(device=device, dtype=dtype)
            mark_log_sig = torch.cat(log_sigs, dim=-1).to(device=device, dtype=dtype)
            sizes_tensor = None
            if "sizes" in batch and batch["sizes"] is not None:
                sizes_tensor = batch["sizes"].to(device=device, dtype=dtype)
            mask_tensor = None
            if isinstance(batch, dict) and "mask" in batch and batch["mask"] is not None:
                mask_tensor = batch["mask"].to(device=device, dtype=dtype)

            native = _maybe_native_tpp_loss(
                lamb,
                mark_mu,
                mark_log_sig,
                etypes,
                dt,
                sizes_tensor,
                mask_tensor,
                self.smoothness_norm,
            )
            if native is not None:
                tpp_mark_loss, smoothness = native
            else:
                gather_idx = etypes.clamp_min(0).unsqueeze(-1)
                lamb_evt = torch.gather(lamb, dim=-1, index=gather_idx).squeeze(-1)
                log_lamb_evt = torch.log(lamb_evt) * event_mask
                comp = (lamb.sum(dim=-1) * dt).sum(dim=1)
                mark_nll = torch.zeros_like(comp)
                if sizes_tensor is not None:
                    log_sizes = torch.log(sizes_tensor.clamp_min(1e-12))
                    mu_evt = torch.gather(mark_mu, dim=-1, index=gather_idx).squeeze(-1)
                    log_sig_evt = torch.gather(mark_log_sig, dim=-1, index=gather_idx).squeeze(-1)
                    z = (log_sizes - mu_evt) / (torch.exp(log_sig_evt) + 1e-12)
                    const = 0.5 * torch.log(
                        torch.tensor(2 * 3.1415926535, device=device, dtype=dtype)
                    )
                    mark_nll = (
                        (0.5 * z.pow(2) + log_sig_evt + log_sizes + const) * event_mask
                    ).sum(dim=1)

                nll = -(log_lamb_evt.sum(dim=1)) + comp + mark_nll
                tpp_mark_loss = nll.mean()
                dl = torch.diff(lamb, dim=1)
                if mask_tensor is not None:
                    pair_mask = (mask_tensor[:, 1:] * mask_tensor[:, :-1]).unsqueeze(-1)
                    if self.smoothness_norm == "none":
                        smoothness = (dl.pow(2) * pair_mask).sum()
                    elif self.smoothness_norm == "per_seq":
                        per_seq = (dl.pow(2) * pair_mask).sum(dim=(1, 2))
                        pairs = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
                        smoothness = (per_seq / pairs).mean()
                    else:
                        denom = pair_mask.sum().clamp_min(1.0)
                        smoothness = (dl.pow(2) * pair_mask).sum() / denom
                else:
                    if self.smoothness_norm == "none":
                        smoothness = dl.pow(2).sum()
                    elif self.smoothness_norm == "per_seq":
                        per_seq = dl.pow(2).sum(dim=(1, 2))
                        pairs = (
                            torch.tensor(
                                dl.shape[1] * dl.shape[2],
                                device=dl.device,
                                dtype=dl.dtype,
                            ).clamp_min(1.0)
                        )
                        smoothness = (per_seq / pairs).mean()
                    else:
                        smoothness = (dl.pow(2).sum(dim=(-1, -2))).mean()

        if self.gamma and self.gamma > 0:
            sqsum = torch.zeros((), device=device)
            for p in parameters:
                if p.requires_grad:
                    sqsum = sqsum + p.pow(2).sum()
            wd = sqsum

        total = (
            tpp_mark_loss + self.lambda_cls * cls_loss + self.beta * smoothness + self.gamma * wd
        )
        return LossOutputs(
            total=total,
            tpp_mark=tpp_mark_loss,
            classification=cls_loss,
            smoothness=smoothness,
            weight_decay=wd,
        )


__all__ = ["HybridLossComputer", "LossOutputs"]
