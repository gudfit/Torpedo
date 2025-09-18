"""Hybrid recurrent model combining classical and topological features."""

from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..utils.ops import hybrid_fuse, has_torpedocode_op


@dataclass
class HybridModelOutputs:
    """Container for the heads of the hybrid model."""

    intensities: Dict[str, torch.Tensor]
    mark_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    instability_logits: torch.Tensor


class HybridLOBModel(nn.Module):
    """LSTM-based hybrid architecture with temporal point process and classifier heads."""

    def __init__(self, feature_dim: int, topo_dim: int, num_event_types: int, config: ModelConfig):
        super().__init__()
        self.config = config
        env_policy = os.environ.get("TORPEDOCODE_USE_FUSE", "auto").lower()
        verbose = os.environ.get("TORPEDOCODE_VERBOSE", "0").lower() in {"1", "true"}
        if env_policy in {"1", "true", "force"}:
            use_fuse = True
            if verbose:
                if has_torpedocode_op():
                    warnings.warn("Using native fuse op (enabled by TORPEDOCODE_USE_FUSE)")
                else:
                    warnings.warn(
                        "TORPEDOCODE_USE_FUSE set but op not found; falling back to PyTorch"
                    )
        elif env_policy in {"0", "false", "off"}:
            use_fuse = False
            if verbose:
                warnings.warn("Native fuse disabled by TORPEDOCODE_USE_FUSE")
        else:
            use_fuse = bool(config.use_native_fuse or has_torpedocode_op())
            if verbose and use_fuse and has_torpedocode_op():
                warnings.warn("Using native fuse op (auto)")
        self.use_native_fuse = use_fuse
        lstm_input_dim = feature_dim + topo_dim
        self.market_embedding_dim = 32
        self.market_vocab_size = getattr(config, "market_vocab_size", None)
        if config.include_market_embedding:
            if self.market_vocab_size is not None and self.market_vocab_size > 0:
                self.market_embedding_table = nn.Embedding(
                    self.market_vocab_size, self.market_embedding_dim
                )
            else:
                self.market_embedding_table = None
            lstm_input_dim += self.market_embedding_dim
        else:
            self.market_embedding_table = None

        if self.use_native_fuse:
            lstm_input_dim += feature_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.intensity_heads = nn.ModuleDict(
            {f"event_{idx}": nn.Linear(config.hidden_size, 1) for idx in range(num_event_types)}
        )
        self.topology_skip = nn.ModuleDict(
            {f"event_{idx}": nn.Linear(topo_dim, 1, bias=False) for idx in range(num_event_types)}
        )
        self.mark_heads = nn.ModuleDict(
            {
                f"event_{idx}": nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size // 2, 2),
                )
                for idx in range(num_event_types)
            }
        )
        self.instability_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        features: torch.Tensor,
        topology: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        market_embedding: Optional[torch.Tensor] = None,
        market_ids: Optional[torch.Tensor] = None,
    ) -> HybridModelOutputs:
        """Run the hybrid model on a batch of sequences."""

        if self.config.include_market_embedding:
            if self.market_embedding_table is not None and market_ids is not None:
                if market_ids.dim() == 1:
                    me = self.market_embedding_table(market_ids)
                    repeated_embed = me.unsqueeze(1).expand(-1, features.shape[1], -1)
                else:
                    me = self.market_embedding_table(market_ids)
                    repeated_embed = me
            else:
                emb = (
                    market_embedding
                    if market_embedding is not None
                    else torch.zeros(
                        features.shape[0], self.market_embedding_dim, device=features.device
                    )
                )
                repeated_embed = emb.unsqueeze(1).expand(-1, features.shape[1], -1)
            fcat_list = [features, topology, repeated_embed]
            if self.use_native_fuse:
                fused = hybrid_fuse(features, topology)
                fcat_list.insert(0, fused)
            features = torch.cat(fcat_list, dim=-1)
        else:
            fcat_list = [features, topology]
            if self.use_native_fuse:
                fused = hybrid_fuse(features, topology)
                fcat_list.insert(0, fused)
            features = torch.cat(fcat_list, dim=-1)

        packed_output, _ = self.lstm(features)
        if mask is not None:
            packed_output = packed_output * mask.unsqueeze(-1)

        intensities = {}
        mark_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, head in self.intensity_heads.items():
            raw = head(packed_output)
            skip = self.topology_skip[name](topology)
            intensities[name] = F.softplus(raw + skip)
            mu_sigma = self.mark_heads[name](packed_output)
            mu, log_sigma = mu_sigma.chunk(2, dim=-1)
            mark_params[name] = (mu, log_sigma)

        logits = self.instability_head(packed_output)

        return HybridModelOutputs(
            intensities=intensities,
            mark_params=mark_params,
            instability_logits=logits,
        )

    def forward_with_state(
        self,
        features: torch.Tensor,
        topology: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        market_embedding: Optional[torch.Tensor] = None,
        market_ids: Optional[torch.Tensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[HybridModelOutputs, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass that accepts and returns LSTM hidden state (for TBPTT)."""

        if self.config.include_market_embedding:
            if self.market_embedding_table is not None and market_ids is not None:
                if market_ids.dim() == 1:
                    me = self.market_embedding_table(market_ids)
                    repeated_embed = me.unsqueeze(1).expand(-1, features.shape[1], -1)
                else:
                    repeated_embed = self.market_embedding_table(market_ids)
            else:
                emb = (
                    market_embedding
                    if market_embedding is not None
                    else torch.zeros(
                        features.shape[0], self.market_embedding_dim, device=features.device
                    )
                )
                repeated_embed = emb.unsqueeze(1).expand(-1, features.shape[1], -1)
            fcat_list = [features, topology, repeated_embed]
            if self.use_native_fuse:
                fused = hybrid_fuse(features, topology)
                fcat_list.insert(0, fused)
            fcat = torch.cat(fcat_list, dim=-1)
        else:
            fcat_list = [features, topology]
            if self.use_native_fuse:
                fused = hybrid_fuse(features, topology)
                fcat_list.insert(0, fused)
            fcat = torch.cat(fcat_list, dim=-1)

        output, hx = self.lstm(fcat, hx)
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        intensities = {}
        mark_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, head in self.intensity_heads.items():
            raw = head(output)
            skip = self.topology_skip[name](topology)
            intensities[name] = F.softplus(raw + skip)
            mu_sigma = self.mark_heads[name](output)
            mu, log_sigma = mu_sigma.chunk(2, dim=-1)
            mark_params[name] = (mu, log_sigma)

        logits = self.instability_head(output)
        return (
            HybridModelOutputs(
                intensities=intensities, mark_params=mark_params, instability_logits=logits
            ),
            hx,
        )


__all__ = ["HybridLOBModel", "HybridModelOutputs"]
