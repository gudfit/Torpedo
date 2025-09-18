"""Configuration schemas for TorpedoCode experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal


@dataclass(slots=True)
class DataConfig:
    """Data ingestion and preprocessing options."""

    raw_data_root: Path
    cache_root: Path
    instruments: List[str]
    levels: int = 10
    expand_event_types_by_level: bool = False
    event_types: Iterable[str] = field(
        default_factory=lambda: ["MO+", "MO-", "LO+", "LO-", "CX+", "CX-"]
    )
    forecast_horizons_s: Iterable[int] = field(default_factory=lambda: [1, 5, 10])
    forecast_horizons_events: Iterable[int] = field(default_factory=lambda: [100, 500])
    time_zone: str = "UTC"
    # Market-local time zone for session hours (e.g., "America/New_York", "Europe/London")
    session_time_zone: str = "America/New_York"
    normalise_prices_to_ticks: bool = True
    drop_auctions: bool = True
    itch_spec: str | None = None
    ouch_spec: str | None = None
    corporate_actions_csv: Path | None = None
    corporate_actions_date_col: str = "date"
    corporate_actions_symbol_col: str = "symbol"
    corporate_actions_factor_col: str = "adj_factor"
    corporate_actions_mode: Literal["multiply", "divide"] = "divide"
    instability_threshold_eta: float = 0.0


@dataclass(slots=True)
class TopologyConfig:
    """Persistent homology extraction parameters."""

    window_sizes_s: Iterable[int] = field(default_factory=lambda: [1, 5, 10])
    complex_type: Literal["cubical", "vietoris_rips"] = "cubical"
    max_homology_dimension: int = 1
    persistence_representation: Literal["landscape", "image"] = "landscape"
    landscape_levels: int = 5
    image_resolution: int = 64
    image_bandwidth: float = 0.05
    strict_tda: bool = False
    vr_auto_epsilon: bool = True
    vr_connectivity_quantile: float = 0.99
    vr_epsilon_rule: Literal["mst_quantile", "largest_cc"] = "largest_cc"
    vr_lcc_threshold: float = 0.99
    vr_lcc_grid_size: int = 25
    use_liquidity_surface: bool = True
    levels_hint: int | None = None
    imbalance_eps: float = 1e-6
    use_raw_liquidity_surface: bool = True
    use_raw_for_vr: bool = False


@dataclass(slots=True)
class ModelConfig:
    """Configuration of the hybrid neural architecture."""

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    include_market_embedding: bool = True
    market_vocab_size: int | None = None
    intensity_smoothness_penalty: float = 1e-4
    classifier_weight: float = 1.0
    weight_decay: float = 1e-4
    classifier_loss_type: Literal["bce", "focal"] = "bce"
    focal_gamma: float = 2.0
    use_native_fuse: bool = False


@dataclass(slots=True)
class TrainingConfig:
    """End-to-end training parameters."""

    batch_size: int = 256
    learning_rate: float = 3e-4
    gradient_clipping: float = 1.0
    max_epochs: int = 50
    patience: int = 5
    mixed_precision: bool = True
    use_cuda_extension: bool = True
    cuda_extension_name: str = "torpedocode_kernels"
    apply_temperature_scaling: bool = False
    bptt_steps: int | None = None
    balanced_minibatching: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    """Composite configuration object consumed by training scripts."""

    data: DataConfig
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_root: Path = Path("./artifacts")
    seed: int = 7


def load_from_dict(config_dict: Dict) -> ExperimentConfig:
    """Instantiate an :class:`ExperimentConfig` from a raw dictionary."""

    data = DataConfig(**config_dict["data"])
    topology = TopologyConfig(**config_dict.get("topology", {}))
    model = ModelConfig(**config_dict.get("model", {}))
    training = TrainingConfig(**config_dict.get("training", {}))
    output_root = Path(config_dict.get("output_root", "./artifacts"))
    seed = int(config_dict.get("seed", 7))

    return ExperimentConfig(
        data=data,
        topology=topology,
        model=model,
        training=training,
        output_root=output_root,
        seed=seed,
    )


__all__ = [
    "DataConfig",
    "TopologyConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "load_from_dict",
]
