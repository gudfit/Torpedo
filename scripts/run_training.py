#!/usr/bin/env python3
"""Entry point for running TorpedoCode experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torpedocode import config as config_module
from torpedocode.cuda import load_extension
from torpedocode.models import HybridLOBModel
from torpedocode.training import HybridLossComputer, TrainingPipeline
from torpedocode.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TorpedoCode training pipeline")
    parser.add_argument("config", type=Path, help="Path to JSON or YAML experiment config")
    parser.add_argument("--log", type=Path, default=None, help="Optional log file")
    parser.add_argument("--cuda", action="store_true", help="Force loading the CUDA extension")
    return parser.parse_args()


def load_config(path: Path) -> config_module.ExperimentConfig:
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    return config_module.load_from_dict(data)


def main() -> None:
    args = parse_args()
    configure_logging(args.log)

    experiment_config = load_config(args.config)

    feature_dim = experiment_config.data.levels * 2
    topo_dim = 16
    num_event_types = len(list(experiment_config.data.event_types))

    model = HybridLOBModel(
        feature_dim=feature_dim,
        topo_dim=topo_dim,
        num_event_types=num_event_types,
        config=experiment_config.model,
    )
    loss = HybridLossComputer(
        lambda_cls=experiment_config.model.classifier_weight,
        beta=experiment_config.model.intensity_smoothness_penalty,
        gamma=experiment_config.model.weight_decay,
        cls_loss_type=experiment_config.model.classifier_loss_type,
        focal_gamma=experiment_config.model.focal_gamma,
    )

    if args.cuda and experiment_config.training.use_cuda_extension:
        load_extension(experiment_config.training.cuda_extension_name)

    pipeline = TrainingPipeline(experiment_config, model, loss)

    dummy_features = torch.zeros((1, 1, feature_dim))
    dummy_topology = torch.zeros((1, 1, topo_dim))
    dummy_labels = torch.zeros((1, 1))
    batch = {
        "features": dummy_features,
        "topology": dummy_topology,
        "instability_labels": dummy_labels,
    }
    metrics = pipeline.fit(train_loader=[batch])
    print(metrics)


if __name__ == "__main__":
    main()
