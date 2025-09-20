"""Manifest-based runner: cache -> train -> eval -> aggregate from a YAML/JSON spec.

Manifest schema (YAML or JSON):

data:
  input: path/to/ndjson_or_dir
  cache_root: path/to/caches
  per_file: false            # if true, one cache per file; else merged to instrument
  instrument: AAPL           # required if merging multiple files into one cache
  drop_auctions: true
  tick_size: 0.01
  price_scale: 1.0

train:
  artifact_root: path/to/artifacts
  label_key: instability_s_5
  epochs: 2
  batch: 128
  bptt: 64
  topo_stride: 5
  device: cpu

aggregate:
  mode: pred                 # pred or eval; pred uses predictions_* CSVs
  output: path/to/aggregate.json
  block_bootstrap: true
  block_length: 50
  n_boot: 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_manifest(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise SystemExit("PyYAML not installed. Provide a JSON manifest or install pyyaml.")
    return yaml.safe_load(path.read_text())


def main():
    ap = argparse.ArgumentParser(description="Run pipeline from a manifest")
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    spec = _load_manifest(args.manifest)

    from . import cache as cache_cli

    data = spec.get("data", {})
    input_path = Path(data["input"]) if "input" in data else None
    cache_root = Path(data["cache_root"]) if "cache_root" in data else None
    if input_path is None or cache_root is None:
        raise SystemExit("Manifest must include data.input and data.cache_root")

    argv = [
        "prog",
        "--input",
        str(input_path),
        "--cache-root",
        str(cache_root),
    ]
    if data.get("per_file", False):
        argv.append("--per-file")
    if data.get("instrument"):
        argv += ["--instrument", str(data["instrument"])]
    if data.get("drop_auctions", False):
        argv.append("--drop-auctions")
    if data.get("tick_size") is not None:
        argv += ["--tick-size", str(data["tick_size"])]
    if data.get("price_scale") is not None:
        argv += ["--price-scale", str(data["price_scale"])]
    if data.get("levels") is not None:
        argv += ["--levels", str(int(data["levels"]))]
    if data.get("horizons_s"):
        argv += ["--horizons-s", *[str(x) for x in data["horizons_s"]]]
    if data.get("horizons_events"):
        argv += ["--horizons-events", *[str(x) for x in data["horizons_events"]]]
    if data.get("eta") is not None:
        argv += ["--eta", str(float(data["eta"]))]
    # Optional: honor market session time zone for auction/halts filtering
    if data.get("session_tz") is not None:
        argv += ["--session-tz", str(data["session_tz"])]

    import sys

    _old = sys.argv
    try:
        sys.argv = argv
        cache_cli.main()
    finally:
        sys.argv = _old

    train = spec.get("train", {})
    if train:
        from . import batch_train as btrain

        argv = [
            "prog",
            "--cache-root",
            str(cache_root),
            "--artifact-root",
            (
                str(Path(train["artifact_root"]))
                if "artifact_root" in train
                else str(Path("artifacts"))
            ),
            "--label-key",
            str(train.get("label_key", "instability_s_5")),
            "--epochs",
            str(int(train.get("epochs", 2))),
            "--batch",
            str(int(train.get("batch", 128))),
            "--bptt",
            str(int(train.get("bptt", 64))),
            "--topo-stride",
            str(int(train.get("topo_stride", 5))),
            "--hidden",
            str(int(train.get("hidden", 128))),
            "--layers",
            str(int(train.get("layers", 1))),
            "--lr",
            str(float(train.get("lr", 3e-4))),
            "--device",
            str(train.get("device", "cpu")),
        ]
        # pass-through optional flags
        if train.get("seed") is not None:
            argv += ["--seed", str(int(train.get("seed")))]
        if train.get("beta") is not None:
            argv += ["--beta", str(float(train.get("beta")))]
        if bool(train.get("temperature_scale", False)):
            argv += ["--temperature-scale"]
        if bool(train.get("tpp_diagnostics", False)):
            argv += ["--tpp-diagnostics"]
        if bool(train.get("strict_tda", False)):
            argv += ["--strict-tda"]
        if bool(train.get("include_market_embedding", False)):
            argv += ["--include-market-embedding"]
        # Topology controls (optional): forward to batch_train
        if bool(train.get("use_topo_selected", False)):
            argv += ["--use-topo-selected"]
        if train.get("topology_json") is not None:
            argv += ["--topology-json", str(train.get("topology_json"))]
        if train.get("pi_res") is not None:
            argv += ["--pi-res", str(int(train.get("pi_res")))]
        if train.get("pi_sigma") is not None:
            argv += ["--pi-sigma", str(float(train.get("pi_sigma")))]
        if bool(train.get("expand_types_by_level", False)):
            argv += ["--expand-types-by-level"]
        if train.get("warm_start"):
            argv += ["--warm-start", str(Path(train["warm_start"]))]
        try:
            sys.argv = argv
            btrain.main()
        finally:
            sys.argv = _old

    agg = spec.get("aggregate", {})
    if agg:
        from . import aggregate as agg_cli

        mode = agg.get("mode", "pred").lower()
        if mode == "eval":
            argv = [
                "prog",
                "--root",
                str(Path(train.get("artifact_root", "artifacts"))),
                "--pattern",
                str(agg.get("pattern", "*/eval_test.json")),
                "--output",
                str(Path(agg.get("output", "aggregate.json"))),
            ]
        else:
            argv = [
                "prog",
                "--pred-root",
                str(Path(train.get("artifact_root", "artifacts"))),
                "--pred-pattern",
                str(agg.get("pred_pattern", "*/predictions_test.csv")),
                "--output",
                str(Path(agg.get("output", "aggregate.json"))),
            ]
            if agg.get("block_bootstrap", False):
                argv += [
                    "--block-bootstrap",
                    "--block-length",
                    str(float(agg.get("block_length", 50))),
                    "--n-boot",
                    str(int(agg.get("n_boot", 200))),
                ]

        try:
            sys.argv = argv
            agg_cli.main()
        finally:
            sys.argv = _old


if __name__ == "__main__":
    main()
