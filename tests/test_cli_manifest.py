import sys
from pathlib import Path
import json
import pytest


def test_cli_manifest_json_roundtrip(tmp_path, monkeypatch):
    from torpedocode.cli import manifest as mcli

    # Prepare NDJSON sample with enough data to generate mixed labels
    nd = tmp_path / "A.ndjson"
    events = []
    base_ts = 1735741800  # 2025-01-01T14:30:00Z
    # Stable period
    for i in range(15):
        events.append(
            f'{{"timestamp":{base_ts + i},"event_type":"LO+","price":{100.0 + i*0.01},"size":10}}'
        )
    # Volatile event to trigger instability label
    events.append(f'{{"timestamp":{base_ts + 15},"event_type":"MO-","price":105.0,"size":100}}')
    # Another stable period
    for i in range(16, 30):
        events.append(
            f'{{"timestamp":{base_ts + i},"event_type":"LO-","price":{105.0 + i*0.01},"size":8}}'
        )
    nd.write_text("\n".join(events))

    cache_root = tmp_path / "caches"
    art_root = tmp_path / "artifacts"
    man = {
        "data": {"input": str(nd), "cache_root": str(cache_root), "drop_auctions": True},
        "train": {
            "artifact_root": str(art_root),
            "label_key": "instability_s_1",
            "epochs": 1,
            "batch": 8,
            "bptt": 8,
            "topo_stride": 4,
            "device": "cpu",
        },
        "aggregate": {
            "mode": "pred",
            "output": str(tmp_path / "agg.json"),
            "block_bootstrap": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(man))
    try:
        import pyarrow  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        pytest.skip("requires pyarrow + torch for full manifest run")

    argv = ["prog", "--manifest", str(manifest_path)]
    monkeypatch.setattr(sys, "argv", argv)
    mcli.main()
    assert (tmp_path / "agg.json").exists()
