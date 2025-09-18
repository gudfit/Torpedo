import sys
from pathlib import Path
import json
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_manifest_passthrough_extras(tmp_path, monkeypatch):
    from torpedocode.cli import manifest as mcli

    # Prepare minimal NDJSON
    nd = tmp_path / "A.ndjson"
    base_ts = 1735741800
    lines = []
    for i in range(120):
        et = "LO+" if i % 3 == 0 else ("MO+" if i % 3 == 1 else "CX+")
        price = 100.0 + (0.01 * i)
        lines.append(json.dumps({"timestamp": base_ts + i, "event_type": et, "price": price, "size": 10}))
    nd.write_text("\n".join(lines))

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
            "seed": 13,
            "beta": 1e-4,
            "temperature_scale": True,
            "tpp_diagnostics": True,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(man))
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")

    argv = ["prog", "--manifest", str(manifest_path)]
    monkeypatch.setattr(sys, "argv", argv)
    mcli.main()
    # Resolve instrument name used by cache CLI (file stem)
    inst = nd.stem
    art = art_root / inst / "instability_s_1"
    assert (art / "predictions_test.csv").exists()
    assert (art / "training_meta.json").exists()
    assert (art / "temperature.json").exists()
    assert (art / "tpp_test_arrays.npz").exists()
    assert (art / "tpp_test_diagnostics.json").exists()

