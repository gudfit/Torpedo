import sys
from pathlib import Path
import json

import pytest


def test_cli_ingest_multiple_raw_dirs(tmp_path, monkeypatch):
    from torpedocode.cli import ingest as ingest_cli

    # Create two small NDJSON files in different dirs
    d1 = tmp_path / "raw1"; d1.mkdir()
    d2 = tmp_path / "raw2"; d2.mkdir()
    (d1 / "a.ndjson").write_text('\n'.join([
        json.dumps({"timestamp": "2025-01-01T00:00:00Z", "event_type": "LO+", "price": 100.0, "size": 1.0}),
    ]))
    (d2 / "b.ndjson").write_text('\n'.join([
        json.dumps({"timestamp": "2025-01-01T00:00:01Z", "event_type": "MO-", "price": 99.9, "size": 2.0}),
    ]))

    cache_root = tmp_path / "cache"
    argv = [
        "prog",
        "--raw-dir", str(d1), str(d2),
        "--cache-root", str(cache_root),
        "--instrument", "MTEST",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pytest.skip("pyarrow not available")
    ingest_cli.main()
    out = cache_root / "MTEST.parquet"
    assert out.exists()

