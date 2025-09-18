from pathlib import Path
import json

from torpedocode.config import DataConfig


def test_cli_ingest_directory(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    nd = raw / "sample.ndjson"
    lines = [
        '{"timestamp":"2025-01-01T14:30:00Z","event_type":"LO+","price":100.0,"size":10}',
        '{"timestamp":"2025-01-01T14:30:01Z","event_type":"MO-","price":99.9,"size":5}',
    ]
    nd.write_text("\n".join(lines))

    # Run as a module to exercise CLI
    import runpy
    import sys

    argv = [
        "prog",
        "--raw-dir",
        str(raw),
        "--cache-root",
        str(tmp_path),
        "--instrument",
        "TEST",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_module("torpedocode.cli.ingest", run_name="__main__")
    finally:
        sys.argv = old_argv

    # Cache should exist
    out = (tmp_path / "TEST.parquet")
    try:
        import pyarrow  # noqa
    except Exception:
        return
    assert out.exists()

