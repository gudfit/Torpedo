import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest


def test_cli_ingest_eta_labels(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    nd = raw / "sample.ndjson"
    # Two timestamps with a price jump to trigger labels when eta=0
    lines = [
        '{"timestamp":"2025-01-01T14:30:00Z","event_type":"LO+","price":100.0,"size":10}',
        '{"timestamp":"2025-01-01T14:30:01Z","event_type":"MO-","price":101.0,"size":5}',
    ]
    nd.write_text("\n".join(lines))

    # Run ingest with --eta 0.0
    import runpy

    argv = [
        "prog",
        "--raw-dir",
        str(raw),
        "--cache-root",
        str(tmp_path),
        "--instrument",
        "TESTE",
        "--eta",
        "0.0",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_module("torpedocode.cli.ingest", run_name="__main__")
    finally:
        sys.argv = old_argv

    out = tmp_path / "TESTE.parquet"
    try:
        import pyarrow  # noqa: F401
        import pandas as pd
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")
    assert out.exists()
    df = pd.read_parquet(out)
    # At least one label column must exist
    label_cols = [
        c for c in df.columns if c.startswith("instability_s_") or c.startswith("instability_e_")
    ]
    assert label_cols, "expected label columns in cached parquet with --eta"
