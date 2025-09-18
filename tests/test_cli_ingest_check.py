from pathlib import Path
import runpy
import sys


def test_cli_ingest_check_runs(tmp_path: Path):
    argv = [
        "prog",
        "--raw-dir",
        str(tmp_path),
        "--cache-root",
        str(tmp_path),
        "--instrument",
        "DUMMY",
        "--check",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_module("torpedocode.cli.ingest", run_name="__main__")
    finally:
        sys.argv = old

