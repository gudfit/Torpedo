import sys
from pathlib import Path


def test_cli_cache_single_file(tmp_path, monkeypatch):
    from torpedocode.cli import cache as cache_cli

    # NDJSON within regular session timezone
    nd = tmp_path / "FOO.ndjson"
    nd.write_text(
        "\n".join(
            [
                '{"timestamp":"2025-01-01T14:30:00Z","event_type":"LO+","price":100.0,"size":10}',
                '{"timestamp":"2025-01-01T14:30:01Z","event_type":"MO-","price":99.9,"size":5}',
                '{"timestamp":"2025-01-01T14:30:02Z","event_type":"CX+","price":100.1,"size":2}',
            ]
        )
    )

    cache_root = tmp_path / "caches"
    argv = [
        "prog",
        "--input",
        str(nd),
        "--cache-root",
        str(cache_root),
        "--drop-auctions",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return  # skip if parquet writer not available
    cache_cli.main()
    assert (cache_root / "FOO.parquet").exists()

