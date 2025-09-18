import json
from pathlib import Path


def test_cli_ca_preview_runs(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import ca_preview as cap

    # Minimal NDJSON
    nd = tmp_path / "X.ndjson"
    nd.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2024-01-02T10:00:00Z",
                        "event_type": "LO+",
                        "price": 100.0,
                        "size": 10,
                        "symbol": "X",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2024-01-02T10:00:01Z",
                        "event_type": "LO-",
                        "price": 100.5,
                        "size": 8,
                        "symbol": "X",
                    }
                ),
            ]
        )
    )
    # Corporate actions CSV
    ca = tmp_path / "actions.csv"
    ca.write_text("symbol,date,adj_factor\nX,2024-01-02,2.0\n")
    out_json = tmp_path / "summary.json"
    out_csv = tmp_path / "sample.csv"
    argv = [
        "prog",
        "--input",
        str(nd),
        "--actions-csv",
        str(ca),
        "--symbol",
        "X",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-03",
        "--round-ticks",
        "--tick-size",
        "0.01",
        "--sample-csv",
        str(out_csv),
        "--output",
        str(out_json),
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr("sys.argv", argv)
    cap.main()
    assert out_json.exists()
    assert out_csv.exists()
    obj = json.loads(out_json.read_text())
    assert obj["rows"] >= 1
    # Ensure price columns were included
    assert any("price" in c for c in obj.get("price_cols", []))
