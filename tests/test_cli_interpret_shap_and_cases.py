import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_minimal_cache(tmp_path: Path, name: str, n: int = 160):
    ts = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "event_type": ["MO+", "MO-", "LO+", "LO-"] * (n // 4),
            "size": np.abs(np.random.randn(n)).astype(float),
            "price": 100 + np.cumsum(np.random.randn(n)).astype(float) * 0.01,
            "bid_price_1": 100 + np.random.randn(n).astype(float) * 0.01,
            "ask_price_1": 100.1 + np.random.randn(n).astype(float) * 0.01,
            "bid_size_1": np.random.randint(1, 100, size=n),
            "ask_size_1": np.random.randint(1, 100, size=n),
        }
    )
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    except Exception:
        pytest.skip("pyarrow not available for parquet cache")


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_cli_interpret_with_shap(tmp_path, monkeypatch, capsys):
    try:
        import shap  # noqa: F401
    except Exception:
        pytest.skip("shap not available")
    from torpedocode.cli import interpret as interp

    inst = "TESTINT"
    _write_minimal_cache(tmp_path, inst, n=160)
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--instrument",
        inst,
        "--label-key",
        "instability_s_1",
        "--device",
        "cpu",
        "--shap",
        "--shap-samples",
        "8",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    interp.main()
    out = capsys.readouterr().out
    assert "baseline" in out and ("shap_summary" in out or "shap_summary_error" in out)


def test_cli_case_studies(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import case_studies as cs

    inst = "TESTCS"
    _write_minimal_cache(tmp_path, inst, n=160)
    out_json = tmp_path / "cases.json"
    argv = [
        "prog",
        "--cache-root",
        str(tmp_path),
        "--instrument",
        inst,
        "--horizon-s",
        "1",
        "--quantile",
        "0.9",
        "--topo-window-s",
        "1",
        "--output",
        str(out_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cs.main()
    assert out_json.exists()
    obj = json_load(out_json)
    assert "cases" in obj and obj["num_cases"] == len(obj["cases"]) and obj["num_cases"] >= 1


def json_load(path: Path):
    import json

    return json.loads(path.read_text())
