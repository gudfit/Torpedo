import json
from pathlib import Path
import numpy as np
import pandas as pd


def test_cli_baselines_logistic(tmp_path, monkeypatch, capsys):
    # Build a tiny cache and run baseline
    from torpedocode.cli import baselines as bl
    inst = "TEST"
    cache_root = tmp_path
    # minimal event frame parquet
    ts = pd.date_range("2025-01-01", periods=100, freq="s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "event_type": ["MO+", "MO-"] * 50,
        "size": np.abs(np.random.randn(100)).astype(float),
        "price": 100 + np.cumsum(np.random.randn(100)).astype(float) * 0.01,
        "bid_price_1": 100 + np.random.randn(100).astype(float) * 0.01,
        "ask_price_1": 100.1 + np.random.randn(100).astype(float) * 0.01,
        "bid_size_1": np.random.randint(1, 50, size=100),
        "ask_size_1": np.random.randint(1, 50, size=100),
    })
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return  # skip if parquet writer missing
    df.to_parquet(cache_root / f"{inst}.parquet", index=False)

    argv = ["prog", "--cache-root", str(cache_root), "--instrument", inst, "--label-key", "instability_s_1"]
    import sys
    monkeypatch.setattr(sys, "argv", argv)
    bl.main()
    out = capsys.readouterr().out
    res = json.loads(out)
    assert "auroc" in res and "auprc" in res

