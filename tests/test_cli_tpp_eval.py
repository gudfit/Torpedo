import json
import numpy as np
from pathlib import Path


def test_cli_tpp_eval_npz(tmp_path, monkeypatch, capsys):
    from torpedocode.cli import tpp_eval as tppc

    T, M = 50, 3
    rng = np.random.default_rng(0)
    intensities = rng.uniform(0.1, 1.0, size=(T, M)).astype(np.float32)
    event_type_ids = rng.integers(0, M, size=(T,)).astype(np.int64)
    delta_t = rng.exponential(0.2, size=(T,)).astype(np.float32)
    npz = tmp_path / "arrs.npz"
    np.savez(npz, intensities=intensities, event_type_ids=event_type_ids, delta_t=delta_t)

    argv = ["prog", "--npz", str(npz)]
    import sys

    monkeypatch.setattr(sys, "argv", argv)
    tppc.main()
    out = capsys.readouterr().out
    res = json.loads(out)
    assert "ks_p_value" in res and "coverage_error" in res and "nll_per_event" in res
    assert isinstance(res["nll_per_event"], (int, float))
    # Per-type diagnostics and coverage present
    assert "per_type" in res and isinstance(res["per_type"], list)
    assert "empirical_freq" in res and "model_coverage" in res
    assert len(res["empirical_freq"]) == M and len(res["model_coverage"]) == M
