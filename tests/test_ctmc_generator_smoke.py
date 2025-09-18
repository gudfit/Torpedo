from __future__ import annotations

import numpy as np

from torpedocode.data.synthetic_ctmc import CTMCConfig, generate_ctmc_sequence


def test_ctmc_generator_basic_shapes():
    cfg = CTMCConfig(T=200, levels=6, emit_topology=False)
    rec = generate_ctmc_sequence(cfg)
    T = cfg.T
    assert rec["event_type_ids"].shape == (T,)
    assert rec["delta_t"].shape == (T,)
    assert rec["sizes"].shape == (T,)
    assert rec["features"].shape == (T, 2 * cfg.levels)
    # Positivity and finiteness checks
    assert np.all(np.isfinite(rec["delta_t"])) and np.all(rec["delta_t"] > 0)
    assert np.all(np.isfinite(rec["sizes"])) and np.all(rec["sizes"] > 0)


def test_ctmc_expand_types_by_level_ids():
    L = 5
    cfg = CTMCConfig(T=400, levels=L, emit_topology=False, expand_types_by_level=True)
    rec = generate_ctmc_sequence(cfg)
    et = rec["event_type_ids"]
    # Expect total types = 2 + 4*L and all ids within range
    M = 2 + 4 * L
    assert et.min() >= 0 and et.max() < M
    # Expect some variety
    assert len(np.unique(et)) >= min(M, 6)
