from __future__ import annotations

import numpy as np

from torpedocode.data.synthetic_ctmc import CTMCConfig, generate_ctmc_sequence


def test_ctmc_topology_image_shape_backend_agnostic():
    # Small sequence with PI settings; even if PH backends are missing, we expect a vector of the right size
    T = 64
    res = 16
    cfg = CTMCConfig(
        T=T,
        levels=4,
        emit_topology=True,
        topo_window=8,
        topo_stride=1,
        topo_representation="image",
        image_resolution=res,
        image_bandwidth=0.05,
    )
    rec = generate_ctmc_sequence(cfg)
    assert "topology" in rec
    Z = rec["topology"]
    # For max_homology_dimension=1 in generator, channels = 2 * res^2
    assert Z.shape == (T, 2 * res * res)
    assert np.isfinite(Z).all()

