import numpy as np

from torpedocode.features.topological import TopologicalFeatureGenerator
from torpedocode.config import TopologyConfig


def test_vr_epsilon_largest_cc_reasonable():
    # 10 points on a line with unit spacing; smallest eps to connect ~all is ~1.0
    X = np.stack([np.arange(10, dtype=float), np.zeros(10)], axis=1)
    cfg = TopologyConfig(
        complex_type="vietoris_rips",
        max_homology_dimension=1,
        persistence_representation="landscape",
        landscape_levels=3,
        vr_auto_epsilon=True,
        vr_epsilon_rule="largest_cc",
        vr_lcc_threshold=0.9,
    )
    gen = TopologicalFeatureGenerator(cfg)
    eps = gen._epsilon_for_lcc(X, threshold=cfg.vr_lcc_threshold)
    assert np.isfinite(eps) and eps > 0
    # Expect around 1.0 to connect neighbors
    assert 0.5 <= eps <= 2.0
