from torpedocode.config import TopologyConfig


def test_topology_defaults_match_methodology():
    cfg = TopologyConfig()
    # Largest connected component epsilon rule with threshold 0.99
    assert cfg.vr_epsilon_rule == "largest_cc"
    assert abs(cfg.vr_lcc_threshold - 0.99) < 1e-9
    # Landscape levels K ∈ {3,5}; default 5
    assert cfg.landscape_levels in (3, 5)
    # Image resolution up to 128; default should not exceed 128
    assert cfg.image_resolution <= 128

