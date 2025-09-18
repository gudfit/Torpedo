import numpy as np

from torpedocode.features.topological import TopologicalFeatureGenerator
from torpedocode.config import TopologyConfig


def test_topological_generator_fallback_shapes_landscape():
    cfg = TopologyConfig(
        complex_type="cubical",
        max_homology_dimension=1,
        persistence_representation="landscape",
        landscape_levels=3,
    )
    gen = TopologicalFeatureGenerator(cfg)
    slab = np.zeros((2, 8))
    out = gen.transform(np.stack([slab, slab], axis=0))
    assert out.shape == (2, 3 * 2)


def test_topological_generator_fallback_shapes_image():
    cfg = TopologyConfig(
        complex_type="vietoris_rips",
        max_homology_dimension=1,
        persistence_representation="image",
        image_resolution=8,
    )
    gen = TopologicalFeatureGenerator(cfg)
    slab = np.zeros((4, 4))
    out = gen.transform(np.stack([slab], axis=0))
    # With separate channels per homology dimension (H0/H1)
    assert out.shape == (1, 2 * 8 * 8)
