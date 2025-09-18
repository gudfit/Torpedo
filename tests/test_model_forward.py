import pytest

torch = pytest.importorskip("torch")

from torpedocode.models.hybrid import HybridLOBModel
from torpedocode.config import ModelConfig


def test_hybrid_model_forward_shapes():
    feature_dim = 6
    topo_dim = 4
    num_event_types = 3
    cfg = ModelConfig(hidden_size=16, num_layers=1, include_market_embedding=False)
    model = HybridLOBModel(feature_dim, topo_dim, num_event_types, cfg)
    x = torch.zeros((2, 5, feature_dim))
    z = torch.zeros((2, 5, topo_dim))
    out = model(x, z)
    assert out.instability_logits.shape == (2, 5, 1)
    assert len(out.intensities) == num_event_types
    for i in range(num_event_types):
        t = out.intensities[f"event_{i}"]
        assert t.shape == (2, 5, 1)
