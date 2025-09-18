import pytest

torch = pytest.importorskip("torch")

from torpedocode.models.hybrid import HybridLOBModel
from torpedocode.config import ModelConfig


def test_model_forward_with_native_fuse():
    torch.manual_seed(0)
    B, T, F, Z, M = 2, 5, 4, 3, 6
    cfg = ModelConfig(hidden_size=32, num_layers=1, dropout=0.0, include_market_embedding=False, use_native_fuse=True)
    model = HybridLOBModel(F, Z, num_event_types=M, config=cfg)
    X = torch.randn(B, T, F)
    Zt = torch.randn(B, T, Z)
    out = model(X, Zt)
    assert out.instability_logits.shape == (B, T, 1)
    # Intensities present for all event types
    assert len(out.intensities) == M
    for k, v in out.intensities.items():
        assert v.shape == (B, T, 1)
