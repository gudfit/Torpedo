import numpy as np
import pytest


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="requires torch")
def test_ctmc_pretrain_loss_decreases():
    import torch
    from torpedocode.config import ModelConfig
    from torpedocode.models.hybrid import HybridLOBModel
    from torpedocode.training.losses import HybridLossComputer
    from torpedocode.data.synthetic_ctmc import CTMCConfig, generate_ctmc_sequence

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = CTMCConfig(T=64, num_event_types=4, feature_dim=6)
    rec = generate_ctmc_sequence(cfg, rng)

    # One batch, add a small synthetic topology vector
    X = rec["features"][None, ...].astype(np.float32)
    Z = np.zeros((1, X.shape[1], 4), dtype=np.float32)
    et = rec["event_type_ids"][None, ...]
    dt = rec["delta_t"][None, ...]
    sz = rec["sizes"][None, ...]

    model = HybridLOBModel(feature_dim=X.shape[2], topo_dim=Z.shape[2], num_event_types=4, config=ModelConfig(hidden_size=32, num_layers=1, include_market_embedding=False))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = HybridLossComputer(lambda_cls=0.0, beta=1e-4, gamma=0.0)

    def step():
        xb = torch.from_numpy(X)
        zb = torch.from_numpy(Z)
        out = model(xb, zb)
        batch = {
            "features": xb,
            "event_type_ids": torch.from_numpy(et),
            "delta_t": torch.from_numpy(dt),
            "sizes": torch.from_numpy(sz),
        }
        lo = loss_fn(out, batch, list(model.parameters()))
        return lo.total

    model.train()
    l0 = step().item()
    opt.zero_grad(); step().backward(); opt.step()
    l1 = step().item()
    assert l1 <= l0 + 1e-5

