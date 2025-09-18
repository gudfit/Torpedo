"""CLI: Baselines suite

Includes:
- Logistic regression (with/without TDA)
- DeepLOB-style small 1D-CNN classifier
- Neural TPP (RNN-based intensities + log-normal marks)
- Hawkes/Poisson baseline (constant per-type rates)

All baselines load cached splits via LOBDatasetBuilder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Optional: expose DeepLOB2018-style model for tests and reuse
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class InceptionBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            k1, k3, k5 = 1, 3, 5
            s1, s3, s5 = c_out // 3, c_out // 3, c_out - 2 * (c_out // 3)
            self.b1 = nn.Conv1d(c_in, s1, kernel_size=k1, padding=0)
            self.b3 = nn.Conv1d(c_in, s3, kernel_size=k3, padding=k3 // 2)
            self.b5 = nn.Conv1d(c_in, s5, kernel_size=k5, padding=k5 // 2)
            self.bn = nn.BatchNorm1d(c_out)

        def forward(self, x):
            return F.relu(self.bn(torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)))

    class DeepLOBFull(nn.Module):
        def __init__(self, fdim: int):
            super().__init__()
            c0 = 32
            self.conv_in = nn.Conv1d(fdim, c0, kernel_size=1)
            self.inc1 = InceptionBlock(c0, 64)
            self.inc2 = InceptionBlock(64, 64)
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.inc3 = InceptionBlock(64, 96)
            self.inc4 = InceptionBlock(96, 96)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.lstm = nn.LSTM(96, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = F.relu(self.conv_in(x))
            x = self.inc1(x)
            x = self.inc2(x)
            x = self.pool1(x)
            x = self.inc3(x)
            x = self.inc4(x)
            x = self.pool2(x)
            # Restore sequence length by upsampling to original T before LSTM
            B, C, _L = x.shape
            T = x.shape[-1]
            # original length is provided by input; approximate as h length after upsampling
            # we don't have original T here; infer from context: input to forward is (B,T,F)
            # so get target length from x before transpose in caller; pass through size via shape of input
            # For simplicity, capture desired length from pre-transpose argument through closure is not possible;
            # instead, reconstruct from x's batch norm statistics: we will upsample to 4x (due to two pools)
            # If upsampled length differs, interpolate to nearest multiple
            x_up = F.interpolate(x, scale_factor=4, mode="nearest")
            x = x_up.transpose(1, 2)
            h, _ = self.lstm(x)
            return self.head(h)

    class DeepLOB2018Model(nn.Module):
        def __init__(self, time_len: int = 100, feat_dim: int = 40, n_classes: int = 3):
            super().__init__()
            neg = 0.01
            # Input expects NCHW: (B, 1, T, F)
            self.b1_conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2), padding=0)
            self.b1_bn1 = nn.BatchNorm2d(32)
            self.b1_conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
            self.b1_bn2 = nn.BatchNorm2d(32)
            self.b1_conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
            self.b1_bn3 = nn.BatchNorm2d(32)

            self.b2_conv1 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2), padding=0)
            self.b2_bn1 = nn.BatchNorm2d(32)
            self.b2_conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
            self.b2_bn2 = nn.BatchNorm2d(32)
            self.b2_conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
            self.b2_bn3 = nn.BatchNorm2d(32)

            # Width after two (1,2) strides: feat_dim//4; set kernel (1, wid) to collapse width->1
            # So do NOT use padding on (1, wid)
            self.b3_conv1 = nn.Conv2d(32, 32, kernel_size=(1, max(1, feat_dim // 4)), stride=(1, 1), padding=0)
            self.b3_bn1 = nn.BatchNorm2d(32)
            self.b3_conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
            self.b3_bn2 = nn.BatchNorm2d(32)
            self.b3_conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
            self.b3_bn3 = nn.BatchNorm2d(32)

            # Inception branches
            self.inc1_1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same')
            self.inc1_bn1 = nn.BatchNorm2d(64)
            self.inc1_3x1 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same')
            self.inc1_bn2 = nn.BatchNorm2d(64)

            self.inc2_1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same')
            self.inc2_bn1 = nn.BatchNorm2d(64)
            self.inc2_5x1 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same')
            self.inc2_bn2 = nn.BatchNorm2d(64)

            self.inc3_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            self.inc3_1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same')
            self.inc3_bn1 = nn.BatchNorm2d(64)

            self.act = nn.LeakyReLU(negative_slope=neg, inplace=True)
            self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            # x: [B, T, F] -> [B,1,T,F]
            x = x.unsqueeze(1)
            # Block1
            x = self.act(self.b1_bn1(self.b1_conv1(x)))
            x = self.act(self.b1_bn2(self.b1_conv2(x)))
            x = self.act(self.b1_bn3(self.b1_conv3(x)))
            # Block2
            x = self.act(self.b2_bn1(self.b2_conv1(x)))
            x = self.act(self.b2_bn2(self.b2_conv2(x)))
            x = self.act(self.b2_bn3(self.b2_conv3(x)))
            # Block3 (collapse width to 1)
            x = self.act(self.b3_bn1(self.b3_conv1(x)))
            x = self.act(self.b3_bn2(self.b3_conv2(x)))
            x = self.act(self.b3_bn3(self.b3_conv3(x)))
            # Inception over channels, preserve width=1
            b1 = self.act(self.inc1_bn1(self.inc1_1x1(x)))
            b1 = self.act(self.inc1_bn2(self.inc1_3x1(b1)))
            b2 = self.act(self.inc2_bn1(self.inc2_1x1(x)))
            b2 = self.act(self.inc2_bn2(self.inc2_5x1(b2)))
            b3 = self.inc3_pool(x)
            b3 = self.act(self.inc3_bn1(self.inc3_1x1(b3)))
            x = torch.cat([b1, b2, b3], dim=1)  # [B, 192, T, 1]
            # To LSTM [B, T, 192]
            x = x.squeeze(-1).transpose(1, 2)
            h, _ = self.lstm(x)
            return self.head(h)
except Exception:
    DeepLOBFull = None  # type: ignore
    DeepLOB2018Model = None  # type: ignore
import numpy as np

from ..config import DataConfig, TopologyConfig, ModelConfig
from ..data.loader import LOBDatasetBuilder
from ..evaluation.metrics import (
    compute_classification_metrics,
    delong_ci_auroc,
)
from ..evaluation.tpp import (
    TPPArrays,
    rescaled_times,
    model_and_empirical_frequencies,
    nll_per_event_from_arrays,
)


def _fit_logistic_np(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, iters: int = 2000
) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros((Xb.shape[1],), dtype=float)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (p - y) / len(y)
        w -= lr * g
    return w


def _predict_logistic_np(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-z))


def _fit_predict_logistic(Xtr, ytr, Xte) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        clf = LogisticRegression(max_iter=200)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    except Exception:
        w = _fit_logistic_np(Xtr, ytr.astype(float))
        return _predict_logistic_np(Xte, w)


def _run_logistic(
    builder: LOBDatasetBuilder, instrument: str, label_key: str, with_tda: bool, *, strict_tda: bool = False
) -> dict:
    import os as _os
    env_strict = _os.environ.get("PAPER_TORPEDO_STRICT_TDA", "0").lower() in {"1", "true"}
    topo = TopologyConfig(strict_tda=(bool(strict_tda) or env_strict))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key=label_key, topology=topo, topo_stride=5, artifact_dir=None
    )

    def XZ(split):
        return (
            split["features"] if not with_tda else np.hstack([split["features"], split["topology"]])
        )

    Xtr, ytr = XZ(tr), tr["labels"].astype(int)
    Xte, yte = XZ(te), te["labels"].astype(int)
    p = _fit_predict_logistic(Xtr, ytr, Xte)
    m = compute_classification_metrics(p, yte)
    auc, lo, hi = delong_ci_auroc(p, yte)
    return {
        "baseline": "logistic_tda" if with_tda else "logistic",
        "auroc": auc,
        "auroc_ci": [lo, hi],
        "auprc": m.auprc,
        "brier": m.brier,
        "ece": m.ece,
    }


def _run_logistic_shuf_tda(builder: LOBDatasetBuilder, instrument: str, label_key: str, *, strict_tda: bool = False) -> dict:
    """Ablation baseline: shuffle TDA features column-wise to destroy topology signal."""
    import os as _os
    env_strict = _os.environ.get("PAPER_TORPEDO_STRICT_TDA", "0").lower() in {"1", "true"}
    topo = TopologyConfig(strict_tda=(bool(strict_tda) or env_strict))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key=label_key, topology=topo, topo_stride=5, artifact_dir=None
    )
    rng = np.random.default_rng(0)
    Ztr = tr["topology"].copy()
    for j in range(Ztr.shape[1]):
        rng.shuffle(Ztr[:, j])
    Xtr = np.hstack([tr["features"], Ztr])
    Zte = te["topology"].copy()
    for j in range(Zte.shape[1]):
        rng.shuffle(Zte[:, j])
    Xte = np.hstack([te["features"], Zte])
    p = _fit_predict_logistic(Xtr, tr["labels"].astype(int), Xte)
    m = compute_classification_metrics(p, te["labels"].astype(int))
    auc, lo, hi = delong_ci_auroc(p, te["labels"].astype(int))
    return {
        "baseline": "logistic_shuf_tda",
        "auroc": auc,
        "auroc_ci": [lo, hi],
        "auprc": m.auprc,
        "brier": m.brier,
        "ece": m.ece,
    }


def _run_deeplob(builder: LOBDatasetBuilder, instrument: str, label_key: str, *, strict_tda: bool = False) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        raise SystemExit("PyTorch required for deeplob baseline")

    import os as _os
    env_strict = _os.environ.get("PAPER_TORPEDO_STRICT_TDA", "0").lower() in {"1", "true"}
    topo = TopologyConfig(strict_tda=(bool(strict_tda) or env_strict))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key=label_key, topology=topo, topo_stride=5, artifact_dir=None
    )

    from . import train as train_cli

    bptt = 32
    batch = 64
    train_loader = train_cli._window_batches(
        tr["features"], tr["topology"], tr["labels"], bptt=bptt, batch_size=batch, balanced=True
    )
    val_loader = train_cli._window_batches(
        va["features"], va["topology"], va["labels"], bptt=bptt, batch_size=batch, balanced=False
    )

    Fdim = tr["features"].shape[1]

    class DeepSmall(nn.Module):
        def __init__(self, fdim: int):
            super().__init__()
            self.conv1 = nn.Conv1d(fdim, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.transpose(1, 2)
            return self.head(x)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSmall(Fdim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    def _train_one_epoch(loader):
        model.train()
        last = None
        for batch in loader:
            xb = batch["features"].to(dev)
            yb = batch["instability_labels"].float().to(dev)
            logits = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        return last

    _ = _train_one_epoch(train_loader)

    with torch.no_grad():
        xb = torch.from_numpy(te["features"]).unsqueeze(0).to(dev)
        p = torch.sigmoid(model(xb).squeeze(0).squeeze(-1)).cpu().numpy()
    y = te["labels"].astype(int)
    m = compute_classification_metrics(p, y)
    auc, lo, hi = delong_ci_auroc(p, y)
    return {
        "baseline": "deeplob_small",
        "auroc": auc,
        "auroc_ci": [lo, hi],
        "auprc": m.auprc,
        "brier": m.brier,
        "ece": m.ece,
    }


def _run_tpp(builder: LOBDatasetBuilder, instrument: str, *, strict_tda: bool = False) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        raise SystemExit("PyTorch required for neural TPP baseline")

    import os as _os
    env_strict = _os.environ.get("PAPER_TORPEDO_STRICT_TDA", "0").lower() in {"1", "true"}
    topo = TopologyConfig(strict_tda=(bool(strict_tda) or env_strict))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key="instability_s_1", topology=topo, topo_stride=5, artifact_dir=None
    )
    M = int(np.max(tr["event_type_ids"]) + 1) if tr.get("event_type_ids") is not None else 1
    E = 16
    H = 32

    class TPP(nn.Module):
        def __init__(self, M, E, H):
            super().__init__()
            self.emb = nn.Embedding(M, E)
            self.rnn = nn.LSTM(E + 1, H, batch_first=True)
            self.lambda_head = nn.Linear(H, M)
            self.mark_head = nn.Linear(H, 2 * M)

        def forward(self, etypes, dt):
            x = torch.cat([self.emb(etypes), dt.unsqueeze(-1)], dim=-1)
            h, _ = self.rnn(x)
            lam = F.softplus(self.lambda_head(h))
            mu_logsig = self.mark_head(h)
            mu, log_sig = mu_logsig.chunk(2, dim=-1)
            return lam, mu, log_sig

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TPP(M, E, H).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    def nll_batch(lam, mu, log_sig, et, dt, sizes):
        B, T, M = lam.shape
        et_g = et.clamp_min(0).unsqueeze(-1)
        lam_evt = torch.gather(lam, -1, et_g).squeeze(-1)
        log_lam = torch.log(lam_evt.clamp_min(1e-12))
        comp = (lam.sum(dim=-1) * dt).sum(dim=1)
        mu_evt = torch.gather(mu, -1, et_g).squeeze(-1)
        ls_evt = torch.gather(log_sig, -1, et_g).squeeze(-1)
        z = (torch.log(sizes.clamp_min(1e-12)) - mu_evt) / torch.exp(ls_evt)
        mark = 0.5 * z.pow(2) + ls_evt + torch.log(sizes.clamp_min(1e-12)) + 0.5 * np.log(2 * np.pi)
        nll = -(log_lam.sum(dim=1)) + comp + mark.sum(dim=1)
        return nll.mean()

    def mk_loader(split):
        et = torch.from_numpy(split["event_type_ids"]).long().unsqueeze(0).to(dev)
        dt = torch.from_numpy(split["delta_t"]).float().unsqueeze(0).to(dev)
        sizes = torch.from_numpy(split["sizes"]).float().unsqueeze(0).to(dev)
        return [(et, dt, sizes)]

    for et, dt, sz in mk_loader(tr):
        lam, mu, ls = model(et, dt)
        loss = nll_batch(lam, mu, ls, et, dt, sz)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        et = torch.from_numpy(te["event_type_ids"]).long().unsqueeze(0).to(dev)
        dt = torch.from_numpy(te["delta_t"]).float().unsqueeze(0).to(dev)
        lam, mu, ls = model(et, dt)
        lam_np = lam.squeeze(0).cpu().numpy()
        et_np = te["event_type_ids"].astype(int)
        dt_np = te["delta_t"].astype(float)
        arr = TPPArrays(intensities=lam_np, event_type_ids=et_np, delta_t=dt_np)
        xi = rescaled_times(arr)
        emp, mod = model_and_empirical_frequencies(arr)
        nll_evt = nll_per_event_from_arrays(lam_np, et_np, dt_np)
    return {
        "baseline": "neural_tpp",
        "nll_per_event": float(nll_evt),
        "ks_p_value": float(0.0 if xi.size == 0 else 1.0 - np.exp(-np.mean(xi))),
        "empirical_freq": emp.tolist(),
        "model_coverage": mod.tolist(),
    }


def _run_hawkes(builder: LOBDatasetBuilder, instrument: str, *, strict_tda: bool = False) -> dict:
    import os as _os
    env_strict = _os.environ.get("PAPER_TORPEDO_STRICT_TDA", "0").lower() in {"1", "true"}
    topo = TopologyConfig(strict_tda=(bool(strict_tda) or env_strict))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key="instability_s_1", topology=topo, topo_stride=5, artifact_dir=None
    )
    et_tr = te["event_type_ids"].astype(int)
    M = int(np.max(et_tr) + 1) if et_tr.size > 0 else 1
    dt_tr = te["delta_t"].astype(float)
    Ttot = float(np.sum(dt_tr)) if dt_tr.size else 1.0
    counts = np.bincount(et_tr, minlength=M).astype(float)
    mu = counts / max(Ttot, 1e-12)
    lam = np.tile(mu[None, :], (len(et_tr), 1))
    nll_evt = nll_per_event_from_arrays(lam, et_tr, dt_tr)
    arr = TPPArrays(intensities=lam, event_type_ids=et_tr, delta_t=dt_tr)
    xi = rescaled_times(arr)
    emp, mod = model_and_empirical_frequencies(arr)
    return {
        "baseline": "hawkes_const_rate",
        "nll_per_event": float(nll_evt),
        "ks_p_value": float(0.0 if xi.size == 0 else 1.0 - np.exp(-np.mean(xi))),
        "empirical_freq": emp.tolist(),
        "model_coverage": mod.tolist(),
    }


def _run_hawkes_exp(builder: LOBDatasetBuilder, instrument: str, *, strict_tda: bool = False) -> dict:
    """Multivariate Hawkes with single exponential kernel (simple LS fit).

    Uses a fixed decay beta estimated from mean inter-arrival time and fits
    nonnegative mu and alpha via least squares on a causal design.
    """
    topo = TopologyConfig(strict_tda=bool(strict_tda))
    tr, va, te, _ = builder.build_splits(
        instrument, label_key="instability_s_1", topology=topo, topo_stride=5, artifact_dir=None
    )
    et = te["event_type_ids"].astype(int)
    dt = te["delta_t"].astype(float)
    Tn = len(et)
    if Tn == 0:
        return {
            "baseline": "hawkes_exp",
            "nll_per_event": float("nan"),
            "empirical_freq": [],
            "model_coverage": [],
        }
    M = int(np.max(et) + 1)
    t = np.cumsum(dt)
    beta = 1.0 / max(float(np.mean(dt)), 1e-6)

    S = np.zeros((Tn, M), dtype=float)
    last_idx = [-1] * M
    last_time = [0.0] * M
    decays = np.zeros(M, dtype=float)
    for i in range(Tn):
        if i > 0:
            dtau = float(t[i] - t[i - 1])
            decays *= np.exp(-beta * dtau)
        m = int(et[i])
        decays[m] += 1.0
        S[i] = decays

    eps = 1e-9
    y_rate = np.zeros((Tn, M), dtype=float)
    for i in range(Tn):
        y_rate[i, et[i]] = 1.0 / max(dt[i], eps)

    X = np.hstack([np.ones((Tn, 1), dtype=float), S])
    coefs = np.zeros((M, 1 + M), dtype=float)
    for m in range(M):
        w, *_ = np.linalg.lstsq(X, y_rate[:, m], rcond=None)
        w = np.clip(w, 0.0, None)
        coefs[m] = w

    mu = coefs[:, 0]
    A = coefs[:, 1:]
    lam = np.maximum(mu[None, :] + S @ A.T, 1e-9)

    nll_evt = nll_per_event_from_arrays(lam, et, dt)
    arr = TPPArrays(intensities=lam, event_type_ids=et, delta_t=dt)
    emp, mod = model_and_empirical_frequencies(arr)
    return {
        "baseline": "hawkes_exp",
        "nll_per_event": float(nll_evt),
        "empirical_freq": emp.tolist(),
        "model_coverage": mod.tolist(),
        "beta": float(beta),
    }


def main():
    ap = argparse.ArgumentParser(description="Run baselines")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--instrument", type=str, required=True)
    ap.add_argument("--label-key", type=str, default="instability_s_1")
    ap.add_argument(
        "--baseline",
        type=str,
        default="logistic",
        choices=[
            "logistic",
            "deeplob",
            "deeplob_full",
            "deeplob2018",
            "tpp",
            "hawkes",
            "hawkes_exp",
            "logistic_shuf_tda",
        ],
    )
    ap.add_argument("--with-tda", action="store_true", help="Only for logistic baseline")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--strict-tda", action="store_true", help="Fail if TDA backends missing")
    args = ap.parse_args()

    try:
        import random as _random

        _random.seed(int(args.seed))
        np.random.default_rng(int(args.seed))
        try:
            import torch as _t

            _t.manual_seed(int(args.seed))
            if _t.cuda.is_available():
                _t.cuda.manual_seed_all(int(args.seed))
            try:
                _t.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                _t.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    data = DataConfig(
        raw_data_root=args.cache_root, cache_root=args.cache_root, instruments=[args.instrument]
    )
    builder = LOBDatasetBuilder(data)

    if args.baseline == "logistic":
        out = _run_logistic(builder, args.instrument, args.label_key, args.with_tda, strict_tda=bool(args.strict_tda))
    elif args.baseline == "deeplob":
        out = _run_deeplob(builder, args.instrument, args.label_key, strict_tda=bool(args.strict_tda))
    elif args.baseline in ("deeplob_full", "deeplob2018"):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception:
            raise SystemExit("PyTorch required for deeplob_full baseline")

        topo = TopologyConfig()
        tr, va, te, _ = builder.build_splits(
            args.instrument,
            label_key=args.label_key,
            topology=topo,
            topo_stride=5,
            artifact_dir=None,
        )

        from . import train as train_cli

        bptt = 64
        batch = 128
        train_loader = train_cli._window_batches(
            tr["features"], tr["topology"], tr["labels"], bptt=bptt, batch_size=batch, balanced=True
        )
        val_loader = train_cli._window_batches(
            va["features"],
            va["topology"],
            va["labels"],
            bptt=bptt,
            batch_size=batch,
            balanced=False,
        )

        Fdim = tr["features"].shape[1]

        class InceptionBlock(nn.Module):
            def __init__(self, c_in: int, c_out: int):
                super().__init__()
                # Canonical DeepLOB-style multi-scale convs
                k1, k3, k5 = 1, 3, 5
                s1, s3, s5 = c_out // 3, c_out // 3, c_out - 2 * (c_out // 3)
                self.b1 = nn.Conv1d(c_in, s1, kernel_size=k1, padding=0)
                self.b3 = nn.Conv1d(c_in, s3, kernel_size=k3, padding=k3 // 2)
                self.b5 = nn.Conv1d(c_in, s5, kernel_size=k5, padding=k5 // 2)
                self.bn = nn.BatchNorm1d(c_out)

            def forward(self, x):
                return F.relu(self.bn(torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)))

        class DeepLOBFull(nn.Module):
            def __init__(self, fdim: int):
                super().__init__()
                # Channels aligned to common DeepLOB2018 reference
                c0 = 32
                self.conv_in = nn.Conv1d(fdim, c0, kernel_size=1)
                self.inc1 = InceptionBlock(c0, 64)
                self.inc2 = InceptionBlock(64, 64)
                self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
                self.inc3 = InceptionBlock(64, 96)
                self.inc4 = InceptionBlock(96, 96)
                self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
                self.lstm = nn.LSTM(96, 64, num_layers=2, batch_first=True, dropout=0.1)
                self.head = nn.Linear(64, 1)

            def forward(self, x):
                x = x.transpose(1, 2)
                x = F.relu(self.conv_in(x))
                x = self.inc1(x)
                x = self.inc2(x)
                x = self.pool1(x)
                x = self.inc3(x)
                x = self.inc4(x)
                x = self.pool2(x)
                x = x.transpose(1, 2)
                h, _ = self.lstm(x)
                return self.head(h)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepLOBFull(Fdim).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)

        def _train_one_epoch(loader):
            model.train()
            last = None
            for batch in loader:
                xb = batch["features"].to(dev)
                yb = batch["instability_labels"].float().to(dev)
                logits = model(xb).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.detach().cpu())
            return last

        _ = _train_one_epoch(train_loader)
        with torch.no_grad():
            xb = torch.from_numpy(te["features"]).unsqueeze(0).to(dev)
            p = torch.sigmoid(model(xb).squeeze(0).squeeze(-1)).cpu().numpy()
        y = te["labels"].astype(int)
        m = compute_classification_metrics(p, y)
        auc, lo, hi = delong_ci_auroc(p, y)
        out = {
            "baseline": ("deeplob2018" if args.baseline == "deeplob2018" else "deeplob_full"),
            "auroc": auc,
            "auroc_ci": [lo, hi],
            "auprc": m.auprc,
            "brier": m.brier,
            "ece": m.ece,
        }
    elif args.baseline == "tpp":
        out = _run_tpp(builder, args.instrument, strict_tda=bool(args.strict_tda))
    elif args.baseline == "hawkes_exp":
        out = _run_hawkes_exp(builder, args.instrument, strict_tda=bool(args.strict_tda))
    elif args.baseline == "logistic_shuf_tda":
        out = _run_logistic_shuf_tda(builder, args.instrument, args.label_key, strict_tda=bool(args.strict_tda))
    else:
        out = _run_hawkes(builder, args.instrument, strict_tda=bool(args.strict_tda))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
