"""Baseline model definitions (DeepLOB variants).

Provides reusable DeepLOB-style architectures to avoid duplication in CLIs.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _require_torch():  # pragma: no cover - guard
    if torch is None or nn is None or F is None:
        raise ImportError("PyTorch is required for DeepLOB baselines")


class InceptionBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        _require_torch()
        super().__init__()
        k1, k3, k5 = 1, 3, 5
        s1, s3, s5 = c_out // 3, c_out // 3, c_out - 2 * (c_out // 3)
        self.b1 = nn.Conv1d(c_in, s1, kernel_size=k1, padding=0)
        self.b3 = nn.Conv1d(c_in, s3, kernel_size=k3, padding=k3 // 2)
        self.b5 = nn.Conv1d(c_in, s5, kernel_size=k5, padding=k5 // 2)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):  # type: ignore[override]
        _require_torch()
        return F.relu(self.bn(torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)))


class DeepLOBFull(nn.Module):
    """A compact DeepLOB-like model (1D conv + inception + LSTM)."""

    def __init__(self, fdim: int):
        _require_torch()
        super().__init__()
        c0 = 32
        self.conv_in = nn.Conv1d(fdim, c0, kernel_size=1)
        self.inc1 = InceptionBlock(c0, 64)
        self.inc2 = InceptionBlock(64, 64)
        self.pool1 = nn.Identity()
        self.inc3 = InceptionBlock(64, 96)
        self.inc4 = InceptionBlock(96, 96)
        self.pool2 = nn.Identity()
        self.lstm = nn.LSTM(96, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(64, 1)

    def forward(self, x):  # type: ignore[override]
        _require_torch()
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


class DeepLOB2018Model(nn.Module):
    """A closer 2D-conv variant of DeepLOB (3-class head can be adapted)."""

    def __init__(self, time_len: int = 100, feat_dim: int = 40, n_classes: int = 3):
        _require_torch()
        super().__init__()
        neg = 0.01
        self.b1_conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.b1_bn1 = nn.BatchNorm2d(32)
        self.b1_conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        self.b1_bn2 = nn.BatchNorm2d(32)
        self.b1_conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        self.b1_bn3 = nn.BatchNorm2d(32)

        self.b2_conv1 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.b2_bn1 = nn.BatchNorm2d(32)
        self.b2_conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        self.b2_bn2 = nn.BatchNorm2d(32)
        self.b2_conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        self.b2_bn3 = nn.BatchNorm2d(32)

        self.b3_conv1 = nn.Conv2d(
            32, 32, kernel_size=(1, max(1, feat_dim // 4)), stride=(1, 1), padding=0
        )
        self.b3_bn1 = nn.BatchNorm2d(32)
        self.b3_conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        self.b3_bn2 = nn.BatchNorm2d(32)
        self.b3_conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        self.b3_bn3 = nn.BatchNorm2d(32)

        self.inc1_1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same")
        self.inc1_bn1 = nn.BatchNorm2d(64)
        self.inc1_3x1 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same")
        self.inc1_bn2 = nn.BatchNorm2d(64)

        self.inc2_1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same")
        self.inc2_bn1 = nn.BatchNorm2d(64)
        self.inc2_5x1 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same")
        self.inc2_bn2 = nn.BatchNorm2d(64)

        self.inc3_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.inc3_1x1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same")
        self.inc3_bn1 = nn.BatchNorm2d(64)

        self.act = nn.LeakyReLU(negative_slope=neg, inplace=True)
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):  # type: ignore[override]
        _require_torch()
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
        x = torch.cat([b1, b2, b3], dim=1)
        x = x.squeeze(-1).transpose(1, 2)
        h, _ = self.lstm(x)
        return self.head(h)


__all__ = [
    "InceptionBlock",
    "DeepLOBFull",
    "DeepLOB2018Model",
]
