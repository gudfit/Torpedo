"""Sampling utilities for balanced mini-batching and walk-forward splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class BalancedBatchSampler:
    """Yield balanced indices for binary labels."""

    labels: Sequence[int]
    batch_size: int

    def __iter__(self) -> Iterable[np.ndarray]:
        y = np.asarray(self.labels).astype(int)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rng = np.random.default_rng(0)
        n_half = self.batch_size // 2
        n_batches = int(np.ceil(len(y) / self.batch_size))
        for _ in range(n_batches):
            pi = rng.choice(pos, size=min(n_half, len(pos)), replace=len(pos) < n_half)
            ni = rng.choice(neg, size=min(n_half, len(neg)), replace=len(neg) < n_half)
            idx = np.concatenate([pi, ni])
            rng.shuffle(idx)
            yield idx


@dataclass
class WalkForwardSplitter:
    """Create walk-forward train/val/test splits over time-ordered indices."""

    n_samples: int
    train_frac: float = 0.6
    val_frac: float = 0.2

    def splits(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        n = int(self.n_samples)
        if n <= 0:
            return []
        t = int(np.floor(self.train_frac * n))
        v = int(np.floor((self.train_frac + self.val_frac) * n))
        train_idx = np.arange(0, t)
        val_idx = np.arange(t, v)
        test_idx = np.arange(v, n)
        return [(train_idx, val_idx, test_idx)]


__all__ = ["BalancedBatchSampler", "WalkForwardSplitter"]
