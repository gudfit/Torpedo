"""Temperature scaling for binary classifiers.

Pure NumPy implementation with optional SciPy acceleration. Fits a single
temperature parameter T>0 by minimizing NLL on a validation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    p = _sigmoid(logits)
    y = labels.astype(float)
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


@dataclass
class TemperatureScaler:
    """Fit and apply a single scalar temperature for calibration."""

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit T by minimizing NLL of sigmoid(logits / T)."""

        z = np.asarray(logits, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        T = max(self.temperature, 1e-3)

        try:
            from scipy.optimize import minimize_scalar  # type: ignore

            def obj(t: float) -> float:
                return _binary_nll_from_logits(z / max(t, 1e-6), y)

            res = minimize_scalar(obj, bounds=(1e-3, 100.0), method="bounded")
            T = float(res.x)
        except Exception:
            logT = np.log(T)
            lr = 0.1
            for _ in range(200):
                T = np.exp(logT)
                s = _sigmoid(z / T)
                grad = np.mean((s - y) * z) / (T * T)
                logT -= lr * grad
            T = float(np.exp(logT))

        self.temperature = max(T, 1e-6)
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) / float(self.temperature)


__all__ = ["TemperatureScaler"]
