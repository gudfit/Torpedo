"""Split-safe standardization utilities and schema export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import json
import numpy as np


@dataclass
class SplitSafeStandardScaler:
    """Fit per-feature mean/std on training data, apply to all splits without leakage."""

    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    feature_names_: Optional[List[str]] = None

    def fit(
        self, X: np.ndarray, *, feature_names: Optional[List[str]] = None
    ) -> "SplitSafeStandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.feature_names_ = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.scale_
        return Z.astype(np.float32)

    def fit_transform(
        self, X: np.ndarray, *, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        return self.fit(X, feature_names=feature_names).transform(X)

    def to_schema(self) -> Dict:
        return {
            "feature_names": self.feature_names_ or [],
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
        }

    def save_schema(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_schema(), f, indent=2)

    @classmethod
    def load_schema(cls, path: str) -> "SplitSafeStandardScaler":
        with open(path, "r") as f:
            obj = json.load(f)
        sc = cls()
        sc.feature_names_ = obj.get("feature_names", None)
        mean = obj.get("mean", None)
        scale = obj.get("scale", None)
        sc.mean_ = None if mean is None else np.asarray(mean, dtype=float)
        sc.scale_ = None if scale is None else np.asarray(scale, dtype=float)
        return sc


__all__ = ["SplitSafeStandardScaler"]
