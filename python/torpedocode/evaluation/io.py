"""Shared IO helpers for loading predictions/labels from CSV/NPZ.

Avoids duplicated loaders across CLI tools.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_preds_labels_csv(
    path: Path,
    pred_col: str = "pred",
    label_col: str = "label",
    ret_col: str | None = None,
    pred2_col: str | None = None,
):
    import pandas as pd

    df = pd.read_csv(path)
    p = df[pred_col].to_numpy(dtype=float)
    y = df[label_col].astype(int).to_numpy() if label_col in df.columns else None
    r = df[ret_col].to_numpy(dtype=float) if (ret_col and ret_col in df.columns) else None
    p2 = df[pred2_col].to_numpy(dtype=float) if (pred2_col and pred2_col in df.columns) else None
    return p, y, r, p2


def load_preds_labels_npz(
    path: Path,
    pred_key: str = "pred",
    label_key: str = "label",
    ret_key: str | None = None,
    pred2_key: str | None = None,
):
    obj = np.load(path, allow_pickle=False)
    p = obj[pred_key]
    y = obj[label_key] if label_key in obj else None
    r = obj[ret_key] if (ret_key and ret_key in obj) else None
    p2 = obj[pred2_key] if (pred2_key and pred2_key in obj) else None
    return p, y, r, p2


__all__ = ["load_preds_labels_csv", "load_preds_labels_npz"]
