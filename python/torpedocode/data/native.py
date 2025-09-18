"""Optional native parser bindings (Rust/pybind11).

If the native module is present (e.g., built with pyo3/pybind11), use it for speed.
Exports the same high-level functions as the pure-Python fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def parse_itch_native(
    path: Path | str,
    *,
    tick_size: Optional[float] = None,
    symbol: Optional[str] = None,
    spec: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    try:
        import torpedocode_ingest as ingest  # type: ignore
    except Exception:
        return None
    try:
        obj = ingest.parse_itch_file(
            str(path), tick_size=tick_size or 0.0, symbol=symbol or "", spec=spec or ""
        )
        if hasattr(obj, "__len__") and len(obj) == 0:
            return None
        import pandas as pd  # type: ignore

        df = pd.DataFrame(obj)
        # Ensure timestamp is UTC datetime
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, unit="ns")
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return None


def parse_ouch_native(
    path: Path | str,
    *,
    tick_size: Optional[float] = None,
    symbol: Optional[str] = None,
    spec: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    try:
        import torpedocode_ingest as ingest  # type: ignore
    except Exception:
        return None
    try:
        obj = ingest.parse_ouch_file(
            str(path), tick_size=tick_size or 0.0, symbol=symbol or "", spec=spec or ""
        )
        if hasattr(obj, "__len__") and len(obj) == 0:
            return None
        import pandas as pd  # type: ignore

        df = pd.DataFrame(obj)
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, unit="ns")
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return None
