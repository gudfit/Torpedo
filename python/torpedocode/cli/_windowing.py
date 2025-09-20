"""Utilities shared across CLIs for applying event-level sliding windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


DEFAULT_AUTO_WINDOW_EVENTS = 5_000_000


@dataclass(slots=True)
class WindowResolution:
    """Result of resolving a training window."""

    row_slice: Optional[slice]
    meta: Optional[Dict[str, int | str]]


def resolve_row_slice(
    *,
    total_rows: int,
    explicit_window: Optional[int],
    step: Optional[int],
    offset: int,
    allow_auto: bool,
    auto_cap: Optional[int] = None,
) -> WindowResolution:
    """Return the slice to apply for training windows along with logging metadata."""

    window = explicit_window if explicit_window is not None else None
    mode: Optional[str] = "explicit" if window is not None else None

    if window is None and allow_auto:
        cap = auto_cap if (auto_cap is not None and auto_cap > 0) else DEFAULT_AUTO_WINDOW_EVENTS
        if cap > 0 and total_rows > cap:
            window = cap
            mode = "auto"

    if window is None or window <= 0 or total_rows <= 0:
        return WindowResolution(row_slice=None, meta=None)

    window = max(1, int(window))
    step_val = step if (step is not None and step > 0) else window
    step_val = max(1, int(step_val))
    offset_val = max(0, int(offset))

    stop = total_rows - offset_val * step_val
    if stop <= 0:
        stop = min(total_rows, window)
    stop = min(total_rows, stop)
    start = max(0, stop - window)
    if start >= stop:
        start = max(0, stop - step_val)

    slc = slice(start, stop)
    meta = {
        "row_slice": [int(start), int(stop)],
        "events": int(stop - start),
        "total": int(total_rows),
        "mode": mode or "explicit",
        "step": int(step_val),
        "offset": int(offset_val),
    }
    return WindowResolution(row_slice=slc, meta=meta)
