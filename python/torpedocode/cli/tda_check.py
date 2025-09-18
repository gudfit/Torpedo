"""CLI: Validate TDA backend availability and versions.

Prints a JSON report with availability and version info for:
- ripser
- gudhi
- persim

Usage:
  python -m torpedocode.cli.tda_check [--output report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _check_mod(name: str, attr: str | None = None) -> dict:
    out = {"available": False, "version": None}
    try:
        mod = __import__(name)
        out["available"] = True
        ver = None
        for key in ("__version__", "version", "__VERSION__"):
            if hasattr(mod, key):
                ver = getattr(mod, key)
                break
        if ver is None and attr is not None and hasattr(mod, attr):
            try:
                ver = getattr(mod, attr).__version__  # type: ignore[attr-defined]
            except Exception:
                ver = None
        if isinstance(ver, bytes):
            ver = ver.decode("utf-8", errors="ignore")
        out["version"] = ver
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Check TDA backend availability")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    report = {
        "ripser": _check_mod("ripser"),
        "gudhi": _check_mod("gudhi"),
        "persim": _check_mod("persim"),
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

