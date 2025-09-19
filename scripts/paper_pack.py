#!/usr/bin/env python3
from __future__ import annotations

"""Pack paper artifacts into a single archive.

Collects schema/configs, predictions, evals, diagnostics, and metadata from an
artifact root and writes a ZIP for submission/sharing.

Example:
  uv run python scripts/paper_pack.py --artifact-root ./artifacts \
      --output ./artifacts/paper_bundle.zip
"""

import argparse
from pathlib import Path
import zipfile


GLOBS = [
    "**/feature_schema.json",
    "**/scaler_schema.json",
    "**/topology_selected.json",
    "**/tda_backends.json",
    "**/training_meta.json",
    "**/split_indices.json",
    "**/predictions_val.csv",
    "**/predictions_test.csv",
    "**/predictions_test_b.csv",
    "**/eval_*.json",
    "**/tpp_test_arrays.npz",
    "**/tpp_test_diagnostics.json",
    "**/aggregate_*.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack paper artifacts into a ZIP archive")
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = Path(args.artifact_root).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        added = 0
        for pat in GLOBS:
            for p in root.glob(pat):
                # Keep relative path inside archive under artifacts/
                arc = Path("artifacts") / p.relative_to(root)
                zf.write(p, arcname=str(arc))
                added += 1
        # Optionally include a minimal README
        readme = (
            "This bundle contains artifacts referenced in the paper: schemas, predictions,\n"
            "evaluation summaries, and diagnostics. Paths mirror the artifacts/ layout.\n"
        )
        import io

        zf.writestr("artifacts/README.txt", readme)
    print(f"Wrote {out} with {added} files")


if __name__ == "__main__":
    main()

