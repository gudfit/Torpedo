#!/usr/bin/env bash
set -euo pipefail
export FAST_EVAL_BIN="/home/crow/OtherNotes/TorpedoPaper/TorpedoCode/cpp/fast_eval"
export PAPER_TORPEDO_STRICT_TDA=1
export OMP_NUM_THREADS=8
# Example: per-instrument training with LO/CX@level expansion and topology
# Replace CACHE_ROOT, ARTIFACT_ROOT, INSTRUMENT, LABEL_KEY as needed
CACHE_ROOT=./cache
ARTIFACT_ROOT=./artifacts
INSTRUMENT=AAPL
LABEL_KEY=instability_s_1
uv run python -m torpedocode.cli.train \
+--instrument "$INSTRUMENT" \
+--label-key "$LABEL_KEY" \
+--artifact-dir "$ARTIFACT_ROOT/$INSTRUMENT/$LABEL_KEY" \
+--epochs 3 --batch 128 --bptt 64 --topo-stride 5 --device cpu \
+--expand-types-by-level

# HOWTO: Count/EWMA options for event-type flow
#  - Add: --count-windows-s 1 5 10   to control causal count windows (seconds)
#  - Add: --ewma-halflives-s 1.0 5.0 to add exponentially decayed counts with half-lives (seconds)
#
# HOWTO: Persistence image quick overrides
#  - Add: --pi-res 128  --pi-sigma 0.02   to match paper configs
