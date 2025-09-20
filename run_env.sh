#!/usr/bin/env bash
export FAST_EVAL_BIN="/home/crow/OtherNotes/TorpedoPaper/TorpedoCode/cpp/fast_eval"
# Default to non-strict TDA to avoid heavy paths during quick runs
export PAPER_TORPEDO_STRICT_TDA=0
# Target RTX 3090 (SM 86) for faster CUDA extension builds
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=1
