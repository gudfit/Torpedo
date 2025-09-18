#!/usr/bin/env bash
set -euo pipefail

# Run all CPU-eligible tests (skip GPU/perf-only) with uv.
# Usage:
#   bash run_tests_cpu.sh            # quick, quiet
#   bash run_tests_cpu.sh --verbose  # show skip reasons

export UV_CACHE_DIR="${UV_CACHE_DIR:-.tmp/uv}"
echo "[info] Using UV_CACHE_DIR=$UV_CACHE_DIR"

REPORT="-q"
if [[ "${1:-}" == "--verbose" ]]; then
  REPORT="-r s -q"
fi

# Exclude CUDA/GPU/perf-only suites by keyword. This keeps all CPU tests.
CPU_EXPR="not cuda and not _gpu and not perf_gpu and not native_parsers_perf"

# Optional: build native components first (best-effort, non-fatal)
uv run -q python scripts/build_native.py rust --verbose || true
uv run -q python scripts/build_native.py lobster --verbose || true
uv run -q python scripts/build_native.py panel --verbose || true

uv run -q pytest ${REPORT} -k "$CPU_EXPR"
