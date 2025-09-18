#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-.tmp/uv}"
mkdir -p "$UV_CACHE_DIR"
echo "[info] Using UV_CACHE_DIR=$UV_CACHE_DIR"
# Best-effort native builds (Rust/CPP) before tests; non-fatal if toolchains missing
uv run -q python scripts/build_native.py rust --verbose || true
uv run -q python scripts/build_native.py lobster --verbose || true
uv run -q python scripts/build_native.py panel --verbose || true
if [ "${1:-}" = "--perf" ]; then
  shift || true
  echo "[info] Running perf tests (TORPEDOCODE_RUN_PERF=1)"
  TORPEDOCODE_RUN_PERF=1 uv run -q pytest -q -k native_parsers_perf "$@"
else
  uv run -q pytest -q "$@"
fi
