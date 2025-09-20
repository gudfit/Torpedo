#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-.tmp/uv}"
# Force uv to target the project venv by default
export UV_PYTHON="${UV_PYTHON:-.venv/bin/python}"
mkdir -p "$UV_CACHE_DIR"
echo "[info] Using UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[info] Using UV_PYTHON=$UV_PYTHON"
# Best-effort native builds (Rust/CPP) before tests; non-fatal if toolchains missing
uv run --python "$UV_PYTHON" -q python scripts/build_native.py rust --verbose || true
uv run --python "$UV_PYTHON" -q python scripts/build_native.py lobster --verbose || true
uv run --python "$UV_PYTHON" -q python scripts/build_native.py panel --verbose || true
if [ "${1:-}" = "--perf" ]; then
  shift || true
  echo "[info] Running perf tests (TORPEDOCODE_RUN_PERF=1)"
  TORPEDOCODE_RUN_PERF=1 uv run --python "$UV_PYTHON" -q pytest -q -k native_parsers_perf "$@"
else
  uv run --python "$UV_PYTHON" -q pytest -q "$@"
fi
