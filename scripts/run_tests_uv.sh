#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-.tmp/uv}"
mkdir -p "$UV_CACHE_DIR"
echo "[info] Using UV_CACHE_DIR=$UV_CACHE_DIR"
uv run -q pytest -q "$@"

