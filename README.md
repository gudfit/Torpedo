# TorpedoCode

CPU smoke testing with uv

- Create a local environment: `uv venv`
- Install project (optionally with dev extras): `uv pip install -e '.[dev]'`
- Run CPU smoke script: `uv run python scripts/cpu_smoke.py`
- Run tests: `uv run pytest -q`

CUDA paths are optional and disabled by default; use `scripts/run_training.py --cuda` only when a CUDA toolchain is available.

## Parsers, Preprocessing, and CLI

- ITCH/OUCH minimal binary parsers: `python/torpedocode/data/itch.py`, `python/torpedocode/data/ouch.py`
  - Map raw messages to the canonical event schema (timestamp UTC, event_type, size, price, level, side, symbol, venue)
  - Support additional common message types (e.g., delete, execute-with-price); unknown types are safely ignored
- LOBSTER parser: `python/torpedocode/data/lobster.py`
  - Merges message/orderbook CSVs and applies tick normalization
  - Directory pairing supported via `LOBPreprocessor` and `discover_lobster_pairs`
- Optional native acceleration:
  - C++ fast LOBSTER merger `cpp/src/lobster_fast.cpp`. Build and export:
    - `g++ -O3 -std=c++17 -o cpp/lobster_fast cpp/src/lobster_fast.cpp`
    - `export LOBSTER_FAST_BIN=$PWD/cpp/lobster_fast`
  - Rust/pybind11 hooks stubbed in `python/torpedocode/data/native.py` (attempt to import `torpedocode_ingest` if available)
- Directory ingestion CLI: `python -m torpedocode.cli.ingest --raw-dir RAW --cache-root CACHE --instrument SYMBOL [--tick-size TS]`

### Panel CLI (Rust)

- Build the Rust panel binary:
  - `UV_CACHE_DIR=.tmp/uvcache uv run python scripts/build_native.py panel`
  - Binary is installed to `bin/torpedocode-panel` and auto-detected by the Python wrapper.
- Usage (compute liquidity deciles and cross-market match groups):
  - `python -m torpedocode.cli.panel --input stats.csv --by liq_decile tick_size --output matched.json`
  - Or directly: `bin/torpedocode-panel --input stats.csv --by liq_decile tick_size --output matched.json`

### One-command native builds

- Use `scripts/build_native.py` to build components:
  - `python scripts/build_native.py all` → Rust pyo3 module, CPU-only Torch op (JIT), and C++ LOBSTER binary
  - `python scripts/build_native.py torch-cuda` → attempt full CUDA build of the Torch extension
  - `python scripts/build_native.py rust` / `torch` / `lobster` for individual pieces
  - The Python runtime can also attempt a CPU-only JIT build of the Torch op when `TORPEDOCODE_AUTO_BUILD_OPS=1` is set and the op is first used

### Free data quickstart (Binance)

- Convert Binance JSON lines to canonical NDJSON:
  - `python scripts/binance_to_ndjson.py --input raw.jsonl --output canon.ndjson --symbol BTCUSDT`
- Ingest and cache (if pyarrow installed):
  - `python -m torpedocode.cli.ingest --raw-dir . --cache-root ./cache --instrument BTCUSDT --verbose`

### Free data quickstart (Coinbase)

- Convert Coinbase JSON lines to canonical NDJSON:
  - `python scripts/coinbase_to_ndjson.py --input raw.jsonl --output canon.ndjson --symbol BTC-USD`
- Ingest and cache:
  - `python -m torpedocode.cli.ingest --raw-dir . --cache-root ./cache --instrument BTC-USD --verbose`

### One-liner convert + ingest

- Binance:
  - `python scripts/binance_to_ndjson.py --input raw.jsonl --output cache/binance.ndjson --symbol BTCUSDT && python -m torpedocode.cli.ingest --raw-dir cache --cache-root cache --instrument BTCUSDT`
- Coinbase:
  - `python scripts/coinbase_to_ndjson.py --input raw.jsonl --output cache/coinbase.ndjson --symbol BTC-USD && python -m torpedocode.cli.ingest --raw-dir cache --cache-root cache --instrument BTC-USD`

### Inspect environment

- Build-script check: `uv run python scripts/build_native.py check`
- Ingest check only: `uv run python -m torpedocode.cli.ingest --check --raw-dir . --cache-root ./cache --instrument DUMMY`

## Preprocessor

`LOBPreprocessor.harmonise(...)` accepts files or directories and will:

- Expand directories to candidate files (NDJSON/JSONL, ITCH/OUCH, LOBSTER CSVs)
- Auto-pair LOBSTER `...message_*.csv` with `...orderbook_*.csv`
- Use native bindings when present; fallback to pure Python parsers

## Optional knobs

- Perf tests

  - Enable native-vs-Python parser perf test: `TORPEDOCODE_RUN_PERF=1 UV_CACHE_DIR=.tmp/uvcache uv run pytest -q`

- Stream rotation
  - Rotate output files while streaming: add `--rotate-seconds N` to `scripts/stream_binance.py` and `scripts/stream_coinbase.py`.

## Benchmark tips

- The benchmark supports warm-up runs and environment capture for reproducibility:
  - Add `--warmup 1` (or higher) to discard initial runs.
  - Add `--env` to include CPU/GPU/Python info in the output JSON.

Example:

```
python -m torpedocode.bench.benchmark --levels 10 --T 5000 --batch 256 --window-s 5 --stride 5 --warmup 1 --env
```

## Topology grid (largest_cc)

- Run a small topology grid search on validation with the largest connected component rule for VR epsilon:
  - `python -m torpedocode.cli.topo_search --cache-root ./cache --instrument AAPL --label-key instability_s_5 --artifact-dir ./artifacts/topo`
  - By default, VR epsilon uses `largest_cc`; see `TopologyConfig.vr_epsilon_rule`.
### Guided Wizard (zero‑to‑results)

- Run the interactive wizard to follow the end‑to‑end pipeline (download → ingest → report → train):
  - `uv run python scripts/run_wizard.py`
  - Features:
    - Environment check (pyarrow/torch/TDA backends/toolchain)
    - Optional native builds (Rust pyo3, Rust panel, Torch C++ op)
    - Data options:
      - Binance/Coinbase crypto (download helpers included)
      - LOBSTER CSVs (tick size + optional corporate actions)
      - ITCH/OUCH (if you have files)
    - Optional topology grid search, quick report, and multi‑horizon training on CPU
- Quality checks: By default, NDJSON harmonisation applies light quality checks
  (drop NaT timestamps, nonpositive prices/sizes, and obvious duplicates). Toggle via
  `HarmoniseConfig(quality_checks=False)` if needed.

Ingest defaults and auctions/halts
- The `ingest` CLI defaults to NOT dropping auctions/halts to avoid surprises on synthetic/off-session data.
  For paper experiments, enable filtering explicitly via `--drop-auctions` and set `--session-tz` per market
  to align with session hours in the methodology.

uv-friendly tests
- This repo supports running tests via `uv` with a workspace-local cache to avoid sandbox permission issues:

  - `make test-uv` (uses `UV_CACHE_DIR=.tmp/uv`)
  - Or directly: `UV_CACHE_DIR=.tmp/uv uv run -q pytest -q`

Metadata and reproducibility
- Ingest writes a sidecar `<instrument>.ingest.json` with run parameters.
- Training persists `feature_schema.json`, `split_indices.json`, and optionally `temperature.json`.
- TDA backend availability/versions are recorded in `tda_backends.json` under the artifact directory.
- Aggregation supports Benjamini–Hochberg FDR control via the `aggregate` CLI.

Acceleration roadmap (CPU/GPU)
- The topological features module attempts to use an optional native module `torpedocode_tda` for VR ε/LCC steps; it falls back to pure NumPy when unavailable. GPU/CUDA kernels and
  broader C++/Rust offloads can be added behind the same API; training already supports an optional native fuse op via `torch.ops.torpedocode.hybrid_forward`.

Topology selection and walk-forward
- Run a lightweight topology search on one market: `python -m torpedocode.cli.topo_search ...` and reuse the selected JSON across
  batch/pooled/LOMO modes via `--topology-json`. The training CLIs honor this and persist the active topology schema.
- For walk-forward validation, `train` supports `--folds K` to produce K sequential folds with cumulative train and split val/test windows. Artifacts per fold are saved under
  `<artifact-dir>/fold_i/`.
