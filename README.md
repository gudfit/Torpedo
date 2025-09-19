# TorpedoCode

Quick start (CPU)

- Create env: `uv venv`
- Install: `uv pip install -e '.[dev]'`
- Run smoke: `uv run python scripts/cpu_smoke.py`
- Tests: `UV_CACHE_DIR=.tmp/uv uv run -q pytest -q`

CUDA is optional; only enable if your toolchain supports it.

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
  - Session filtering: By default, ingestion excludes auctions/halts using market local hours (09:30–16:00 America/New_York by default). If your data is outside session (e.g., midnight UTC in tests), pass `--no-drop-auctions`.

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

Note: `torpedocode.data.preprocessing` is a shim that re‑exports `LOBPreprocessor` from
`torpedocode.data.pipeline` to preserve backward compatibility with older imports.

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

## Paper Pipeline (One-Click)

This mirrors the paper’s end-to-end protocol: cache → topology selection → train (with PH) → evaluate/aggregate → economic checks.

1) Cache raw feeds (NDJSON/JSONL or ITCH/OUCH/LOBSTER) with labels

```
python -m torpedocode.cli.cache \
  --input ./data/ndjson_or_dir \
  --cache-root ./cache \
  --instrument AAPL \
  --drop-auctions --tick-size 0.01 --levels 10 \
  --horizons-s 1 5 10 --horizons-events 100 500 --eta 0.0
```

2) Select a topology configuration on validation (optional but recommended)

```
python -m torpedocode.cli.topo_search \
  --cache-root ./cache --instrument AAPL --label-key instability_s_5 \
  --artifact-dir ./artifacts/topo/AAPL --strict-tda
# writes ./artifacts/topo/AAPL/topology_selected.json
```

3) Run the manifest pipeline (caches → batch train → aggregate)

Create `paper_manifest.yaml`:

```
data:
  input: ./cache                 # points at cached parquet(s) directory
  cache_root: ./cache
  instrument: AAPL               # or omit when using multiple caches per-file
  drop_auctions: true
  tick_size: 0.01
  levels: 10
  horizons_s: [1, 5, 10]
  horizons_events: [100, 500]
  eta: 0.0

train:
  artifact_root: ./artifacts
  label_key: instability_s_5
  epochs: 3
  batch: 128
  bptt: 64
  topo_stride: 5
  device: cpu
  temperature_scale: true
  tpp_diagnostics: true
  # Topology options forwarded to batch_train:
  use_topo_selected: true        # reuse topology_selected.json if present under artifacts/topo/<inst>/
  # Alternatively, reference an explicit JSON:
  # topology_json: ./artifacts/topo/AAPL/topology_selected.json
  # Optional persistence image tweaks:
  # pi_res: 128
  # pi_sigma: 0.05

aggregate:
  mode: pred
  output: ./artifacts/aggregate.json
  block_bootstrap: true
  block_length: 50
  n_boot: 200
```

Run it:

```
python -m torpedocode.cli.manifest --manifest paper_manifest.yaml
```

4) Economic significance (VaR/ES, Kupiec/Christoffersen) for a given instrument

```
python -m torpedocode.cli.economic \
  --input ./artifacts/AAPL/instability_s_5/predictions_test.csv \
  --bootstrap-ci --threshold-sweep --alpha 0.99
```

Notes:
- The manifest now forwards topology parameters to `batch_train.py` (`use_topo_selected`, `topology_json`, `pi_res`, `pi_sigma`).
- Training writes TPP arrays and diagnostics (`tpp_test_arrays.npz`, `tpp_test_diagnostics.json`), including per‑type KS p‑values.

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
- For landscapes, you can sweep resolutions via `--landscape-res 64 128` in `cli.topo_search` in addition to levels `K`.
- For walk-forward validation, `train` supports `--folds K` to produce K sequential folds with cumulative train and split val/test windows. Artifacts per fold are saved under
  `<artifact-dir>/fold_i/`.

Manual quick commands (if you prefer)
- `. .venv/bin/activate`
- `make build-rust`
- `python -c "import torpedocode_tda, torpedocode_ingest; print('native OK')"`
- `pytest -q`
- `uv pip install pyarrow`
- `make ci-regime-smoke`

## Event Mapping Notes (LOBSTER/ITCH/OUCH)

- Side-aware signs (MO±/CX±):
  - Python LOBSTER parser and the C++ `lobster_fast` now emit `MO+` for buy and `MO-` for sell trades (and similarly for `CX±`) when side is available. Enables optional use of the canonical set {MO±, LO±_ℓ, CX±_ℓ} described in the paper.
  - Enable via `DataConfig(side_aware_events=True)` for ingestion; otherwise default (+) mapping is kept for backward compatibility.
- Level tagging for LOBSTER:
  - Python and C++ LOBSTER paths attempt to infer the `level` by matching the message price to the nearest book level on the same row (within 0.5 tick). This is best‑effort and depends on input alignment.
  - Use `--expand-types-by-level` in training to expand `LO/CX` types to `type@level` as needed.
- ITCH/OUCH:
  - ITCH minimal parser sets `MO±` on trade messages using the side bit when present.
  - OUCH minimal parser sets `LO±` by side; other messages are mapped with default signs due to limited payload. Extending full side tracking requires a deeper order book reconstruction (future work).

## Fast Evaluation (Wizard)

- The wizard’s fast evaluation avoids subprocess overhead and computes metrics in‑process. It uses:
  - AUROC, AUPRC, Brier, ECE, and optional paired DeLong comparison when a sibling predictions file is detected.
  - A progress bar via `tqdm` if installed (`pip install tqdm`).
  - Output JSON: `eval_fast.json` alongside `predictions_test.csv`.
  - Optional native path: set `FAST_EVAL_BIN` to point to a built C++ tool (`cpp/src/fast_eval.cpp`) to compute single‑model metrics quickly:
    - `FAST_EVAL_OPENMP=1 uv run python scripts/build_native.py fast-eval` (enables `-fopenmp`)
    - `export FAST_EVAL_BIN=$PWD/cpp/fast_eval`
    - `export OMP_NUM_THREADS=$(nproc)` (or desired threads)
    - Wizard will use it when no paired comparison is needed.

### Wizard Run Env

- The wizard can write `run_env.sh` exporting:
  - `FAST_EVAL_BIN`, `PAPER_TORPEDO_STRICT_TDA=1`, and `OMP_NUM_THREADS` (defaults to CPU cores)
- It can also generate `train_cmd.sh` exporting the same and running train with `--expand-types-by-level`.

### Paper Packager

- Bundle key artifacts (schemas, predictions, evals, diagnostics) into a single archive for submission:

```
uv run python scripts/paper_pack.py --artifact-root ./artifacts --output ./artifacts/paper_bundle.zip
```

- Or via Makefile:

```
make paper-pack
```

- Contents include `feature_schema.json`, `scaler_schema.json`, `topology_selected.json`, `tda_backends.json`, `predictions_*.csv`, `eval_*.json`, and TPP diagnostics.

## New helpers and flags (methodology alignment)

- Shared IO + helpers
  - `torpedocode.evaluation.io`: `load_preds_labels_csv/npz` to avoid duplicate CSV/NPZ readers.
  - `torpedocode.evaluation.helpers`:
    - `write_tda_backends_json(path)`: writes availability/version for `torpedocode_tda`, `ripser`, `gudhi`, `persim`.
    - `save_tpp_arrays_and_diagnostics(dir, intensities, event_type_ids, delta_t)`: emits `tpp_test_arrays.npz` and `tpp_test_diagnostics.json`.
    - `temperature_scale_from_probs(preds, labels)`: fits temperature on logits derived from probabilities and returns calibrated metrics.

- Economic CLI (`python -m torpedocode.cli.economic`)
  - New flag `--threshold-sweep` to emit threshold sensitivity (precision/recall/F1) over a default grid. Works with optional `--val-*` inputs.
  - New flags `--bootstrap-ci --boot-n N --block-l L` to report VaR/ES confidence intervals via stationary block bootstrap.

- Feature engineering controls
  - `train.py` / `batch_train.py` accept:
    - `--count-windows-s 1 5 ...` to control event-type count windows (seconds).
    - `--ewma-halflives-s 1.0 5.0 ...` to add exponentially decayed event counts with given half-lives.
  - Internally, `features.lob.build_lob_feature_matrix` now computes:
    - Event-type counts per window and total.
    - EWMA counts per type and total (time-aware via Δt), appended to the counts block.
    - Additional temporal covariates: normalized time-of-day (`tod_progress`) and day-of-week (`dow_sin`, `dow_cos`).
  - Queue ages: exact ages use `last_update_*` columns when present; fallback ages now incorporate event-aware resets for LO/CX at specific levels and MO± at best opposite side, reducing false positives versus pure change detection.

- Topological controls
  - `TopologyConfig` adds:
    - `vr_zscore` (default True) to z-score features per window for VR complexes.

## Run Wizard Step-by-Step

- Launch: `uv run python scripts/run_wizard.py`
- The wizard guides you through:
  - Environment check and optional native builds (Rust panel, Torch C++/CUDA op, C++ fast_eval)
  - Data choice: Binance/Coinbase, LOBSTER CSVs, or ITCH/OUCH
  - Harmonize & cache into Parquet
  - Optional: Topology grid search on a validation slice
  - Optional: CTMC pretrain (synthetic) and warm‑start training
  - Train multi‑horizon hybrid (CPU/GPU)
  - Fast eval + DeLong where available (writes `eval_fast.json`)
  - Optional: Aggregate across instruments

Notes
- The wizard will prompt for device and paths; defaults are shown in brackets.
- CTMC pretraining produces a checkpoint and threads it into training automatically.
- You can re‑run the wizard at any time; it will detect available artifacts.

### Save Checkpoints for Reproducibility

- Save weights from the simple train CLI:

```
python -m torpedocode.cli.train \
  --instrument AAPL \
  --label-key instability_s_5 \
  --artifact-dir ./artifacts/AAPL/instability_s_5 \
  --epochs 3 --batch 128 --bptt 64 --device cpu \
  --save-state-dict ./artifacts/AAPL/model.pt
```

- Pretrain + warm‑start:

```
python -m torpedocode.cli.pretrain_ctmc \
  --epochs 3 --steps 400 --batch 64 --T 128 --hidden 128 --layers 1 \
  --device cpu --output ./artifacts/pretrained/model.pt

python -m torpedocode.cli.train_multi \
  --cache-root ./cache \
  --artifact-root ./artifacts \
  --epochs 3 --device cpu \
  --warm-start ./artifacts/pretrained/model.pt
```

## Methodology Coverage

- Feature engineering (conventional)
  - Implemented in `python/torpedocode/features/lob.py`:
  - Depths and cumulative depths, multi‑scale imbalance, spreads/mid and returns, inter‑event time, event‑type counts and exponentially decayed counts, time‑of‑day and day‑of‑week cyclical encodings, and queue ages (with optional Rust fast paths). Event‑aware resets (LO/CX at their levels and MO± at best opposite side) reduce false positives for ages.

- Topological features (TDA)
  - Implemented in `python/torpedocode/features/topological.py` with `TopologyConfig`:
  - Cubical complexity via a liquidity surface, Vietoris–Rips with optional per‑window z‑scoring; persistence landscapes and images; auto‑range for images; VR epsilon strategies (largest CC or MST quantile); strict/fallback toggles; stride/windowing.
  - Grid selection and reproducibility: `python/torpedocode/cli/topo_search.py` searches windows and vectorisations; `cli/train.py` writes `feature_schema.json`, records the active topology config (and image ranges), and supports reusing a selected topology JSON.

- Hybrid model architecture and loss
  - Implemented in `python/torpedocode/models/hybrid.py` and `python/torpedocode/training/losses.py`:
  - LSTM over concatenated `[x_t, z_t, e_mkt]`, per‑type intensity heads with softplus and topology skip, log‑normal mark head, and a classifier head.
  - Exact TPP compensator using piecewise‑constant intensities, per‑event log λ(m_i) + compensator + log‑normal mark NLL. Smoothness penalty uses time differences. Weight decay is applied in the loss (optimizer `weight_decay=0.0` to avoid double‑counting; see `training/pipeline.py`).
  - Calibration: ECE and Brier in `evaluation/metrics.py`. Optional temperature scaling via `cli/train.py` and `TrainingConfig.apply_temperature_scaling`.

- Pretraining CTMC
  - Implemented in `python/torpedocode/data/synthetic_ctmc.py` with Cont‑style queues, state‑dependent intensities, marks, optional topology, and per‑level event expansion (LO/CX; optional MO by level).
  - CLI in `python/torpedocode/cli/pretrain_ctmc.py` trains a small model and saves a checkpoint. The generator remaps event types to the requested `--num-event-types` (e.g., 6 → keep; 4 → collapse CX into corresponding LO side; 2 → collapse by side) to match loss head indexing.
  - (Legacy simulator removed) Prior generic simulator has been removed in favor of `synthetic_ctmc`.

- Experimental protocol
  - Walk‑forward/train splits: `LOBDatasetBuilder.build_splits` and `.build_walkforward_splits()`; TBPTT, balanced windows, gradient clipping in `cli/train.py` + `training/pipeline.py`.
  - Multi‑instrument/multi‑horizon orchestration: `cli/batch_train.py` and `cli/train_multi.py`; reporting across horizons in `cli/report_multi.py`.
  - Economic significance / backtests: `cli/economic.py` and `evaluation/economic.py` (VaR/ES, Kupiec/Christoffersen, realized‑volatility regime splits, utility‑based threshold selection). The CLI exposes bootstrap CIs for VaR/ES via `--bootstrap-ci`.
  - Panel/liquidity matching: `cli/panel.py` with `compute_liquidity_panel` and `match_instruments_across_markets` in `data/preprocess.py` (optional Rust `bin/torpedocode-panel`).

    - `cubical_scalar_field` to choose the scalar field for cubical complexes (`imbalance` [default], `bid`, `ask`, `net`).
  - Persistence images: you can set `image_resolution` up to 128 and `image_bandwidth` (e.g., 0.02/0.05) via topology JSON or CLI selection.

All changes preserve existing defaults and tests while exposing the paper’s parameterizations where needed.
## Cross-Market Orchestration

- Build a matched panel via `python -m torpedocode.cli.panel ...` (produces CSV/JSON with columns including `market,symbol`).
- Run pooled or leave‑one‑market‑out (LOMO) evaluations with a fast logistic baseline:
  - `python -m torpedocode.cli.cross_market --panel matched.csv --cache-root ./cache --label-key instability_s_5 --mode pooled --with-tda --output pooled.json`
  - Use `--mode lomo` for LOMO. Outputs include per‑market metrics and global micro/macro aggregates.
