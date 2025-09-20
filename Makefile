PYTHON ?= .venv/bin/python

.PHONY: test test-uv smoke paper-smoke build-rust ci-regime-smoke

test:
	. .venv/bin/activate && pytest -q

test-uv:
	UV_CACHE_DIR=.tmp/uv UV_PYTHON=.venv/bin/python uv run --python .venv/bin/python -q pytest -q

.PHONY: test-uv-perf
test-uv-perf:
	UV_CACHE_DIR=.tmp/uv UV_PYTHON=.venv/bin/python bash scripts/run_tests_uv.sh --perf

.PHONY: test-uv-cpu
test-uv-cpu:
	UV_CACHE_DIR=.tmp/uv UV_PYTHON=.venv/bin/python bash scripts/run_tests_cpu.sh

.PHONY: build-native
build-native:
	UV_CACHE_DIR=.tmp/uv UV_PYTHON=.venv/bin/python uv run --python .venv/bin/python -q python scripts/build_native.py rust --verbose || true
	UV_CACHE_DIR=.tmp/uv UV_PYTHON=.venv/bin/python uv run --python .venv/bin/python -q python scripts/build_native.py lobster --verbose || true
	UV_CACHE_DIR=.tmp/uv UV_PYTHON=.venv/bin/python uv run --python .venv/bin/python -q python scripts/build_native.py panel --verbose || true

.PHONY: ci-cpu
ci-cpu: build-native test-uv-cpu

smoke:
	$(PYTHON) scripts/cpu_smoke.py

paper-smoke:
	$(PYTHON) scripts/paper_smoke.py

.PHONY: paper-smoke-gpu
paper-smoke-gpu:
	$(PYTHON) scripts/paper_smoke.py --device cuda

ci-regime-smoke-gpu:
	$(PYTHON) scripts/paper_smoke.py --device cuda
	@set -e; \
	for f in `ls artifacts_smoke/*/instability_s_1/predictions_test.csv 2>/dev/null || true`; do \
	  out=$${f%predictions_test.csv}predictions_test_with_ret.csv; \
	  echo "[ci] add ret -> $$out"; \
	  $(PYTHON) scripts/add_fake_returns.py --input $$f --output $$out; \
	  $(PYTHON) -m torpedocode.cli.metrics_regime --input $$out --output $${out%.csv}.metrics.json; \
	done

# Build optional Rust extensions (requires toolchain and network to fetch crates)
build-rust:
	RUSTFLAGS="-D warnings" UV_CACHE_DIR=.tmp/uv uv run -q python scripts/build_native.py rust --verbose

# CI helper: run paper smoke, add fake returns, and regime metrics on generated predictions
ci-regime-smoke: paper-smoke
	@set -e; \
	for f in `ls artifacts_smoke/*/instability_s_1/predictions_test.csv 2>/dev/null || true`; do \
	  out=$${f%predictions_test.csv}predictions_test_with_ret.csv; \
	  echo "[ci] add ret -> $$out"; \
	  $(PYTHON) scripts/add_fake_returns.py --input $$f --output $$out; \
	  $(PYTHON) -m torpedocode.cli.metrics_regime --input $$out --output $${out%.csv}.metrics.json; \
	done

.PHONY: wizard-deps
wizard-deps:
	UV_CACHE_DIR=.tmp/uv uv pip install --python .venv/bin/python tqdm

.PHONY: paper-pack
paper-pack:
	$(PYTHON) scripts/paper_pack.py --artifact-root ./artifacts --output ./artifacts/paper_bundle.zip
