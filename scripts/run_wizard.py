#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


def prompt(msg: str, default: str | None = None) -> str:
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"{msg}{sfx}: ").strip()
    return val or (default or "")


def yesno(msg: str, default: bool = True) -> bool:
    sfx = " [Y/n]" if default else " [y/N]"
    val = input(f"{msg}{sfx}: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    import subprocess

    return subprocess.call(cmd)


def env_check():
    print("\n[Environment Check]")
    try:
        import pyarrow  # noqa: F401
        print("pyarrow: OK")
    except Exception as e:
        print("pyarrow: MISSING (pip install pyarrow)")
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load  # noqa: F401
        print("torch: OK")
    except Exception as e:
        print(f"torch: issue ({e}); ensure torch>=2.0 is installed")
    def _check(name: str):
        try:
            m = __import__(name)
            ver = getattr(m, "__version__", "?")
            print(f"{name}: OK (v{ver})")
        except Exception:
            print(f"{name}: missing (optional)")
    _check("ripser"); _check("gudhi"); _check("persim")
    # Native components
    import shutil as _sh
    print("maturin:", _sh.which("maturin") or "missing (optional)")
    print("cargo:", _sh.which("cargo") or "missing (optional)")
    print("nvcc:", _sh.which("nvcc") or "missing (optional)")


def build_native_step():
    if not yesno("Build native components now?", default=False):
        return
    # Incremental prompts
    if yesno("Build Rust pyo3 module (torpedocode_ingest)?", default=True):
        run([sys.executable, "scripts/build_native.py", "rust", "--verbose"])
    if yesno("Build Rust panel binary?", default=True):
        run([sys.executable, "scripts/build_native.py", "panel", "--verbose"])
    # Torch extension build
    has_cuda = _torch_cuda_available()
    if has_cuda:
        choice = prompt("Build Torch C++/CUDA extension? [cpu/cuda/skip]", default="cuda").lower()
        if choice.startswith("cu"):
            os.environ.setdefault("TORCH_EXTENSIONS_DIR", ".tmp/torch_extensions")
            run([sys.executable, "scripts/build_native.py", "torch-cuda", "--verbose"])
        elif choice.startswith("cp"):
            os.environ.setdefault("TORCH_EXTENSIONS_DIR", ".tmp/torch_extensions")
            run([sys.executable, "scripts/build_native.py", "torch", "--verbose"])
    else:
        if yesno("Build Torch C++ extension (CPU-only)?", default=False):
            os.environ.setdefault("TORCH_EXTENSIONS_DIR", ".tmp/torch_extensions")
            run([sys.executable, "scripts/build_native.py", "torch", "--verbose"])


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _topology_selected_exists(instrument: str, artifact_root: Path | None) -> bool:
    # Check common locations for topology_selected.json
    cands = []
    if artifact_root is not None:
        cands.append(artifact_root / instrument / "topology_selected.json")
    cands.append(Path("artifacts") / "topo" / instrument / "topology_selected.json")
    for p in cands:
        if p.exists():
            return True
    return False


def fast_eval_predictions(artifact_root: Path, instrument: str) -> None:
    # Evaluate predictions_test.csv under artifact_root/<instrument>/*/
    import glob
    pattern = str(artifact_root / instrument / "*" / "predictions_test.csv")
    files = glob.glob(pattern)
    if not files:
        print("[fast-eval] no predictions_test.csv found")
        return
    import sys as _sys
    for f in files:
        out = Path(f).with_name("eval_fast.json")
        # Try paired comparison if a sibling predictions file exists
        sib_pred2 = None
        for cand in ("predictions_test_b.csv", "predictions_test_baseline.csv"):
            p2 = Path(f).with_name(cand)
            if p2.exists():
                sib_pred2 = p2
                break
        if sib_pred2 is not None:
            # Merge on idx and label to produce combined CSV with pred and pred2 columns
            import pandas as _pd
            try:
                a = _pd.read_csv(f)
                b = _pd.read_csv(sib_pred2)
                # Normalize column names
                if "pred2" not in b.columns:
                    if "pred_b" in b.columns:
                        b = b.rename(columns={"pred_b": "pred2"})
                    elif "pred" in b.columns:
                        b = b.rename(columns={"pred": "pred2"})
                combo = a.merge(b[["idx", "pred2", "label"]], on=["idx", "label"], how="inner")
                combo_path = Path(f).with_name("predictions_test_combined.csv")
                combo.to_csv(combo_path, index=False)
                run([_sys.executable, "-m", "torpedocode.cli.eval", "--input", str(combo_path), "--pred2-col", "pred2", "--output", str(out)])
                print(f"[fast-eval] compared {Path(f).name} vs {sib_pred2.name} → {out}")
                continue
            except Exception as e:
                print(f"[fast-eval] merge for DeLong failed: {e}; falling back to single-model eval")
        # Single-model eval
        run([_sys.executable, "-m", "torpedocode.cli.eval", "--input", f, "--output", str(out)])
        print(f"[fast-eval] wrote {out}")


def option_crypto(cache_root: Path):
    print("Option A — Free crypto (Binance/Coinbase)")
    source = prompt("Choose source [binance/coinbase]", default="binance").lower()
    symbol = prompt("Symbol (e.g., BTCUSDT or BTC-USD)")
    raw_dir = Path(prompt("Raw dir (will create if downloading)", default="./raw")).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source == "binance":
        def _binance_url(sym: str, year: int, month: int) -> str:
            return f"https://data.binance.vision/data/spot/monthly/aggTrades/{sym}/{sym}-aggTrades-{year}-{month:02d}.zip"

        def _head_available(sym: str, pairs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if requests is None:
                print("[warn] requests not available; skipping availability check")
                return list(pairs)
            pairs_list = list(pairs)
            bar = (tqdm(pairs_list, desc="check", leave=False) if tqdm is not None else pairs_list)
            avail: List[Tuple[int, int]] = []
            for yy, mm in bar:
                url = _binance_url(sym, yy, mm)
                try:
                    r = requests.head(url, timeout=8)
                    if r.status_code == 200:
                        avail.append((yy, mm))
                    else:
                        print(f"[skip] {yy}-{mm:02d} not available (status {r.status_code})")
                except Exception as e:
                    print(f"[skip] {yy}-{mm:02d} head check failed: {e}")
            return avail

        if yesno("Prep last N complete months automatically?", default=True):
            try:
                n = int(prompt("N months", default="3"))
            except Exception:
                n = 3
            # Compute last N completed months as (year, month) pairs, crossing year boundary if needed
            now = datetime.now(timezone.utc)
            y = now.year
            m = now.month - 1  # last completed month
            pairs = []
            for _ in range(n):
                if m < 1:
                    y -= 1
                    m += 12
                pairs.append((y, m))
                m -= 1
            # Group by year and download per-year to avoid 404 for non-existent months
            out = raw_dir / f"binance_{symbol}.ndjson"
            # Start with an empty file
            try:
                out.unlink(missing_ok=True)
            except Exception:
                pass
            ok_any = False
            from collections import defaultdict
            # Preflight availability via HEAD checks
            avail_pairs = _head_available(symbol, pairs)
            if not avail_pairs:
                print("[warn] no available months found; falling back to manual prompts.")
            by_year = defaultdict(list)
            for yy, mm in (avail_pairs or pairs):
                by_year[yy].append(mm)
            for yy, months in by_year.items():
                months_sorted = sorted(months)
                code = run([sys.executable, "scripts/download_binance_monthly.py", "--symbol", symbol, "--year", str(yy), "--months", *[str(mo) for mo in months_sorted], "--output", str(out)])
                if code == 0:
                    ok_any = True
            if ok_any:
                print("Downloaded Binance monthly data.")
                goto_ingest = True
            else:
                print("[warn] downloader failed; months may be unavailable yet. Falling back to manual prompts.")
        # Manual monthly prompt
        if yesno("Download Binance Vision monthly aggTrades?", default=False):
            # Pre-fill defaults to last completed month trio
            now = datetime.now(timezone.utc)
            y = now.year
            m = now.month - 1
            if m < 1:
                y -= 1
                m += 12
            default_months = " ".join(str(x) for x in [max(1, m-2), max(1, m-1), m])
            year = int(prompt("Year", default=str(y)))
            months = prompt("Months (space-separated)", default=default_months).split()
            out = raw_dir / f"binance_{symbol}.ndjson"
            # HEAD check and progress bar for manual months
            try:
                months_int = [int(m) for m in months]
            except Exception:
                months_int = [int(x) for x in months if str(x).isdigit()]
            pairs = [(int(year), m) for m in months_int]
            avail_pairs = pairs
            if pairs:
                avail_pairs = _head_available(symbol, pairs)
                if not avail_pairs:
                    print("[warn] none of the requested months appear available; attempting anyway.")
            months_to_dl = [str(m) for (_, m) in (avail_pairs or pairs)]
            code = run([sys.executable, "scripts/download_binance_monthly.py", "--symbol", symbol, "--year", str(year), "--months", *months_to_dl, "--output", str(out)])
            if code != 0:
                print("[warn] downloader failed; falling back to manual JSONL path")
        else:
            print("Place Binance JSONL (websocket capture) under", raw_dir)
            print("Then run: uv run python scripts/binance_to_ndjson.py --input raw.jsonl --output cache/binance_SYMBOL.ndjson --symbol SYMBOL")
    else:
        if yesno("Prep last N months automatically?", default=True):
            try:
                n = int(prompt("N months", default="3"))
            except Exception:
                n = 3
            now = datetime.now(timezone.utc)
            # Approximate start n months ago (ignoring month length edge cases)
            m = now.month - (n - 1)
            y = now.year
            if m < 1:
                y -= 1
                m += 12
            start = f"{y:04d}-{m:02d}-01T00:00:00Z"
            end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            out = raw_dir / f"coinbase_{symbol}.ndjson"
            code = run([sys.executable, "scripts/download_coinbase_trades.py", "--product", symbol, "--start", start, "--end", end, "--output", str(out)])
            if code != 0:
                print("[warn] coinbase downloader failed; place JSONL under", raw_dir)
        elif yesno("Download Coinbase public trades via REST? (rate-limited)", default=False):
            start = prompt("Start ISO (UTC)", default="2024-06-01T00:00:00Z")
            end = prompt("End ISO (UTC)", default="2024-08-31T23:59:59Z")
            out = raw_dir / f"coinbase_{symbol}.ndjson"
            code = run([sys.executable, "scripts/download_coinbase_trades.py", "--product", symbol, "--start", start, "--end", end, "--output", str(out)])
            if code != 0:
                print("[warn] coinbase downloader failed; place JSONL under", raw_dir)
        else:
            print("Place Coinbase JSONL under", raw_dir)
            print("Then run: uv run python scripts/coinbase_to_ndjson.py --input raw.jsonl --output cache/coinbase_SYMBOL.ndjson --symbol SYMBOL")

    cache_root.mkdir(parents=True, exist_ok=True)
    # Ingest and cache
    code = run([sys.executable, "-m", "torpedocode.cli.ingest", "--raw-dir", str(raw_dir), "--cache-root", str(cache_root), "--instrument", symbol])
    if code != 0:
        sys.exit(code)
    # Optional topology grid search (validation)
    if yesno("Run topology grid search on validation?", default=False):
        topo_art = Path(prompt("Topology artifact dir", default=str(Path("./artifacts/topo") / symbol))).resolve()
        label_key = prompt("Label key", default="instability_s_5")
        run([sys.executable, "-m", "torpedocode.cli.topo_search", "--cache-root", str(cache_root), "--instrument", symbol, "--label-key", label_key, "--artifact-dir", str(topo_art)])
    # Quick report
    if yesno("Run quick multi-horizon report?", default=True):
        run([sys.executable, "-m", "torpedocode.cli.report_multi", "--cache-root", str(cache_root), "--instrument", symbol, "--output", str(cache_root / f"{symbol}_multi.json")])
    # Optional: batch train across all instruments in cache-root
    import glob
    caches = glob.glob(str(cache_root / "*.parquet"))
    if caches and yesno("Batch train across all instruments in cache-root?", default=False):
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        labels = prompt("Label keys (space-separated)", default="instability_s_1 instability_s_5").split()
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        # Discover instruments by cache files
        instruments = [Path(p).stem for p in caches]
        cmd = [sys.executable, "-m", "torpedocode.cli.batch_train", "--cache-root", str(cache_root), "--artifact-root", str(art), "--instruments", *instruments, "--label-keys", *labels, "--epochs", "2", "--device", device, "--use-topo-selected"]
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            # Run fast eval per instrument
            for inst in instruments:
                fast_eval_predictions(art, inst)
    # Train multi-horizon hybrid
    if yesno("Train multi-horizon hybrid?", default=True):
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        # Choose device (cuda if available and desired)
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        cmd = [sys.executable, "-m", "torpedocode.cli.train_multi", "--cache-root", str(cache_root), "--artifact-root", str(art), "--epochs", "3", "--device", device]
        if _topology_selected_exists(symbol, art):
            cmd.append("--use-topo-selected")
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            fast_eval_predictions(art, symbol)


def option_lobster(cache_root: Path):
    print("Option B — Equities via LOBSTER CSVs")
    multi = yesno("Ingest multiple month directories under a root?", default=False)
    raw_dirs: list[Path] = []
    if multi:
        root = Path(prompt("Root with monthly subdirs (YYYY-MM/*)" , default=".")).resolve()
        months = prompt("Months (space-separated, e.g., 2024-06 2024-07)", default="").split()
        for m in months:
            cand = root / m
            if cand.exists() and cand.is_dir():
                raw_dirs.append(cand)
        if not raw_dirs:
            print("[warn] no month dirs matched; falling back to single directory prompt")
            day_dir = Path(prompt("Path to LOBSTER day directory (contains message_*.csv and orderbook_*.csv)")).resolve()
            raw_dirs = [day_dir]
    else:
        day_dir = Path(prompt("Path to LOBSTER day directory (contains message_*.csv and orderbook_*.csv)")).resolve()
        raw_dirs = [day_dir]
    symbol = prompt("Instrument symbol (e.g., AAPL)")
    tick = prompt("Tick size (e.g., 0.01)", default="0.01")
    ca_csv = prompt("Corporate actions CSV (or blank)", default="").strip()
    ingest_cmd = [sys.executable, "-m", "torpedocode.cli.ingest", "--raw-dir", *[str(p) for p in raw_dirs], "--cache-root", str(cache_root), "--instrument", symbol, "--tick-size", tick]
    if ca_csv:
        ingest_cmd += ["--actions-csv", ca_csv, "--apply-actions"]
    code = run(ingest_cmd)
    if code != 0:
        sys.exit(code)
    # Optional topology grid search
    if yesno("Run topology grid search on validation?", default=False):
        topo_art = Path(prompt("Topology artifact dir", default=str(Path("./artifacts/topo") / symbol))).resolve()
        label_key = prompt("Label key", default="instability_s_5")
        run([sys.executable, "-m", "torpedocode.cli.topo_search", "--cache-root", str(cache_root), "--instrument", symbol, "--label-key", label_key, "--artifact-dir", str(topo_art)])
    if yesno("Train multi-horizon hybrid?", default=True):
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        cmd = [sys.executable, "-m", "torpedocode.cli.train_multi", "--cache-root", str(cache_root), "--artifact-root", str(art), "--epochs", "3", "--device", device]
        if _topology_selected_exists(symbol, art):
            cmd.append("--use-topo-selected")
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            fast_eval_predictions(art, symbol)


def option_itch_ouch(cache_root: Path):
    print("Option C — Equities via ITCH/OUCH")
    if not yesno("Do you already have ITCH/OUCH files?", default=False):
        print("Download of ITCH/OUCH is not automated (data access requires licensing).")
        print("If you obtain files, re-run this wizard and choose this option again.")
        return
    raw_dir = Path(prompt("Directory containing .itch/.ouch files")).resolve()
    symbol = prompt("Instrument symbol (e.g., AAPL)")
    spec = prompt("Vendor spec (e.g., nasdaq-itch-5.0)", default="nasdaq-itch-5.0")
    code = run([sys.executable, "-m", "torpedocode.cli.ingest", "--raw-dir", str(raw_dir), "--cache-root", str(cache_root), "--instrument", symbol, "--itch-spec", spec])
    if code != 0:
        sys.exit(code)
    # Optional topology grid search
    if yesno("Run topology grid search on validation?", default=False):
        topo_art = Path(prompt("Topology artifact dir", default=str(Path("./artifacts/topo") / symbol))).resolve()
        label_key = prompt("Label key", default="instability_s_5")
        run([sys.executable, "-m", "torpedocode.cli.topo_search", "--cache-root", str(cache_root), "--instrument", symbol, "--label-key", label_key, "--artifact-dir", str(topo_art)])
    if yesno("Train multi-horizon hybrid?", default=True):
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        cmd = [sys.executable, "-m", "torpedocode.cli.train_multi", "--cache-root", str(cache_root), "--artifact-root", str(art), "--epochs", "3", "--device", device]
        if _topology_selected_exists(symbol, art):
            cmd.append("--use-topo-selected")
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            fast_eval_predictions(art, symbol)


def main():
    print("TorpedoCode Run Wizard")
    if yesno("Run environment check?", default=True):
        env_check()
    build_native_step()
    cache_root = Path(prompt("Cache root", default="./cache")).resolve()
    print("\nGuided pipeline (per thesis steps):")
    print("  1) Acquire data (download/convert)")
    print("  2) Harmonize & cache (canonical parquet)")
    print("  3) Optional: Topology grid search (validation)")
    print("  4) Train multi-horizon hybrid (CPU)")
    print("  5) Evaluate & export artifacts")
    print("Select data option:")
    print("  1) Free crypto (Binance/Coinbase)")
    print("  2) LOBSTER CSVs (equities)")
    print("  3) ITCH/OUCH (equities; requires files)")
    choice = prompt("Enter 1/2/3", default="1")
    if choice == "1":
        option_crypto(cache_root)
    elif choice == "2":
        option_lobster(cache_root)
    else:
        option_itch_ouch(cache_root)
    # Aggregate across instruments (optional)
    if yesno("Aggregate across instruments for one label?", default=False):
        # Discover instruments by cached parquet files
        ins = sorted([p.stem for p in Path(cache_root).glob("*.parquet")])
        if len(ins) < 2:
            print("[note] Need at least 2 instruments under cache-root for aggregation.")
            return
        label = prompt("Label key to aggregate", default="instability_s_5")
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        # Batch train on CPU per instrument for this label set
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device for batch training [cpu/cuda]", default=dev_default)
        cmd = [sys.executable, "-m", "torpedocode.cli.batch_train", "--cache-root", str(cache_root), "--artifact-root", str(art), "--instruments", *ins, "--label-keys", label, "--epochs", "2", "--device", device]
        # Try to apply topology_selected.json per instrument when present
        cmd.append("--use-topo-selected")
        run(cmd)
        # Aggregate pooled micro/macro across instruments for this label
        pattern = f"*/{label}/predictions_test.csv"
        out = art / f"aggregate_{label}.json"
        run([sys.executable, "-m", "torpedocode.cli.aggregate", "--pred-root", str(art), "--pred-pattern", pattern, "--output", str(out)])


if __name__ == "__main__":
    main()
