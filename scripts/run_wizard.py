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


def _summary_banner(artifact_root: Path, instrument: str, *, label_key: str | None = None) -> None:
    """Print a compact summary of key artifact paths for one instrument.

    If label_key is None, summarize all label subfolders under the instrument.
    """
    base = artifact_root / instrument
    if label_key is None:
        import glob
        pattern = str(base / "*" / "predictions_test.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print("[Summary] No predictions found under", base)
            return
        print("\n[Summary — Artifacts]")
        for f in files:
            p = Path(f)
            d = p.parent
            print(f"- [{d.name}] Predictions (test): {p}")
            val = d / "predictions_val.csv"
            if val.exists():
                print(f"  Predictions (val):  {val}")
            ev = d / "eval_fast.json"
            if ev.exists():
                print(f"  Fast eval:          {ev}")
            print(f"  Schemas:            {d / 'feature_schema.json'}, {d / 'scaler_schema.json'}")
            print(f"  Splits:             {d / 'split_indices.json'}")
            print(f"  TPP arrays/diag:    {d / 'tpp_test_arrays.npz'}, {d / 'tpp_test_diagnostics.json'}")
            print(f"  Training meta:      {d / 'training_meta.json'}")
            print(f"  TDA backends:       {d / 'tda_backends.json'}")
    else:
        d = base / label_key
        print("\n[Summary — Artifacts]")
        print(f"- Predictions (test): {d / 'predictions_test.csv'}")
        print(f"- Predictions (val):  {d / 'predictions_val.csv'}")
        ev = d / 'eval_fast.json'
        if ev.exists():
            print(f"- Fast eval:          {ev}")
        print(f"- Schemas:            {d / 'feature_schema.json'}, {d / 'scaler_schema.json'}")
        print(f"- Splits:             {d / 'split_indices.json'}")
        print(f"- TPP arrays/diag:    {d / 'tpp_test_arrays.npz'}, {d / 'tpp_test_diagnostics.json'}")
        print(f"- Training meta:      {d / 'training_meta.json'}")
        print(f"- TDA backends:       {d / 'tda_backends.json'}")


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

def _has_predictions(artifact_root: Path, instrument: str) -> bool:
    import glob as _glob
    pattern = str(Path(artifact_root) / instrument / "*" / "predictions_test.csv")
    files = _glob.glob(pattern)
    return len(files) > 0


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

    _check("ripser")
    _check("gudhi")
    _check("persim")
    import shutil as _sh

    print("maturin:", _sh.which("maturin") or "missing (optional)")
    print("cargo:", _sh.which("cargo") or "missing (optional)")
    print("nvcc:", _sh.which("nvcc") or "missing (optional)")
    try:
        from torpedocode.features.topological import topology_backend_status

        status = topology_backend_status()
        cuda_info = status.get("torch_cuda_extension", {})
        visible = bool(cuda_info.get("cuda_visible"))
        compiled = bool(cuda_info.get("compiled"))
        op_reg = bool(cuda_info.get("op_registered"))
        session = cuda_info.get("session_backends", [])
        parts = [
            f"torch_cuda_extension: cuda_visible={visible}",
            f"compiled={compiled}",
            f"op_registered={op_reg}",
        ]
        if session:
            parts.append("session_backends=" + ",".join(str(s) for s in session))
        err = cuda_info.get("error")
        if err:
            parts.append(f"error={err}")
        print(" ".join(parts))
    except Exception as e:  # pragma: no cover - optional torch dependency
        print(f"torch_cuda_extension: status unavailable ({e})")


def build_native_step():
    if not yesno("Build native components now?", default=False):
        return
    if yesno("Build Rust pyo3 module (torpedocode_ingest)?", default=True):
        run([sys.executable, "scripts/build_native.py", "rust", "--verbose"])
    if yesno("Build Rust panel binary?", default=True):
        run([sys.executable, "scripts/build_native.py", "panel", "--verbose"])
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
    if yesno("Build C++ lobster_fast merger?", default=True):
        run([sys.executable, "scripts/build_native.py", "lobster", "--verbose"])
        lb = Path("cpp/lobster_fast").resolve()
        if lb.exists():
            os.environ["LOBSTER_FAST_BIN"] = str(lb)
            print(f"[wizard] LOBSTER_FAST_BIN set for this session → {lb}")
    if yesno("Build C++ fast evaluator (fast_eval)?", default=True):
        run([sys.executable, "scripts/build_native.py", "fast-eval", "--verbose"])
        fb = Path("cpp/fast_eval").resolve()
        if fb.exists():
            os.environ["FAST_EVAL_BIN"] = str(fb)
            print(f"[wizard] FAST_EVAL_BIN set for this session → {fb}")
            # Offer to set OpenMP threads for this session now (or auto if env set)
            if os.environ.get("WIZARD_OMP_AUTOCONFIG", "0").lower() in {"1", "true"}:
                th = str(os.cpu_count() or 8)
                os.environ["OMP_NUM_THREADS"] = th
                print(f"[wizard] OMP_NUM_THREADS={th} (session, auto)")
            else:
                if yesno("Set OMP_NUM_THREADS for this session now?", default=True):
                    cores = os.cpu_count() or 8
                    th = prompt("OMP_NUM_THREADS", default=str(cores))
                    os.environ["OMP_NUM_THREADS"] = str(th)
                    print(f"[wizard] OMP_NUM_THREADS={th} (session)")
            if yesno(
                "Write run_env.sh with FAST_EVAL_BIN and PAPER_TORPEDO_STRICT_TDA?", default=True
            ):
                env_sh = Path("run_env.sh").resolve()
                with env_sh.open("w") as f:
                    f.write("#!/usr/bin/env bash\n")
                    f.write(f'export FAST_EVAL_BIN="{fb}"\n')
                    f.write("export PAPER_TORPEDO_STRICT_TDA=1\n")
                    if os.environ.get("WIZARD_OMP_AUTOCONFIG", "0").lower() in {"1", "true"}:
                        th = str(os.cpu_count() or 8)
                        f.write(f"export OMP_NUM_THREADS={th}\n")
                    else:
                        if yesno("Set OMP_NUM_THREADS in run_env.sh?", default=True):
                            cores = os.cpu_count() or 8
                            th = prompt("OMP_NUM_THREADS", default=str(cores))
                            f.write(f"export OMP_NUM_THREADS={th}\n")
                os.chmod(env_sh, 0o755)
                print(f"[wizard] Wrote {env_sh}")
            if yesno("Generate train_cmd.sh with recommended flags?", default=True):
                train_sh = Path("train_cmd.sh").resolve()
                with train_sh.open("w") as f:
                    f.write("#!/usr/bin/env bash\n")
                    f.write("set -euo pipefail\n")
                    f.write(f'export FAST_EVAL_BIN="{fb}"\n')
                    f.write("export PAPER_TORPEDO_STRICT_TDA=1\n")
                    cores = os.cpu_count() or 8
                    f.write(f'export OMP_NUM_THREADS={cores}\n')
                    f.write(
                        "# Example: per-instrument training with LO/CX@level expansion and topology\n"
                    )
                    f.write(
                        "# Replace CACHE_ROOT, ARTIFACT_ROOT, INSTRUMENT, LABEL_KEY as needed\n"
                    )
                    f.write("CACHE_ROOT=./cache\n")
                    f.write("ARTIFACT_ROOT=./artifacts\n")
                    f.write("INSTRUMENT=AAPL\n")
                    f.write("LABEL_KEY=instability_s_1\n")
                    device_flag = "cuda" if _torch_cuda_available() else "cpu"
                    cmd_lines = [
                        "uv run python -m torpedocode.cli.train \\",
                        "--instrument "$INSTRUMENT" \\",
                        "--label-key "$LABEL_KEY" \\",
                        "--artifact-dir "$ARTIFACT_ROOT/$INSTRUMENT/$LABEL_KEY" \\",
                        "--epochs 3 --batch 128 --bptt 64 --topo-stride 5 "
                        f"--device {device_flag} \\",
                        "--expand-types-by-level",
                    ]
                    f.write("\n".join(cmd_lines) + "\n")
                    f.write("\n# HOWTO: Count/EWMA options for event-type flow\n")
                    f.write("#  - Add: --count-windows-s 1 5 10   to control causal count windows (seconds)\n")
                    f.write("#  - Add: --ewma-halflives-s 1.0 5.0 to add exponentially decayed counts with half-lives (seconds)\n")
                    f.write("#\n# HOWTO: Persistence image quick overrides\n")
                    f.write("#  - Add: --pi-res 128  --pi-sigma 0.02   to match paper configs\n")
                    # Optional interactive line with user-selected flags
                    if yesno("Add a second command line with custom count/EWMA/PI flags?", default=False):
                        base_cmd_parts = [
                            "uv run python -m torpedocode.cli.train \\",
                            "    --instrument \"$INSTRUMENT\" \\",
                            "    --label-key \"$LABEL_KEY\" \\",
                            "    --artifact-dir \"$ARTIFACT_ROOT/$INSTRUMENT/$LABEL_KEY\" \\",
                            "    --epochs 3 --batch 128 --bptt 64 --topo-stride 5 "
                            f"--device {device_flag} \\",
                            "    --expand-types-by-level",
                        ]
                        cw = prompt("Count windows (seconds, space-separated)", default="1 5").strip()
                        hl = prompt("EWMA half-lives (seconds, space-separated)", default="1.0 5.0").strip()
                        res = prompt("PI resolution", default="128").strip()
                        sig = prompt("PI sigma", default="0.02").strip()
                        if cw:
                            base_cmd_parts.append(f"    +--count-windows-s {cw} \\")
                        if hl:
                            base_cmd_parts.append(f"    +--ewma-halflives-s {hl} \\")
                        if res:
                            base_cmd_parts.append(f"    +--pi-res {res} \\")
                        if sig:
                            base_cmd_parts.append(f"    +--pi-sigma {sig}")
                        f.write("\n# With wizard-chosen flags\n")
                        base_cmd = "\n".join(base_cmd_parts)
                        f.write(base_cmd + "\n")
                os.chmod(train_sh, 0o755)
                print(f"[wizard] Wrote {train_sh}")
                

def maybe_ctmc_pretrain() -> Path | None:
    """Interactive CTMC pretrain step returning a checkpoint path if created."""
    if not yesno("Pretrain on synthetic CTMC and warm-start?", default=False):
        return None
    out = Path(prompt("CTMC checkpoint path", default="./artifacts/pretrained/model.pt")).resolve()
    epochs = prompt("CTMC epochs", default="3").strip() or "3"
    steps = prompt("CTMC steps/epoch", default="400").strip() or "400"
    batch = prompt("CTMC batch", default="64").strip() or "64"
    Tseq = prompt("CTMC events/sequence T", default="128").strip() or "128"
    hidden = prompt("Hidden size", default="128").strip() or "128"
    layers = prompt("LSTM layers", default="1").strip() or "1"
    dev_default = "cuda" if _torch_cuda_available() else "cpu"
    device = prompt("Device for CTMC pretrain [cpu/cuda]", default=dev_default)
    cmd = [
        sys.executable,
        "-m",
        "torpedocode.cli.pretrain_ctmc",
        "--epochs",
        epochs,
        "--steps",
        steps,
        "--batch",
        batch,
        "--T",
        Tseq,
        "--hidden",
        hidden,
        "--layers",
        layers,
        "--device",
        device,
        "--output",
        str(out),
    ]
    code = run(cmd)
    if code != 0:
        print("[wizard] CTMC pretrain failed; continuing without warm-start.")
        return None
    print(f"[wizard] CTMC checkpoint saved → {out}")
    return out


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _topology_selected_exists(instrument: str, artifact_root: Path | None) -> bool:
    cands = []
    if artifact_root is not None:
        cands.append(artifact_root / instrument / "topology_selected.json")
    cands.append(Path("artifacts") / "topo" / instrument / "topology_selected.json")
    for p in cands:
        if p.exists():
            return True
    return False


def fast_eval_predictions(artifact_root: Path, instrument: str) -> None:
    """Fast in-process eval with optional paired DeLong and tqdm progress.

    Avoids subprocess overhead and provides progress feedback for large files.
    """
    import glob
    import json as _json
    import pandas as _pd
    from torpedocode.evaluation.metrics import (
        compute_classification_metrics as _cm,
        delong_ci_auroc as _ci,
        delong_test_auroc as _delong,
    )

    pattern = str(artifact_root / instrument / "*" / "predictions_test.csv")
    files = glob.glob(pattern)
    if not files:
        print("[fast-eval] no predictions_test.csv found")
        return
    bar = tqdm(files, desc="eval", leave=False) if tqdm is not None else files
    import shutil as _sh

    fast_eval_bin = os.environ.get("FAST_EVAL_BIN", "")
    fast_eval_pr = os.environ.get("FAST_EVAL_PR_MODE", "").strip().lower()
    for f in bar:
        try:
            df = _pd.read_csv(f)
            p = df["pred"].to_numpy()
            y = df["label"].to_numpy().astype(int)
            m = _cm(p, y)
            auc, lo, hi = _ci(p, y)
            out_obj = {
                "auroc": auc,
                "auroc_ci": [lo, hi],
                "auprc": m.auprc,
                "brier": m.brier,
                "ece": m.ece,
            }
            for cand in ("predictions_test_b.csv", "predictions_test_baseline.csv"):
                sib = Path(f).with_name(cand)
                if sib.exists():
                    try:
                        b = _pd.read_csv(sib)
                        if "pred2" not in b.columns:
                            if "pred_b" in b.columns:
                                b = b.rename(columns={"pred_b": "pred2"})
                            elif "pred" in b.columns:
                                b = b.rename(columns={"pred": "pred2"})
                        combo = df.merge(
                            b[["idx", "pred2", "label"]], on=["idx", "label"], how="inner"
                        )
                        delta, z, pval = _delong(
                            combo["pred"].to_numpy(),
                            combo["pred2"].to_numpy(),
                            combo["label"].to_numpy().astype(int),
                        )
                        out_obj.update(
                            {
                                "delta_auroc": float(delta),
                                "delong_z": float(z),
                                "delong_p": float(pval),
                            }
                        )
                        break
                    except Exception as e:
                        print(f"[fast-eval] paired DeLong failed on {sib.name}: {e}")
            out = Path(f).with_name("eval_fast.json")
            if fast_eval_bin and _sh.which(fast_eval_bin) and not any(
                Path(f).with_name(c).exists()
                for c in ("predictions_test_b.csv", "predictions_test_baseline.csv")
            ):
                cmd = [fast_eval_bin]
                if fast_eval_pr in {"trap", "sklearn", "step"}:
                    mode = "sklearn" if fast_eval_pr in {"sklearn", "step"} else "trap"
                    cmd += ["--pr-mode", mode]
                cmd += [f, str(out)]
                code = run(cmd)
                if code != 0:
                    out.write_text(_json.dumps(out_obj, indent=2))
            else:
                out.write_text(_json.dumps(out_obj, indent=2))
            # Always print where results were written so users can find them
            print(f"[fast-eval] wrote {out}")
        except Exception as e:
            print(f"[fast-eval] error evaluating {f}: {e}")


def pack_artifacts(artifact_root: Path, output: Path | None = None) -> None:
    """Bundle key artifacts into a zip using scripts/paper_pack.py.

    If output is None, writes artifacts/paper_bundle.zip under artifact_root's parent.
    """
    try:
        out = (
            output
            if output is not None
            else (artifact_root / "paper_bundle.zip")
        )
        cmd = [
            sys.executable,
            "scripts/paper_pack.py",
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(out),
        ]
        code = run(cmd)
        if code == 0:
            print(f"[wizard] Paper bundle created → {out}")
        else:
            print("[wizard] Paper pack failed; see logs above.")
    except Exception as e:
        print(f"[wizard] Error creating paper bundle: {e}")


def quick_run_hints():
    print("\n[Quick Run Hints — copy/paste]")
    print("- Build C++ tools (lobster + fast eval):")
    print("  uv run python scripts/build_native.py lobster")
    print("  uv run python scripts/build_native.py fast-eval")
    print("- Wizard fast eval (single-model):")
    print("  export FAST_EVAL_BIN=$PWD/cpp/fast_eval")
    print("  uv run python scripts/run_wizard.py")
    print("- Strict TDA for paper runs:")
    print("  export PAPER_TORPEDO_STRICT_TDA=1")
    print("- Side-aware + level mapping:")
    print("  Ingest with DataConfig(side_aware_events=True);")
    print("  Train with --expand-types-by-level to enable LO/CX@level expansion.")
    print("- DeepLOB2018 baseline:")
    print(
        "  uv run python -m torpedocode.cli.baselines --cache-root cache --instrument AAPL --baseline deeplob2018"
    )


def option_quick_demo(cache_root: Path) -> None:
    """Quick synthetic demo: generate a tiny NDJSON, cache, train, eval, and pack.

    No external data required. Uses simple synthetic LOB snapshots with L=5 levels.
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        print(f"[demo] pandas/numpy required: {e}")
        return
    symbol = prompt("Demo instrument symbol", default="DEMO")
    L = int(prompt("Levels L", default="5") or "5")
    T = int(prompt("Events T", default="600") or "600")
    tick = float(prompt("Tick size", default="0.01") or "0.01")
    eta = float(prompt("Instability threshold eta (abs mid change)", default="0.02") or "0.02")
    # Auto-tune sequence/window sizes to avoid empty loaders on tiny demos
    auto_tune = yesno("Auto-tune bptt/batch for this demo?", default=True)
    bptt_opt = 64
    batch_opt = 128
    if auto_tune:
        T_train = max(1, int(0.6 * T))
        # Aim for a handful of windows while keeping sequence length reasonable
        bptt_opt = int(max(8, min(64, max(1, T_train // 4))))
        n_windows = max(0, T_train - bptt_opt + 1)
        if n_windows <= 0:
            bptt_opt = max(1, min(16, T_train))
            n_windows = max(0, T_train - bptt_opt + 1)
        batch_opt = int(max(8, min(128, n_windows if n_windows > 0 else 8)))
    print("[demo] generating synthetic NDJSON…")
    rng = np.random.default_rng(7)
    base = 100.0
    # Simple random walk for mid, spread ~ 2 ticks
    steps = rng.normal(0, tick, size=T).cumsum()
    mid = base + steps
    spr = np.full(T, 2 * tick)
    bid1 = mid - spr / 2
    ask1 = mid + spr / 2
    ts = pd.date_range("2024-06-03 14:00:00Z", periods=T, freq="s")
    rows = []
    etypes = ["MO+", "MO-", "LO+", "LO-", "CX+", "CX-"]
    for i in range(T):
        rec = {
            "timestamp": ts[i].isoformat(),
            "event_type": str(rng.choice(etypes)),
            "size": float(np.exp(rng.normal(0.0, 0.4))),
        }
        # random level and side for LO/CX
        lvl = int(rng.integers(1, L + 1))
        rec["level"] = lvl
        rec["side"] = ("bid" if rec["event_type"].endswith("+") else "ask")
        # prices and sizes per level
        for l in range(1, L + 1):
            rec[f"bid_price_{l}"] = float(bid1[i] - (l - 1) * tick)
            rec[f"ask_price_{l}"] = float(ask1[i] + (l - 1) * tick)
            rec[f"bid_size_{l}"] = float(rng.gamma(2.0, 50.0))
            rec[f"ask_size_{l}"] = float(rng.gamma(2.0, 50.0))
        rows.append(rec)
    tmp = cache_root / "_demo"
    tmp.mkdir(parents=True, exist_ok=True)
    ndjson = tmp / f"{symbol}.ndjson"
    with ndjson.open("w") as f:
        for r in rows:
            import json as _json

            f.write(_json.dumps(r) + "\n")
    print(f"[demo] wrote {ndjson}")
    # Cache → train → eval → pack
    print("[demo] caching to parquet…")
    run([
        sys.executable,
        "-m",
        "torpedocode.cli.cache",
        "--input",
        str(ndjson),
        "--cache-root",
        str(cache_root),
        "--instrument",
        symbol,
        "--drop-auctions",
        "--session-tz",
        "America/New_York",
        "--levels",
        str(L),
        "--horizons-s",
        "1",
        "5",
        "10",
        "--horizons-events",
        "100",
        "500",
        "--eta",
        str(eta),
    ])
    # Optional: quick topology search and reuse selection
    use_topo = False
    topo_art_dir = Path("./artifacts/topo") / symbol
    if yesno("Run quick topology search and reuse selected config?", default=True):
        strict = yesno("Strict TDA for topology search?", default=False)
        run(
            [
                sys.executable,
                "-m",
                "torpedocode.cli.topo_search",
                "--cache-root",
                str(cache_root),
                "--instrument",
                symbol,
                "--label-key",
                "instability_s_5",
                "--artifact-dir",
                str(topo_art_dir),
                *( ["--strict-tda"] if strict else [] ),
            ]
        )
        if (topo_art_dir / "topology_selected.json").exists():
            use_topo = True

    print("[demo] training (CPU)…")
    art = Path("./artifacts").resolve()
    # Optional strict TDA and progress during training
    strict_train = yesno("Strict TDA during training?", default=False)
    show_tda_prog = yesno("Show topology feature progress during training?", default=True)
    if show_tda_prog:
        os.environ["WIZARD_TOPO_PROGRESS"] = "1"
    run(
        [
            sys.executable,
            "-m",
            "torpedocode.cli.train",
            "--instrument",
            symbol,
            "--cache-root",
            str(cache_root),
            "--label-key",
            "instability_s_5",
            "--artifact-dir",
            str(art / symbol / "instability_s_5"),
            "--epochs",
            "2",
            "--batch",
            str(batch_opt if auto_tune else 128),
            "--bptt",
            str(bptt_opt if auto_tune else 64),
            "--topo-stride",
            "5",
            "--device",
            "cpu",
            "--tpp-diagnostics",
            *( ["--use-topo-selected"] if use_topo else [] ),
            *( ["--strict-tda"] if strict_train else [] ),
        ]
    )
    eval_ran = False
    if yesno("Run fast eval now?", default=True):
        fast_eval_predictions(art, symbol)
        eval_ran = True
    if yesno("Pack artifacts into paper_bundle.zip?", default=True):
        pack_artifacts(art)
    # Summary banner of key artifact paths
    pfx = art / symbol / "instability_s_5"
    print("\n[Summary — Demo Artifacts]")
    print(f"- Predictions (test): {pfx / 'predictions_test.csv'}")
    print(f"- Predictions (val):  {pfx / 'predictions_val.csv'}")
    if eval_ran:
        print(f"- Fast eval:          {pfx / 'eval_fast.json'}")
    print(f"- Schemas:            {pfx / 'feature_schema.json'}, {pfx / 'scaler_schema.json'}")
    print(f"- Splits:             {pfx / 'split_indices.json'}")
    print(f"- TPP arrays/diag:    {pfx / 'tpp_test_arrays.npz'}, {pfx / 'tpp_test_diagnostics.json'}")
    print(f"- Training meta:      {pfx / 'training_meta.json'}")
    print(f"- TDA backends:       {pfx / 'tda_backends.json'}")


# Note: Single main() defined at the end of the file.


def option_crypto(cache_root: Path):
    print("Option A — Free crypto (Binance/Coinbase)")
    source = prompt("Choose source [binance/coinbase]", default="binance").lower()
    symbol = prompt("Symbol (e.g., BTCUSDT or BTC-USD)")
    eta = float(
        prompt("Instability threshold eta (abs mid change)", default="0.02") or "0.02"
    )
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
            bar = tqdm(pairs_list, desc="check", leave=False) if tqdm is not None else pairs_list
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
            now = datetime.now(timezone.utc)
            y = now.year
            m = now.month - 1
            pairs = []
            for _ in range(n):
                if m < 1:
                    y -= 1
                    m += 12
                pairs.append((y, m))
                m -= 1

            out = raw_dir / f"binance_{symbol}.ndjson"
            try:
                out.unlink(missing_ok=True)
            except Exception:
                pass
            ok_any = False
            from collections import defaultdict

            avail_pairs = _head_available(symbol, pairs)
            if not avail_pairs:
                print("[warn] no available months found; falling back to manual prompts.")
            by_year = defaultdict(list)
            for yy, mm in avail_pairs or pairs:
                by_year[yy].append(mm)
            for yy, months in by_year.items():
                months_sorted = sorted(months)
                code = run(
                    [
                        sys.executable,
                        "scripts/download_binance_monthly.py",
                        "--symbol",
                        symbol,
                        "--year",
                        str(yy),
                        "--months",
                        *[str(mo) for mo in months_sorted],
                        "--output",
                        str(out),
                    ]
                )
                if code == 0:
                    ok_any = True
            if ok_any:
                print("Downloaded Binance monthly data.")
                goto_ingest = True
            else:
                print(
                    "[warn] downloader failed; months may be unavailable yet. Falling back to manual prompts."
                )
        if yesno("Download Binance Vision monthly aggTrades?", default=False):
            now = datetime.now(timezone.utc)
            y = now.year
            m = now.month - 1
            if m < 1:
                y -= 1
                m += 12
            default_months = " ".join(str(x) for x in [max(1, m - 2), max(1, m - 1), m])
            year = int(prompt("Year", default=str(y)))
            months = prompt("Months (space-separated)", default=default_months).split()
            out = raw_dir / f"binance_{symbol}.ndjson"
            try:
                months_int = [int(m) for m in months]
            except Exception:
                months_int = [int(x) for x in months if str(x).isdigit()]
            pairs = [(int(year), m) for m in months_int]
            avail_pairs = pairs
            if pairs:
                avail_pairs = _head_available(symbol, pairs)
                if not avail_pairs:
                    print(
                        "[warn] none of the requested months appear available; attempting anyway."
                    )
            months_to_dl = [str(m) for (_, m) in (avail_pairs or pairs)]
            code = run(
                [
                    sys.executable,
                    "scripts/download_binance_monthly.py",
                    "--symbol",
                    symbol,
                    "--year",
                    str(year),
                    "--months",
                    *months_to_dl,
                    "--output",
                    str(out),
                ]
            )
            if code != 0:
                print("[warn] downloader failed; falling back to manual JSONL path")
        else:
            print("Place Binance JSONL (websocket capture) under", raw_dir)
            print(
                "Then run: uv run python scripts/binance_to_ndjson.py --input raw.jsonl --output cache/binance_SYMBOL.ndjson --symbol SYMBOL"
            )
    else:
        if yesno("Prep last N months automatically?", default=True):
            try:
                n = int(prompt("N months", default="3"))
            except Exception:
                n = 3
            now = datetime.now(timezone.utc)
            m = now.month - (n - 1)
            y = now.year
            if m < 1:
                y -= 1
                m += 12
            start = f"{y:04d}-{m:02d}-01T00:00:00Z"
            end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            out = raw_dir / f"coinbase_{symbol}.ndjson"
            code = run(
                [
                    sys.executable,
                    "scripts/download_coinbase_trades.py",
                    "--product",
                    symbol,
                    "--start",
                    start,
                    "--end",
                    end,
                    "--output",
                    str(out),
                ]
            )
            if code != 0:
                print("[warn] coinbase downloader failed; place JSONL under", raw_dir)
        elif yesno("Download Coinbase public trades via REST? (rate-limited)", default=False):
            start = prompt("Start ISO (UTC)", default="2024-06-01T00:00:00Z")
            end = prompt("End ISO (UTC)", default="2024-08-31T23:59:59Z")
            out = raw_dir / f"coinbase_{symbol}.ndjson"
            code = run(
                [
                    sys.executable,
                    "scripts/download_coinbase_trades.py",
                    "--product",
                    symbol,
                    "--start",
                    start,
                    "--end",
                    end,
                    "--output",
                    str(out),
                ]
            )
            if code != 0:
                print("[warn] coinbase downloader failed; place JSONL under", raw_dir)
        else:
            print("Place Coinbase JSONL under", raw_dir)
            print(
                "Then run: uv run python scripts/coinbase_to_ndjson.py --input raw.jsonl --output cache/coinbase_SYMBOL.ndjson --symbol SYMBOL"
            )

    cache_root.mkdir(parents=True, exist_ok=True)
    code = run(
        [
            sys.executable,
            "-m",
            "torpedocode.cli.ingest",
            "--raw-dir",
            str(raw_dir),
            "--cache-root",
            str(cache_root),
            "--instrument",
            symbol,
            "--eta",
            str(eta),
        ]
    )
    if code != 0:
        sys.exit(code)
    if yesno("Run topology grid search on validation?", default=False):
        topo_art = Path(
            prompt("Topology artifact dir", default=str(Path("./artifacts/topo") / symbol))
        ).resolve()
        label_key = prompt("Label key", default="instability_s_5")
        strict = yesno("Strict TDA for topology search?", default=False)
        run(
            [
                sys.executable,
                "-m",
                "torpedocode.cli.topo_search",
                "--cache-root",
                str(cache_root),
                "--instrument",
                symbol,
                "--label-key",
                label_key,
                "--artifact-dir",
                str(topo_art),
                *( ["--strict-tda"] if strict else [] ),
            ]
        )
    if yesno("Run quick multi-horizon report?", default=True):
        run(
            [
                sys.executable,
                "-m",
                "torpedocode.cli.report_multi",
                "--cache-root",
                str(cache_root),
                "--instrument",
                symbol,
                "--output",
                str(cache_root / f"{symbol}_multi.json"),
            ]
        )
    import glob

    caches = glob.glob(str(cache_root / "*.parquet"))
    if caches and yesno("Batch train across all instruments in cache-root?", default=False):
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        # Auto-tune bptt/batch from cache size when possible
        auto_tune = yesno("Auto-tune bptt/batch?", default=True)
        bptt_opt = 64
        batch_opt = 128
        strict_train = yesno("Strict TDA during training?", default=False)
        show_tda_prog = yesno("Show topology feature progress during training?", default=True)
        if show_tda_prog:
            os.environ["WIZARD_TOPO_PROGRESS"] = "1"
        if auto_tune:
            try:
                import pandas as _pd
                # Estimate using first cache file
                n0 = len(_pd.read_parquet(caches[0]))
                T_train = max(1, int(0.6 * n0))
                bptt_opt = int(max(8, min(64, max(1, T_train // 4))))
                n_windows = max(0, T_train - bptt_opt + 1)
                if n_windows <= 0:
                    bptt_opt = max(1, min(16, T_train))
                    n_windows = max(0, T_train - bptt_opt + 1)
                batch_opt = int(max(8, min(128, n_windows if n_windows > 0 else 8)))
            except Exception:
                pass
        labels = prompt(
            "Label keys (space-separated)", default="instability_s_1 instability_s_5"
        ).split()
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        instruments = [Path(p).stem for p in caches]
        cmd = [
            sys.executable,
            "-m",
            "torpedocode.cli.batch_train",
            "--cache-root",
            str(cache_root),
            "--artifact-root",
            str(art),
            "--instruments",
            *instruments,
            "--label-keys",
            *labels,
            "--epochs",
            "2",
            "--device",
            device,
            "--use-topo-selected",
        ]
        if auto_tune:
            cmd += ["--batch", str(batch_opt), "--bptt", str(bptt_opt)]
        if strict_train:
            cmd.append("--strict-tda")
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            for inst in instruments:
                fast_eval_predictions(art, inst)
        for inst in instruments:
            _summary_banner(art, inst)
    if yesno("Train multi-horizon hybrid?", default=True):
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        warm = maybe_ctmc_pretrain()
        # Optional: quick topology search
        use_topo = False
        topo_art = Path("./artifacts/topo") / symbol
        if yesno("Run quick topology search and reuse selected config?", default=True):
            strict = yesno("Strict TDA for topology search?", default=False)
            run(
                [
                    sys.executable,
                    "-m",
                    "torpedocode.cli.topo_search",
                    "--cache-root",
                    str(cache_root),
                    "--instrument",
                    symbol,
                    "--label-key",
                    "instability_s_5",
                    "--artifact-dir",
                    str(topo_art),
                    *( ["--strict-tda"] if strict else [] ),
                ]
            )
            if (topo_art / "topology_selected.json").exists():
                use_topo = True
        # Auto-tune bptt/batch from cache size
        auto_tune = yesno("Auto-tune bptt/batch?", default=True)
        bptt_opt = 64
        batch_opt = 128
        if auto_tune:
            try:
                import pandas as _pd
                n0 = len(_pd.read_parquet(cache_root / f"{symbol}.parquet"))
                T_train = max(1, int(0.6 * n0))
                bptt_opt = int(max(8, min(64, max(1, T_train // 4))))
                n_windows = max(0, T_train - bptt_opt + 1)
                if n_windows <= 0:
                    bptt_opt = max(1, min(16, T_train))
                    n_windows = max(0, T_train - bptt_opt + 1)
                batch_opt = int(max(8, min(128, n_windows if n_windows > 0 else 8)))
            except Exception:
                pass
        # Strict TDA and progress during training
        strict_train = yesno("Strict TDA during training?", default=False)
        show_tda_prog = yesno("Show topology feature progress during training?", default=True)
        if show_tda_prog:
            os.environ["WIZARD_TOPO_PROGRESS"] = "1"
            os.environ["WIZARD_TRAIN_PROGRESS"] = "1"
        cmd = [
            sys.executable,
            "-m",
            "torpedocode.cli.train_multi",
            "--cache-root",
            str(cache_root),
            "--artifact-root",
            str(art),
            "--epochs",
            "3",
            "--device",
            device,
        ]
        if _topology_selected_exists(symbol, art):
            cmd.append("--use-topo-selected")
        if use_topo:
            cmd.append("--use-topo-selected")
        if auto_tune:
            cmd += ["--batch", str(batch_opt), "--bptt", str(bptt_opt)]
        if warm is not None:
            cmd += ["--warm-start", str(warm)]
        if strict_train:
            cmd.append("--strict-tda")
        if show_tda_prog:
            cmd.append("--progress")
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            fast_eval_predictions(art, symbol)
        # Ensure predictions exist; if not, offer to rerun with more epochs
        if not _has_predictions(art, symbol):
            print(f"[note] No predictions found under {art / symbol}.")
            if yesno("Rerun training with more epochs?", default=True):
                more = int(prompt("Epochs", default="5") or "5")
                cmd = [
                    sys.executable,
                    "-m",
                    "torpedocode.cli.train_multi",
                    "--cache-root",
                    str(cache_root),
                    "--artifact-root",
                    str(art),
                    "--epochs",
                    str(more),
                    "--device",
                    device,
                    "--use-topo-selected",
                ]
                run(cmd)
        if _has_predictions(art, symbol) and yesno("Pack artifacts into paper_bundle.zip?", default=True):
            pack_artifacts(art)
        _summary_banner(art, symbol)


def option_lobster(cache_root: Path):
    print("Option B — Equities via LOBSTER CSVs")
    multi = yesno("Ingest multiple month directories under a root?", default=False)
    raw_dirs: list[Path] = []
    if multi:
        root = Path(prompt("Root with monthly subdirs (YYYY-MM/*)", default=".")).resolve()
        months = prompt("Months (space-separated, e.g., 2024-06 2024-07)", default="").split()
        for m in months:
            cand = root / m
            if cand.exists() and cand.is_dir():
                raw_dirs.append(cand)
        if not raw_dirs:
            print("[warn] no month dirs matched; falling back to single directory prompt")
            day_dir = Path(
                prompt("Path to LOBSTER day directory (contains message_*.csv and orderbook_*.csv)")
            ).resolve()
            raw_dirs = [day_dir]
    else:
        day_dir = Path(
            prompt("Path to LOBSTER day directory (contains message_*.csv and orderbook_*.csv)")
        ).resolve()
        raw_dirs = [day_dir]
    symbol = prompt("Instrument symbol (e.g., AAPL)")
    tick = prompt("Tick size (e.g., 0.01)", default="0.01")
    eta = float(
        prompt("Instability threshold eta (abs mid change)", default="0.02") or "0.02"
    )
    ca_csv = prompt("Corporate actions CSV (or blank)", default="").strip()
    ingest_cmd = [
        sys.executable,
        "-m",
        "torpedocode.cli.ingest",
        "--raw-dir",
        *[str(p) for p in raw_dirs],
        "--cache-root",
        str(cache_root),
        "--instrument",
        symbol,
        "--tick-size",
        tick,
        "--eta",
        str(eta),
    ]
    if ca_csv:
        ingest_cmd += ["--actions-csv", ca_csv, "--apply-actions"]
    code = run(ingest_cmd)
    if code != 0:
        sys.exit(code)
    if yesno("Run topology grid search on validation?", default=False):
        topo_art = Path(
            prompt("Topology artifact dir", default=str(Path("./artifacts/topo") / symbol))
        ).resolve()
        label_key = prompt("Label key", default="instability_s_5")
        strict = yesno("Strict TDA for topology search?", default=False)
        run(
            [
                sys.executable,
                "-m",
                "torpedocode.cli.topo_search",
                "--cache-root",
                str(cache_root),
                "--instrument",
                symbol,
                "--label-key",
                label_key,
                "--artifact-dir",
                str(topo_art),
                *( ["--strict-tda"] if strict else [] ),
            ]
        )
    if yesno("Train multi-horizon hybrid?", default=True):
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        warm = maybe_ctmc_pretrain()
        # Optional: quick topology search
        use_topo = False
        topo_art = Path("./artifacts/topo") / symbol
        if yesno("Run quick topology search and reuse selected config?", default=True):
            strict = yesno("Strict TDA for topology search?", default=False)
            run(
                [
                    sys.executable,
                    "-m",
                    "torpedocode.cli.topo_search",
                    "--cache-root",
                    str(cache_root),
                    "--instrument",
                    symbol,
                    "--label-key",
                    "instability_s_5",
                    "--artifact-dir",
                    str(topo_art),
                    *( ["--strict-tda"] if strict else [] ),
                ]
            )
            if (topo_art / "topology_selected.json").exists():
                use_topo = True
        # Auto-tune bptt/batch from cache size
        auto_tune = yesno("Auto-tune bptt/batch?", default=True)
        bptt_opt = 64
        batch_opt = 128
        if auto_tune:
            try:
                import pandas as _pd
                n0 = len(_pd.read_parquet(cache_root / f"{symbol}.parquet"))
                T_train = max(1, int(0.6 * n0))
                bptt_opt = int(max(8, min(64, max(1, T_train // 4))))
                n_windows = max(0, T_train - bptt_opt + 1)
                if n_windows <= 0:
                    bptt_opt = max(1, min(16, T_train))
                    n_windows = max(0, T_train - bptt_opt + 1)
                batch_opt = int(max(8, min(128, n_windows if n_windows > 0 else 8)))
            except Exception:
                pass
        cmd = [
            sys.executable,
            "-m",
            "torpedocode.cli.train_multi",
            "--cache-root",
            str(cache_root),
            "--artifact-root",
            str(art),
            "--epochs",
            "3",
            "--device",
            device,
        ]
        if _topology_selected_exists(symbol, art):
            cmd.append("--use-topo-selected")
        if use_topo:
            cmd.append("--use-topo-selected")
        if auto_tune:
            cmd += ["--batch", str(batch_opt), "--bptt", str(bptt_opt)]
        if warm is not None:
            cmd += ["--warm-start", str(warm)]
        if strict_train:
            cmd.append("--strict-tda")
        run(cmd)
        if yesno("Run fast eval on saved predictions?", default=True):
            fast_eval_predictions(art, symbol)
        if yesno("Pack artifacts into paper_bundle.zip?", default=True):
            pack_artifacts(art)
        _summary_banner(art, symbol)


def option_itch_ouch(cache_root: Path):
    print("Option C — Equities via ITCH/OUCH")
    if not yesno("Do you already have ITCH/OUCH files?", default=False):
        print("Download of ITCH/OUCH is not automated (data access requires licensing).")
        print("If you obtain files, re-run this wizard and choose this option again.")
        return
    raw_dir = Path(prompt("Directory containing .itch/.ouch files")).resolve()
    symbol = prompt("Instrument symbol (e.g., AAPL)")
    spec = prompt("Vendor spec (e.g., nasdaq-itch-5.0)", default="nasdaq-itch-5.0")
    eta = float(
        prompt("Instability threshold eta (abs mid change)", default="0.02") or "0.02"
    )
    code = run(
        [
            sys.executable,
            "-m",
            "torpedocode.cli.ingest",
            "--raw-dir",
            str(raw_dir),
            "--cache-root",
            str(cache_root),
            "--instrument",
            symbol,
            "--itch-spec",
            spec,
            "--eta",
            str(eta),
        ]
    )
    if code != 0:
        sys.exit(code)
    if yesno("Run topology grid search on validation?", default=False):
        topo_art = Path(
            prompt("Topology artifact dir", default=str(Path("./artifacts/topo") / symbol))
        ).resolve()
        label_key = prompt("Label key", default="instability_s_5")
        run(
            [
                sys.executable,
                "-m",
                "torpedocode.cli.topo_search",
                "--cache-root",
                str(cache_root),
                "--instrument",
                symbol,
                "--label-key",
                label_key,
                "--artifact-dir",
                str(topo_art),
            ]
        )
    if yesno("Train multi-horizon hybrid?", default=True):
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device [cpu/cuda]", default=dev_default)
        warm = maybe_ctmc_pretrain()
        cmd = [
            sys.executable,
            "-m",
            "torpedocode.cli.train_multi",
            "--cache-root",
            str(cache_root),
            "--artifact-root",
            str(art),
            "--epochs",
            "3",
            "--device",
            device,
        ]
        if _topology_selected_exists(symbol, art):
            cmd.append("--use-topo-selected")
        if warm is not None:
            cmd += ["--warm-start", str(warm)]
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
    print("  4) Optional: CTMC pretrain (warm-start)")
    print("  5) Train multi-horizon hybrid (CPU/GPU)")
    print("  6) Fast eval + DeLong where available")
    print("  7) Optional: Aggregate across instruments")
    print("Select data option:")
    print("  0) Quick synthetic demo (no external data)")
    print("  1) Free crypto (Binance/Coinbase)")
    print("  2) LOBSTER CSVs (equities)")
    print("  3) ITCH/OUCH (equities; requires files)")
    choice = prompt("Enter 0/1/2/3", default="0")
    if choice == "0":
        option_quick_demo(cache_root)
    elif choice == "1":
        option_crypto(cache_root)
    elif choice == "2":
        option_lobster(cache_root)
    else:
        option_itch_ouch(cache_root)
    if yesno("Aggregate across instruments for one label?", default=False):
        ins = sorted([p.stem for p in Path(cache_root).glob("*.parquet")])
        if len(ins) < 2:
            print("[note] Need at least 2 instruments under cache-root for aggregation.")
            return
        label = prompt("Label key to aggregate", default="instability_s_5")
        art = Path(prompt("Artifact root", default="./artifacts")).resolve()
        dev_default = "cuda" if _torch_cuda_available() else "cpu"
        device = prompt("Device for batch training [cpu/cuda]", default=dev_default)
        cmd = [
            sys.executable,
            "-m",
            "torpedocode.cli.batch_train",
            "--cache-root",
            str(cache_root),
            "--artifact-root",
            str(art),
            "--instruments",
            *ins,
            "--label-keys",
            label,
            "--epochs",
            "2",
            "--device",
            device,
        ]
        cmd.append("--use-topo-selected")
        run(cmd)
        pattern = f"*/{label}/predictions_test.csv"
        out = art / f"aggregate_{label}.json"
        run(
            [
                sys.executable,
                "-m",
                "torpedocode.cli.aggregate",
                "--pred-root",
                str(art),
                "--pred-pattern",
                pattern,
                "--output",
                str(out),
            ]
        )
        if yesno("Pack aggregated artifacts into paper_bundle.zip?", default=True):
            pack_artifacts(art)

    # Optional: Cross-market LOMO protocol
    if yesno("Run cross-market LOMO protocol?", default=False):
        panel_path = Path(
            prompt(
                "Panel CSV/JSON with market,symbol columns",
                default=str(Path("./artifacts/panel_matched.csv")),
            )
        ).resolve()
        if not panel_path.exists():
            print(f"[warn] Panel not found at {panel_path}; skipping LOMO.")
        else:
            label_key = prompt("Label key", default="instability_s_5").strip() or "instability_s_5"
            dev_default = "cuda" if _torch_cuda_available() else "cpu"
            device = prompt("Device [cpu/cuda]", default=dev_default)
            with_tda = yesno("Include TDA features (with_tda)?", default=True)
            strict = yesno("Strict TDA (fail if backends missing)?", default=False)
            out_path = Path(
                prompt(
                    "Output JSON path",
                    default=str(Path("./artifacts/cross_market_lomo.json")),
                )
            ).resolve()
            cmd = [
                sys.executable,
                "-m",
                "torpedocode.cli.cross_market",
                "--panel",
                str(panel_path),
                "--cache-root",
                str(cache_root),
                "--label-key",
                label_key,
                "--mode",
                "lomo",
                "--epochs",
                "1",
                "--device",
                device,
                "--output",
                str(out_path),
            ]
            if with_tda:
                cmd.append("--with-tda")
            if strict:
                cmd.append("--strict-tda")
            run(cmd)


if __name__ == "__main__":
    main()
