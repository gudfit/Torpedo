#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None, verbose: bool = True) -> int:
    if verbose:
        print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def build_rust(verbose: bool = True) -> None:
    """Build optional Rust Python extensions using maturin if available.

    Modules:
      - torpedocode_ingest (rust/torpedocode-py)
      - torpedocode_tda (rust/torpedocode-tda)
    """
    maturin = shutil.which("maturin")
    built_any = False
    for name, rel in (
        ("torpedocode_ingest", ROOT / "rust" / "torpedocode-py" / "Cargo.toml"),
        ("torpedocode_tda", ROOT / "rust" / "torpedocode-tda" / "Cargo.toml"),
    ):
        if not rel.exists():
            if verbose:
                print(f"[rust] Skipping {name}: Cargo.toml not found at {rel}")
            continue
        if maturin:
            # Handle common env clash where both VIRTUAL_ENV and CONDA_PREFIX are set
            base = ["maturin"]
            if os.environ.get("VIRTUAL_ENV") and os.environ.get("CONDA_PREFIX"):
                # Prefer the active virtualenv; remove conda hint for maturin
                base = ["env", "-u", "CONDA_PREFIX", "maturin"]
            code = run(base + ["develop", "-m", str(rel), "--release"], verbose=verbose)
            if code == 0:
                print(f"[rust] Built {name} via maturin")
                built_any = True
                continue
        else:
            if verbose:
                print("[rust] maturin not found; will try pip editable install once for all modules")
    if not built_any:
        # Fallback once: pip install -e . triggers setuptools-rust optional builds declared in pyproject
        print("[rust] Trying pip install -e . for optional Rust extensions")
        code = run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT, verbose=verbose)
        if code != 0:
            print("[rust] Editable install failed. Ensure Rust toolchain is installed.")


def build_panel(verbose: bool = True) -> None:
    panel_dir = ROOT / "rust" / "torpedocode-panel"
    cargo = panel_dir / "Cargo.toml"
    if not cargo.exists():
        print("[panel] Skipping: Cargo.toml not found")
        return
    code = run(["cargo", "build", "--release"], cwd=panel_dir, verbose=verbose)
    if code != 0:
        print("[panel] Build failed")
        return
    target = (
        panel_dir
        / "target"
        / "release"
        / ("torpedocode-panel" + (".exe" if os.name == "nt" else ""))
    )
    if not target.exists():
        print("[panel] Build completed but binary not found")
        return
    bindir = ROOT / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    dst = bindir / target.name
    shutil.copy2(target, dst)
    print(f"[panel] Installed {dst}")


def build_torch(cuda: bool = False, verbose: bool = True) -> None:
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load
    except Exception as e:
        print(f"[torch] PyTorch not available: {e}")
        return

    cpp = ROOT / "cpp" / "src" / "extension.cpp"
    tpp = ROOT / "cpp" / "src" / "tpp_loss.cpp"
    cu = ROOT / "cpp" / "src" / "lob_kernels.cu"

    name = "torpedocode_kernels"
    # Always include C++ sources that define symbols referenced by extension.cpp
    sources = [str(cpp)]
    if tpp.exists():
        sources.append(str(tpp))
    # Ensure C++17 for host compilation; CUDA will inherit as needed
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "-std=c++17"]
    if cuda:
        sources.append(str(cu))
        # Define macro to disable CUDA fallback stub in extension.cpp
        extra_cflags = list(extra_cflags) + ["-DTORPEDOCODE_ENABLE_CUDA"]
        extra_cuda_cflags = list(extra_cuda_cflags) + ["-DTORPEDOCODE_ENABLE_CUDA"]

    try:
        if verbose:
            print(f"[torch] Building extension (cuda={cuda}) ...")
        # Some Torch versions don't accept include_dirs kwarg in load(); pass -I via cflags
        inc = str(ROOT / "cpp" / "include")
        extra_cflags = list(extra_cflags) + [f"-I{inc}"]
        if cuda:
            extra_cuda_cflags = list(extra_cuda_cflags) + [f"-I{inc}"]
        _ = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags if cuda else None,
            verbose=verbose,
        )
        if verbose:
            print("[torch] Extension built and loaded.")
    except Exception as e:
        print(f"[torch] Build failed: {e}")


def build_lobster_fast(verbose: bool = True) -> None:
    bin_path = ROOT / "cpp" / "lobster_fast"
    src = ROOT / "cpp" / "src" / "lobster_fast.cpp"
    if not src.exists():
        print("[lobster] Source not found; skipping")
        return
    code = run(["g++", "-O3", "-std=c++17", "-o", str(bin_path), str(src)], verbose=verbose)
    if code == 0:
        print(f"[lobster] Built {bin_path}")
    else:
        print("[lobster] Build failed")


def install_torch(index: str | None, version: str | None) -> None:
    if index is None:
        print("[torch] No index specified. Provide --pytorch-index cpu|cu121|cu124 or a full URL.")
        return
    if index.startswith("http"):
        url = index
    else:
        mapping = {
            "cpu": "https://download.pytorch.org/whl/cpu",
            "cu121": "https://download.pytorch.org/whl/cu121",
            "cu124": "https://download.pytorch.org/whl/cu124",
        }
        if index not in mapping:
            print(f"[torch] Unknown index '{index}'. Use cpu|cu121|cu124 or provide a full URL.")
            return
        url = mapping[index]
    pkg = "torch" + (f"=={version}" if version else "")
    code = run([sys.executable, "-m", "pip", "install", pkg, "--index-url", url])
    if code == 0:
        print("[torch] Installed", pkg, "from", url)
    else:
        print("[torch] Installation failed. Check CUDA toolkit/index URL compatibility.")


def main():
    ap = argparse.ArgumentParser(description="Build native components (Rust, Torch, C++)")
    ap.add_argument(
        "component",
        choices=[
            "all",
            "rust",
            "torch",
            "torch-cuda",
            "lobster",
            "panel",
            "check",
            "torch-install",
            "fast-eval",
        ],
        help="What to build",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose build logs")
    ap.add_argument(
        "--pytorch-index",
        type=str,
        default=None,
        help="PyTorch index selector: cpu, cu121, cu124, or full index URL",
    )
    ap.add_argument(
        "--torch-version", type=str, default=None, help="Optional torch version spec (e.g., 2.4.0)"
    )
    args = ap.parse_args()

    if args.component == "check":
        print("[check] torch installed:", end=" ")
        try:
            import torch  # noqa

            print("yes, version", torch.__version__)
            try:
                print("[check] torch.cuda.is_available:", torch.cuda.is_available())
                print("[check] torch compiled cuda:", getattr(torch.version, "cuda", None))
            except Exception:
                pass
        except Exception as e:
            print(f"no ({e})")
        print("[check] nvcc available:", shutil.which("nvcc") or "no")
        print("[check] maturin available:", shutil.which("maturin") or "no")
        mod = None
        try:
            import torpedocode_ingest as mod  # noqa

            print("[check] torpedocode_ingest importable: yes")
        except Exception as e:
            print(f"[check] torpedocode_ingest importable: no ({e})")
        lb = ROOT / "cpp" / "lobster_fast"
        print("[check] lobster_fast exists:", "yes" if lb.exists() else "no")
        pb = ROOT / "bin" / ("torpedocode-panel" + (".exe" if os.name == "nt" else ""))
        print("[check] torpedocode-panel exists:", "yes" if pb.exists() else "no")
        return
    if args.component in ("rust", "all"):
        build_rust(verbose=args.verbose)
        if args.component != "all":
            return

    if args.component == "torch-install":
        install_torch(args.pytorch_index, args.torch_version)
        return

    if args.component in ("all", "rust"):
        build_rust(verbose=args.verbose)
    if args.component in ("all", "torch"):
        build_torch(cuda=False, verbose=args.verbose)
    if args.component in ("all", "torch-cuda"):
        build_torch(cuda=True, verbose=args.verbose)
    if args.component in ("all", "lobster"):
        build_lobster_fast(verbose=args.verbose)
    if args.component in ("all", "fast-eval"):
        # local build of C++ fast evaluator
        bin_path = ROOT / "cpp" / "fast_eval"
        src = ROOT / "cpp" / "src" / "fast_eval.cpp"
        if src.exists():
            flags = ["g++", "-O3", "-march=native", "-ffast-math", "-std=c++17"]
            if os.environ.get("FAST_EVAL_OPENMP", "0").lower() in {"1", "true"}:
                flags.append("-fopenmp")
            flags += ["-o", str(bin_path), str(src)]
            _ = run(flags, verbose=args.verbose)
            print(f"[fast-eval] Built {bin_path if bin_path.exists() else '(failed)'}")
        else:
            print("[fast-eval] Source not found; skipping")
    if args.component in ("all", "panel"):
        build_panel(verbose=args.verbose)


if __name__ == "__main__":
    main()
