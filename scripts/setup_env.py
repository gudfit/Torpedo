#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import json


APT_PACKAGES = [
    "build-essential",
    "python3-dev",
    "python3-venv",
    "pkg-config",
    "git",
    "curl",
    "vim",
    "cmake",
    # cargo via rustup preferred; leave here for minimal systems that package it
    "cargo",
]


def sh(cmd: list[str], check: bool = True) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def install_uv() -> None:
    if shutil.which("uv"):
        print("uv already installed")
        return
    # No bundled installer here; print guidance
    print("[warn] 'uv' CLI not found. Install from https://astral.sh/uv (recommended).\n"
          "      Falling back to std venv + pip if available.")


def _ensure_pip(python_bin: str) -> None:
    # Try to ensure pip exists inside the venv
    code = sh([python_bin, "-m", "pip", "--version"], check=False)
    if code != 0:
        _ = sh([python_bin, "-m", "ensurepip", "--upgrade"], check=False)
    # Always try to upgrade pip/setuptools/wheel
    _ = sh([python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=False)


def _ensure_rust() -> None:
    if shutil.which("cargo"):
        print("cargo already installed")
        return
    print("[info] cargo not found; installing Rust toolchain via rustup...")
    if not shutil.which("curl"):
        print("[warn] curl not found; please install curl to bootstrap rustup, or install Rust manually from https://rustup.rs/")
        return
    # Install rustup non-interactively to default location
    code = sh(["sh", "-c", "curl https://sh.rustup.rs -sSf | sh -s -- -y"], check=False)
    if code == 0:
        # Load cargo into PATH for current shell session if possible
        cargo_env = str(Path.home() / ".cargo" / "env")
        if Path(cargo_env).exists():
            # Note: this affects only subshells spawned by this script
            os.environ["PATH"] = f"{str(Path.home() / '.cargo' / 'bin')}:{os.environ.get('PATH','')}"
        print("[info] Rust installed via rustup.")
    else:
        print("[warn] Rust installation failed. Please install manually: https://rustup.rs/")


def _torch_cuda_available(python_bin: str) -> bool | None:
    try:
        out = subprocess.check_output(
            [python_bin, "-c", "import json,torch;print(json.dumps({'avail': torch.cuda.is_available(), 'compiled': getattr(torch.version, 'cuda', None)}))"],
            text=True,
        ).strip()
        data = json.loads(out)
        print(f"[detect] torch cuda available={data.get('avail')} (compiled cuda={data.get('compiled')})")
        return bool(data.get("avail"))
    except Exception as e:
        print(f"[detect] torch cuda detection failed: {e}")
        return None


def setup(
    env_dir: Path,
    *,
    dev: bool,
    cuda: bool | None,
    rust: bool = True,
    install_pyarrow: bool = False,
    cuda_arch: str | None = None,
) -> None:
    env_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = env_dir / ".venv"
    uv_bin = shutil.which("uv")

    # Create venv
    if uv_bin:
        sh([uv_bin, "venv", str(venv_dir)])
    else:
        sh([sys.executable, "-m", "venv", str(venv_dir)])

    py = str(venv_dir / "bin" / ("python.exe" if os.name == "nt" else "python"))
    _ensure_pip(py)

    # Install project
    pkgs = "-e .[dev]" if dev else "."
    if uv_bin:
        # Ensure pip toolchain upgraded inside venv as well
        sh([uv_bin, "pip", "install", "--python", py, "--upgrade", "pip", "setuptools", "wheel"])  # upgrade basics
        sh([uv_bin, "pip", "install", pkgs, "--python", py])
    else:
        sh([py, "-m", "pip", "install", pkgs])

    # Ensure Rust unless disabled
    if rust:
        _ensure_rust()
        # Ensure maturin is available for Rust pyo3 editable builds
        if uv_bin:
            sh([uv_bin, "pip", "install", "maturin", "--python", py])
        else:
            sh([py, "-m", "pip", "install", "maturin"])

    # Build native components
    # Deny warnings for Rust builds to keep artifacts clean
    os.environ.setdefault("RUSTFLAGS", "-D warnings")
    sh([py, "scripts/build_native.py", "rust", "--verbose"])
    sh([py, "scripts/build_native.py", "panel", "--verbose"])
    # Decide CUDA vs CPU for torch extension
    cuda_decision = cuda
    if cuda_decision is None:
        cuda_decision = _torch_cuda_available(py)
    if cuda_decision:
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(env_dir / ".tmp/torch_extensions"))
        if cuda_arch:
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", str(cuda_arch))
        print("[build] Building Torch CUDA extension (auto)")
        sh([py, "scripts/build_native.py", "torch-cuda", "--verbose"])
    else:
        print("[build] Building Torch CPU extension")
        sh([py, "scripts/build_native.py", "torch", "--verbose"])

    # Optional: install pyarrow for parquet support and smoke scripts
    if install_pyarrow:
        if uv_bin:
            sh([uv_bin, "pip", "install", "pyarrow", "--python", py])
        else:
            sh([py, "-m", "pip", "install", "pyarrow"])

    # Sanity checks: try importing native modules
    try:
        out = subprocess.check_output([py, "-c", "import torpedocode_ingest; print('torpedocode_ingest OK')"], text=True).strip()
        print(out)
    except Exception as e:
        print(f"[check] torpedocode_ingest import failed: {e}")
    try:
        out = subprocess.check_output([py, "-c", "import torpedocode_tda; print('torpedocode_tda OK')"], text=True).strip()
        print(out)
    except Exception as e:
        print(f"[check] torpedocode_tda import failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Set up TorpedoCode environment (system + Python)")
    ap.add_argument("--env-dir", type=Path, default=Path.cwd())
    ap.add_argument("--apt", action="store_true", help="Install system packages via apt (requires sudo)")
    ap.add_argument("--dev", action="store_true", help="Install dev extras (tests, tda backends)")
    ap.add_argument(
        "--cuda",
        action="store_true",
        help="Force building CUDA extension (skip auto-detection)",
    )
    ap.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU-only build (skip auto-detection)",
    )
    ap.add_argument(
        "--no-rust",
        action="store_true",
        help="Do not attempt to install Rust toolchain (cargo)",
    )
    ap.add_argument(
        "--rust",
        action="store_true",
        help="Force install Rust toolchain (cargo) if missing",
    )
    # Install pyarrow by default; allow opting out
    ap.add_argument(
        "--pyarrow",
        dest="pyarrow",
        action="store_true",
        default=True,
        help="Install pyarrow for parquet support (default: on)",
    )
    ap.add_argument(
        "--no-pyarrow",
        dest="pyarrow",
        action="store_false",
        help="Skip installing pyarrow",
    )
    ap.add_argument(
        "--cuda-arch",
        type=str,
        default=None,
        help="Optional TORCH_CUDA_ARCH_LIST value (e.g., '90' or '89;90')",
    )
    args = ap.parse_args()

    if args.apt:
        if shutil.which("apt-get") is None:
            print("[warn] apt-get not found; skipping system package installation")
        else:
            sh(["sudo", "apt-get", "update"])
            sh(["sudo", "apt-get", "install", "-y", *APT_PACKAGES])
    install_uv()

    # Decide CUDA policy: explicit flags override auto-detect.
    cuda_policy: bool | None
    if args.cuda:
        cuda_policy = True
    elif args.no_cuda:
        cuda_policy = False
    else:
        cuda_policy = None  # auto-detect after install

    setup(
        args.env_dir.resolve(),
        dev=bool(args.dev),
        cuda=cuda_policy,  # allow None to trigger auto-detection
        rust=(True if args.rust else (not bool(args.no_rust))),
        install_pyarrow=bool(args.pyarrow),
        cuda_arch=args.cuda_arch,
    )


if __name__ == "__main__":
    main()
