#!/usr/bin/env python3
"""CUDA/PyTorch environment diagnostics.

Usage:
  python scripts/check_cuda.py [-v] [--build-test]

Reports:
  - Torch and CUDA versions (compiled and runtime visibility)
  - torch.cuda.is_available() and device info
  - cuDNN, NCCL availability
  - CUDA toolkit (nvcc) and NVIDIA driver (nvidia-smi) if present
  - Optional: builds/loads a tiny CUDA kernel via torch.utils.cpp_extension
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from textwrap import indent


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = getattr(e, "output", "") or str(e)
        return 1, msg.strip()


def info_from_torch(verbose: bool) -> dict:
    info: dict = {"torch_importable": False}
    try:
        import torch  # type: ignore

        info["torch_importable"] = True
        info["torch_version"] = getattr(torch, "__version__", None)
        info["torch_compiled_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
        info["cudnn_available"] = bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.is_available())
        info["cudnn_version"] = (
            int(torch.backends.cudnn.version()) if info["cudnn_available"] else None
        )
        info["nccl_available"] = bool(getattr(torch.backends, "nccl", None) and torch.distributed.is_available())

        devices = []
        if info["cuda_device_count"]:
            for i in range(info["cuda_device_count"]):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_GB": round(props.total_memory / (1024**3), 2),
                        "multi_processor_count": props.multi_processor_count,
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
            # Try a tiny CUDA op
            try:
                x = torch.ones(4, device="cuda")
                y = x * 2
                info["cuda_tensor_op_ok"] = bool((y == 2).all().item())
            except Exception as e:
                if verbose:
                    print("[warn] simple CUDA op failed:", e, file=sys.stderr)
                info["cuda_tensor_op_ok"] = False
        info["devices"] = devices
    except Exception as e:
        if verbose:
            print("[warn] torch import failed:", e, file=sys.stderr)
    return info


def info_from_system(verbose: bool) -> dict:
    env = {
        k: os.environ.get(k)
        for k in [
            "CUDA_VISIBLE_DEVICES",
            "CUDA_HOME",
            "CUDA_PATH",
            "LD_LIBRARY_PATH",
            "PATH",
        ]
    }
    nvcc = shutil.which("nvcc")
    smi = shutil.which("nvidia-smi")
    code_nvcc, out_nvcc = _run([nvcc, "--version"]) if nvcc else (1, "nvcc not on PATH")
    code_smi, out_smi = _run([smi, "-L"]) if smi else (1, "nvidia-smi not on PATH")
    # Also grab driver/runtime summary if possible
    code_smi2, out_smi2 = _run([smi]) if smi else (1, "n/a")
    return {
        "env": env,
        "nvcc_path": nvcc,
        "nvcc_version": out_nvcc if code_nvcc == 0 else None,
        "nvidia_smi_path": smi,
        "nvidia_smi_list": out_smi if code_smi == 0 else None,
        "nvidia_smi": out_smi2 if code_smi2 == 0 else None,
    }


def try_build_inline(verbose: bool) -> dict:
    """Optionally tries to JIT-compile a trivial CUDA kernel via Torch extension.
    Returns dict with keys: supported, built, error
    """
    result = {"supported": False, "built": False, "error": None}
    try:
        import torch  # type: ignore
        from torch.utils.cpp_extension import is_ninja_available, load_inline

        result["supported"] = True
        if verbose:
            print("[info] ninja available:", is_ninja_available())

        cpp = r"""
        #include <torch/extension.h>
        // Forward declaration of the launcher implemented in CUDA file.
        void dbl(torch::Tensor x, torch::Tensor y);
        """

        cuda = r"""
        #include <ATen/cuda/CUDAContext.h>
        #include <torch/extension.h>

        __global__ void dbl_kernel(const float* x, float* y, int n){
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < n) y[i] = 2.0f * x[i];
        }

        void dbl(torch::Tensor x, torch::Tensor y){
            const int n = x.numel();
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            dbl_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<float>(), y.data_ptr<float>(), n);
        }
        """

        module = load_inline(
            name="cuda_inline_check",
            cpp_sources=cpp,
            cuda_sources=cuda,
            functions=["dbl"],
            verbose=verbose,
        )

        x = torch.arange(16, dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)
        module.dbl(x, y)
        result["built"] = bool(torch.allclose(y, x * 2))
    except Exception as e:
        result["error"] = str(e)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="CUDA/PyTorch diagnostics")
    ap.add_argument("--build-test", action="store_true", help="JIT-compile and run a tiny CUDA kernel")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print("== Torch/CUDA Info ==")
    tinf = info_from_torch(args.verbose)
    print(indent(json.dumps(tinf, indent=2), prefix="  "))

    print("\n== System Info ==")
    sinf = info_from_system(args.verbose)
    # Truncate PATH/LD_LIBRARY_PATH for readability
    if sinf.get("env"):
        env = dict(sinf["env"])  # shallow copy
        for k in ["PATH", "LD_LIBRARY_PATH"]:
            v = env.get(k)
            if v and len(v) > 300:
                env[k] = v[:300] + "... (truncated)"
        sinf["env"] = env
    print(indent(json.dumps(sinf, indent=2), prefix="  "))

    if args.build_test:
        print("\n== Build Test (CUDA inline) ==")
        res = try_build_inline(args.verbose)
        print(indent(json.dumps(res, indent=2), prefix="  "))

    # Common hints
    hints = []
    if not tinf.get("torch_importable"):
        hints.append("Torch not importable: install torch that matches your CUDA (or CPU-only).")
    elif tinf.get("cuda_available") is False:
        cc = tinf.get("torch_compiled_cuda")
        if cc in (None, "None"):
            hints.append("You installed a CPU-only torch build; install a CUDA wheel (e.g., cu124).")
        if not sinf.get("nvidia_smi_path"):
            hints.append("nvidia-smi not found: NVIDIA driver likely missing or container lacks /dev/nvidia*." )
        if not sinf.get("nvcc_path"):
            hints.append("nvcc not found: CUDA toolkit not on PATH (only required for building extensions).")
        hints.append("Ensure CUDA_VISIBLE_DEVICES allows at least one GPU and that the driver supports your torch CUDA version.")

    if hints:
        print("\n== Hints ==")
        for h in hints:
            print(" -", h)


if __name__ == "__main__":
    main()
