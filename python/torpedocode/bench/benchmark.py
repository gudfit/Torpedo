"""Benchmark end-to-end throughput/latency for features+PH+model.

Example:
  python -m torpedocode.bench.benchmark --levels 10 --window-s 5 --T 5000 --batch 256 --stride 5
"""

from __future__ import annotations

import argparse
import time
from statistics import median
import resource
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..config import ModelConfig, TopologyConfig
from ..features.topological import TopologicalFeatureGenerator


def run_once(
    L: int, T: int, F: int, Z: int, batch: int, window_s: int, stride: int, device: str = "cpu"
):
    rng = np.random.default_rng(0)
    ts = np.arange(T, dtype=np.int64) * 10_000_000
    X = rng.normal(size=(T, F)).astype(np.float32)
    topo_cfg = TopologyConfig(
        window_sizes_s=[window_s],
        complex_type="cubical",
        max_homology_dimension=1,
        persistence_representation="landscape",
        landscape_levels=3,
        use_liquidity_surface=True,
    )
    topo = TopologicalFeatureGenerator(topo_cfg)

    t0 = time.perf_counter()
    Zmat = topo.rolling_transform(ts.astype("datetime64[ns]"), X, stride=stride)
    t1 = time.perf_counter()

    Fdim = X.shape[1]
    Zdim = Zmat.shape[1]
    if torch is None:
        return {
            "ph_time": t1 - t0,
            "forward_time": float("nan"),
            "events": T,
            "feature_dim": Fdim,
            "topo_dim": Zdim,
        }

    try:
        from ..models.hybrid import (
            HybridLOBModel,
        )
    except Exception:
        return {
            "ph_time": t1 - t0,
            "forward_time": float("nan"),
            # Fallback path before batch tensorization; use provided batch argument
            "events": T * int(batch),
            "feature_dim": Fdim,
            "topo_dim": Zdim,
        }

    cfg = ModelConfig(hidden_size=64, num_layers=1, include_market_embedding=False)
    model = HybridLOBModel(Fdim, Zdim, num_event_types=6, config=cfg)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)

    B = batch
    Xb = np.broadcast_to(X, (B, T, Fdim)).copy()
    Zb = np.broadcast_to(Zmat, (B, T, Zdim)).copy()
    xb = torch.from_numpy(Xb).to(dev)
    zb = torch.from_numpy(Zb).to(dev)

    torch.cuda.synchronize() if dev.type == "cuda" else None
    f0 = time.perf_counter()
    with torch.no_grad():
        out = model(xb, zb)
        _ = out.instability_logits.shape
    torch.cuda.synchronize() if dev.type == "cuda" else None
    f1 = time.perf_counter()

    # Peak RSS in MB (platform-dependent units; ru_maxrss is KB on Linux, bytes on macOS)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_kb = float(ru.ru_maxrss)
    peak_mb = rss_kb / 1024.0 if rss_kb > 10_000 else rss_kb / 1.0

    return {
        "ph_time": t1 - t0,
        "forward_time": f1 - f0,
        "events": T * B,
        "feature_dim": Fdim,
        "topo_dim": Zdim,
        "peak_rss_mb": peak_mb,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark features+PH+model throughput")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--T", type=int, default=5000, help="number of events per sequence")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--window-s", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="Number of warm-up runs to discard")
    ap.add_argument("--env", action="store_true", help="Include environment (CPU/GPU) info")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--sweep", action="store_true", help="Sweep L in {10,20,40} and window_s in {1,5,10}"
    )
    args = ap.parse_args()

    def mad(xs):
        m = median(xs)
        return median([abs(x - m) for x in xs])

    def env_info():
        info = {}
        try:
            import platform

            info["machine"] = platform.machine()
            info["processor"] = platform.processor()
            info["python"] = platform.python_version()
        except Exception:
            pass
        try:
            import torch as _t

            if _t.cuda.is_available():
                info["gpu"] = _t.cuda.get_device_name(0)
        except Exception:
            pass
        return info

    if args.sweep:
        Ls = [10, 20, 40]
        Ws = [1, 5, 10]
        print("L,window_s,ph_time_med_s,ph_time_mad_s,ph_events_per_s,fwd_time_med_s,latency_ms")
        for L in Ls:
            F = L * 2
            for w in Ws:
                results = []
                for _ in range(max(0, int(args.warmup))):
                    _ = run_once(
                        L=L,
                        T=args.T,
                        F=F,
                        Z=0,
                        batch=args.batch,
                        window_s=w,
                        stride=args.stride,
                        device=args.device,
                    )
                for _ in range(args.repeats):
                    res = run_once(
                        L=L,
                        T=args.T,
                        F=F,
                        Z=0,
                        batch=args.batch,
                        window_s=w,
                        stride=args.stride,
                        device=args.device,
                    )
                    results.append(res)
                ph_times = [r["ph_time"] for r in results]
                fwd_times = [r["forward_time"] for r in results if np.isfinite(r["forward_time"])]
                events = results[0]["events"]
                ph_med = float(median(ph_times))
                ph_mad = float(mad(ph_times))
                thr = float(events / ph_med if ph_med > 0 else float("nan"))
                fwd_med = float(median(fwd_times)) if fwd_times else float("nan")
                lat = float(
                    1000.0 * (ph_med + (fwd_med if np.isfinite(fwd_med) else 0.0)) / args.batch
                )
                print(f"{L},{w},{ph_med:.6f},{ph_mad:.6f},{thr:.2f},{fwd_med:.6f},{lat:.3f}")
        return

    L = args.levels
    F = L * 2
    results = []
    for _ in range(max(0, int(args.warmup))):
        _ = run_once(
            L=L,
            T=args.T,
            F=F,
            Z=0,
            batch=args.batch,
            window_s=args.window_s,
            stride=args.stride,
            device=args.device,
        )
    for _ in range(args.repeats):
        res = run_once(
            L=L,
            T=args.T,
            F=F,
            Z=0,
            batch=args.batch,
            window_s=args.window_s,
            stride=args.stride,
            device=args.device,
        )
        results.append(res)

    ph_times = [r["ph_time"] for r in results]
    fwd_times = [r["forward_time"] for r in results if np.isfinite(r["forward_time"])]
    events = results[0]["events"]
    out = {
        "events_per_run": events,
        "ph_time_median_s": float(median(ph_times)),
        "ph_time_mad_s": float(median([abs(x - median(ph_times)) for x in ph_times])),
        "ph_throughput_events_per_s": float(
            events / median(ph_times) if median(ph_times) > 0 else float("nan")
        ),
    }
    if fwd_times:
        fwd_med = float(median(fwd_times))
        out.update(
            {
                "forward_time_median_s": fwd_med,
                "forward_time_mad_s": float(median([abs(x - fwd_med) for x in fwd_times])),
                "end_to_end_latency_ms": float(
                    1000.0 * (out["ph_time_median_s"] + fwd_med) / args.batch
                ),
            }
        )
    if args.env:
        out["env"] = env_info()
    print(out)


if __name__ == "__main__":
    main()
