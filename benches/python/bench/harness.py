"""Benchmark harness — timing, warmup, statistics, JSON output."""

import json
import time
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable

# Backend availability
HAS_TORCH = False
HAS_TORCH_MPS = False
HAS_MLX = False

try:
    import torch
    HAS_TORCH = True
    HAS_TORCH_MPS = torch.backends.mps.is_available()
except ImportError:
    pass

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    pass

import numpy as np


WARMUP_RUNS = 5
BENCH_RUNS = 20
DEFAULT_N = 10_000_000
DEFAULT_MATRIX = (4096, 4096)
DEFAULT_SEQ = 2048
DEFAULT_HEADS = 32
DEFAULT_DIM = 128


@dataclass
class BenchResult:
    kernel: str
    domain: str
    backend: str
    n: int
    median_us: float
    min_us: float
    max_us: float
    throughput_meps: float  # million elements per second
    bytes_per_sec_gb: float  # GB/s bandwidth


@dataclass
class BenchSuite:
    results: list[BenchResult] = field(default_factory=list)

    def add(self, result: BenchResult):
        self.results.append(result)

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self.results], indent=2)

    def print_table(self, domain: str | None = None):
        filtered = self.results
        if domain:
            filtered = [r for r in filtered if r.domain == domain]

        if not filtered:
            return

        # Group by kernel
        kernels: dict[str, list[BenchResult]] = {}
        for r in filtered:
            kernels.setdefault(r.kernel, []).append(r)

        header = f"{'Kernel':<35} {'Backend':<12} {'Median (µs)':>12} {'Throughput':>14} {'BW (GB/s)':>10}"
        print(header)
        print("─" * len(header))

        for kernel_name, results in kernels.items():
            for r in sorted(results, key=lambda x: x.median_us):
                tp = f"{r.throughput_meps:.1f} Melem/s"
                print(f"{r.kernel:<35} {r.backend:<12} {r.median_us:>12.1f} {tp:>14} {r.bytes_per_sec_gb:>10.1f}")
            print()


SUITE = BenchSuite()


def _time_fn(fn: Callable, warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> list[float]:
    """Time a function, return list of durations in microseconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # ns -> µs
    return times


def _sync_mps():
    """Synchronize MPS to get accurate timing."""
    if HAS_TORCH_MPS:
        torch.mps.synchronize()


def _sync_mlx():
    """Synchronize MLX to get accurate timing."""
    if HAS_MLX:
        mx.eval(mx.array(0))  # force sync


def bench_fn(
    kernel: str,
    domain: str,
    backend: str,
    fn: Callable,
    n: int,
    elem_bytes: int = 4,
    sync: Callable | None = None,
):
    """Benchmark a single function and record the result."""
    if sync:
        def timed():
            fn()
            sync()
    else:
        timed = fn

    times = _time_fn(timed)
    med = statistics.median(times)
    lo = min(times)
    hi = max(times)
    throughput = n / med  # elements per µs = Melem/s
    bw = (n * elem_bytes) / (med * 1e-6) / 1e9  # GB/s

    result = BenchResult(
        kernel=kernel,
        domain=domain,
        backend=backend,
        n=n,
        median_us=round(med, 1),
        min_us=round(lo, 1),
        max_us=round(hi, 1),
        throughput_meps=round(throughput, 1),
        bytes_per_sec_gb=round(bw, 2),
    )
    SUITE.add(result)
    return result


def bench_unary(name: str, domain: str, np_fn, torch_fn=None, mlx_fn=None, n: int = DEFAULT_N):
    """Benchmark a unary elementwise operation across all backends."""
    data_np = np.random.randn(n).astype(np.float32)

    bench_fn(name, domain, "numpy", lambda: np_fn(data_np), n)

    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        if torch_fn:
            bench_fn(name, domain, "torch_cpu", lambda: torch_fn(data_cpu), n)

        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            _sync_mps()
            if torch_fn:
                bench_fn(name, domain, "torch_mps", lambda: torch_fn(data_mps), n, sync=_sync_mps)

    if HAS_MLX and mlx_fn:
        data_mx = mx.array(data_np)
        mx.eval(data_mx)
        bench_fn(name, domain, "mlx", lambda: mx.eval(mlx_fn(data_mx)), n, sync=_sync_mlx)


def bench_binary(name: str, domain: str, np_fn, torch_fn=None, mlx_fn=None, n: int = DEFAULT_N):
    """Benchmark a binary elementwise operation across all backends."""
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)

    bench_fn(name, domain, "numpy", lambda: np_fn(a_np, b_np), n, elem_bytes=8)

    if HAS_TORCH:
        a_cpu = torch.from_numpy(a_np)
        b_cpu = torch.from_numpy(b_np)
        if torch_fn:
            bench_fn(name, domain, "torch_cpu", lambda: torch_fn(a_cpu, b_cpu), n, elem_bytes=8)

        if HAS_TORCH_MPS:
            a_mps = a_cpu.to("mps")
            b_mps = b_cpu.to("mps")
            _sync_mps()
            if torch_fn:
                bench_fn(name, domain, "torch_mps", lambda: torch_fn(a_mps, b_mps), n, elem_bytes=8, sync=_sync_mps)

    if HAS_MLX and mlx_fn:
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        mx.eval(a_mx, b_mx)
        bench_fn(name, domain, "mlx", lambda: mx.eval(mlx_fn(a_mx, b_mx)), n, elem_bytes=8, sync=_sync_mlx)


def bench_reduce(name: str, domain: str, np_fn, torch_fn=None, mlx_fn=None, n: int = DEFAULT_N):
    """Benchmark a reduction operation across all backends."""
    data_np = np.random.randn(n).astype(np.float32)

    bench_fn(name, domain, "numpy", lambda: np_fn(data_np), n)

    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        if torch_fn:
            bench_fn(name, domain, "torch_cpu", lambda: torch_fn(data_cpu), n)

        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            _sync_mps()
            if torch_fn:
                bench_fn(name, domain, "torch_mps", lambda: torch_fn(data_mps), n, sync=_sync_mps)

    if HAS_MLX and mlx_fn:
        data_mx = mx.array(data_np)
        mx.eval(data_mx)
        bench_fn(name, domain, "mlx", lambda: mx.eval(mlx_fn(data_mx)), n, sync=_sync_mlx)
