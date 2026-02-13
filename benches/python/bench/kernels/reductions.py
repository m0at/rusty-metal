"""Reductions: sum, mean, min, max, L2, variance, stddev, histogram, argmax, argmin."""

import numpy as np
from bench.harness import (
    bench_reduce, bench_fn, DEFAULT_N, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX,
    _sync_mps, _sync_mlx,
)

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = DEFAULT_N):
    # ── Simple reductions (table-driven) ──
    simple = [
        ("reduce_sum",    np.sum,     torch.sum     if HAS_TORCH else None, mx.sum     if HAS_MLX else None),
        ("reduce_mean",   np.mean,    torch.mean    if HAS_TORCH else None, mx.mean    if HAS_MLX else None),
        ("reduce_min",    np.min,     torch.min     if HAS_TORCH else None, mx.min     if HAS_MLX else None),
        ("reduce_max",    np.max,     torch.max     if HAS_TORCH else None, mx.max     if HAS_MLX else None),
    ]
    for name, np_fn, torch_fn, mlx_fn in simple:
        bench_reduce(name, "reductions", np_fn, torch_fn, mlx_fn, n=n)

    # ── L2 norm ──
    data_np = np.random.randn(n).astype(np.float32)
    bench_fn("reduce_l2", "reductions", "numpy", lambda: np.linalg.norm(data_np), n)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        bench_fn("reduce_l2", "reductions", "torch_cpu", lambda: torch.linalg.norm(data_cpu), n)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            bench_fn("reduce_l2", "reductions", "torch_mps", lambda: torch.linalg.norm(data_mps), n, sync=_sync_mps)
    if HAS_MLX:
        data_mx = mx.array(data_np)
        bench_fn("reduce_l2", "reductions", "mlx", lambda: mx.eval(mx.sqrt(mx.sum(mx.square(data_mx)))), n, sync=_sync_mlx)

    # ── Variance ──
    bench_reduce("reduce_var", "reductions", np.var,
                 torch.var if HAS_TORCH else None,
                 mx.var if HAS_MLX else None, n=n)

    # ── Stddev ──
    bench_reduce("reduce_stddev", "reductions", np.std,
                 torch.std if HAS_TORCH else None,
                 None, n=n)  # MLX: compute manually
    if HAS_MLX:
        data_mx = mx.array(np.random.randn(n).astype(np.float32))
        bench_fn("reduce_stddev", "reductions", "mlx",
                 lambda: mx.eval(mx.sqrt(mx.var(data_mx))), n, sync=_sync_mlx)

    # ── Histogram ──
    data_np = np.random.randn(n).astype(np.float32)
    bench_fn("reduce_histogram", "reductions", "numpy",
             lambda: np.histogram(data_np, bins=256), n)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        bench_fn("reduce_histogram", "reductions", "torch_cpu",
                 lambda: torch.histc(data_cpu, bins=256), n)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            bench_fn("reduce_histogram", "reductions", "torch_mps",
                     lambda: torch.histc(data_mps, bins=256), n, sync=_sync_mps)

    # ── Argmax / Argmin ──
    bench_reduce("reduce_argmax", "reductions", np.argmax,
                 torch.argmax if HAS_TORCH else None,
                 mx.argmax if HAS_MLX else None, n=n)
    bench_reduce("reduce_argmin", "reductions", np.argmin,
                 torch.argmin if HAS_TORCH else None,
                 mx.argmin if HAS_MLX else None, n=n)
