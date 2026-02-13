"""Correlation & Covariance: Pearson, covariance matrix, weighted sum/mean, pairwise."""

import numpy as np
from bench.harness import (
    bench_fn, DEFAULT_N, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX,
    _sync_mps, _sync_mlx,
)

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = DEFAULT_N):
    ncols = 64
    nrows = n // ncols

    # ── Pearson correlation (single pair) ──
    x_np = np.random.randn(n).astype(np.float32)
    y_np = np.random.randn(n).astype(np.float32)

    bench_fn("reduce_correlation", "correlation", "numpy",
             lambda: np.corrcoef(x_np, y_np)[0, 1], n)
    if HAS_TORCH:
        x_cpu = torch.from_numpy(x_np)
        y_cpu = torch.from_numpy(y_np)
        bench_fn("reduce_correlation", "correlation", "torch_cpu",
                 lambda: torch.corrcoef(torch.stack([x_cpu, y_cpu]))[0, 1], n)
        if HAS_TORCH_MPS:
            x_mps = x_cpu.to("mps")
            y_mps = y_cpu.to("mps")
            bench_fn("reduce_correlation", "correlation", "torch_mps",
                     lambda: torch.corrcoef(torch.stack([x_mps, y_mps]))[0, 1], n, sync=_sync_mps)

    # ── Covariance (single pair) ──
    bench_fn("reduce_covariance", "correlation", "numpy",
             lambda: np.cov(x_np, y_np)[0, 1], n)
    if HAS_TORCH:
        bench_fn("reduce_covariance", "correlation", "torch_cpu",
                 lambda: torch.cov(torch.stack([x_cpu, y_cpu]))[0, 1], n)

    # ── Weighted sum ──
    w_np = np.random.randn(n).astype(np.float32)
    bench_fn("reduce_weighted_sum", "correlation", "numpy",
             lambda: np.dot(w_np, x_np), n, elem_bytes=8)
    if HAS_TORCH:
        w_cpu = torch.from_numpy(w_np)
        bench_fn("reduce_weighted_sum", "correlation", "torch_cpu",
                 lambda: torch.dot(w_cpu, x_cpu), n, elem_bytes=8)
        if HAS_TORCH_MPS:
            w_mps = w_cpu.to("mps")
            bench_fn("reduce_weighted_sum", "correlation", "torch_mps",
                     lambda: torch.dot(w_mps, x_mps), n, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        w_mx = mx.array(w_np)
        x_mx = mx.array(x_np)
        bench_fn("reduce_weighted_sum", "correlation", "mlx",
                 lambda: mx.eval(mx.sum(w_mx * x_mx)), n, elem_bytes=8, sync=_sync_mlx)

    # ── Weighted mean ──
    bench_fn("reduce_weighted_mean", "correlation", "numpy",
             lambda: np.average(x_np, weights=np.abs(w_np)), n, elem_bytes=8)

    # ── Covariance matrix ──
    mat_np = np.random.randn(nrows, ncols).astype(np.float32)
    total = nrows * ncols
    bench_fn("reduce_covariance_matrix", "correlation", "numpy",
             lambda: np.cov(mat_np, rowvar=False), total)
    if HAS_TORCH:
        mat_cpu = torch.from_numpy(mat_np)
        bench_fn("reduce_covariance_matrix", "correlation", "torch_cpu",
                 lambda: torch.cov(mat_cpu.T), total)

    # ── Pairwise Pearson ──
    bench_fn("reduce_pearson_pairwise", "correlation", "numpy",
             lambda: np.corrcoef(mat_np, rowvar=False), total)
