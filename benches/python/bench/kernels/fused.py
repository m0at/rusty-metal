"""Fused Map-Reduce: square+sum, abs+sum, masked sum, threshold count."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx, DEFAULT_N

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = DEFAULT_N):
    data_np = np.random.randn(n).astype(np.float32)
    mask_np = (np.random.rand(n) > 0.5).astype(np.float32)

    # ── Square + Sum (L2 squared) ──
    bench_fn("map_reduce_square_sum", "fused", "numpy",
             lambda: np.sum(data_np ** 2), n)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        bench_fn("map_reduce_square_sum", "fused", "torch_cpu",
                 lambda: torch.sum(data_cpu ** 2), n)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            bench_fn("map_reduce_square_sum", "fused", "torch_mps",
                     lambda: torch.sum(data_mps ** 2), n, sync=_sync_mps)
    if HAS_MLX:
        data_mx = mx.array(data_np)
        bench_fn("map_reduce_square_sum", "fused", "mlx",
                 lambda: mx.eval(mx.sum(mx.square(data_mx))), n, sync=_sync_mlx)

    # ── Abs + Sum (L1 norm) ──
    bench_fn("map_reduce_abs_sum", "fused", "numpy",
             lambda: np.sum(np.abs(data_np)), n)
    if HAS_TORCH:
        bench_fn("map_reduce_abs_sum", "fused", "torch_cpu",
                 lambda: torch.sum(torch.abs(data_cpu)), n)
        if HAS_TORCH_MPS:
            bench_fn("map_reduce_abs_sum", "fused", "torch_mps",
                     lambda: torch.sum(torch.abs(data_mps)), n, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("map_reduce_abs_sum", "fused", "mlx",
                 lambda: mx.eval(mx.sum(mx.abs(data_mx))), n, sync=_sync_mlx)

    # ── Masked sum ──
    bench_fn("map_reduce_masked_sum", "fused", "numpy",
             lambda: np.sum(data_np * mask_np), n, elem_bytes=8)
    if HAS_TORCH:
        mask_cpu = torch.from_numpy(mask_np)
        bench_fn("map_reduce_masked_sum", "fused", "torch_cpu",
                 lambda: torch.sum(data_cpu * mask_cpu), n, elem_bytes=8)
        if HAS_TORCH_MPS:
            mask_mps = mask_cpu.to("mps")
            bench_fn("map_reduce_masked_sum", "fused", "torch_mps",
                     lambda: torch.sum(data_mps * mask_mps), n, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        mask_mx = mx.array(mask_np)
        bench_fn("map_reduce_masked_sum", "fused", "mlx",
                 lambda: mx.eval(mx.sum(data_mx * mask_mx)), n, elem_bytes=8, sync=_sync_mlx)

    # ── Threshold count ──
    threshold = 0.0
    bench_fn("map_reduce_threshold_count", "fused", "numpy",
             lambda: np.sum(data_np > threshold), n)
    if HAS_TORCH:
        bench_fn("map_reduce_threshold_count", "fused", "torch_cpu",
                 lambda: torch.sum(data_cpu > threshold), n)
        if HAS_TORCH_MPS:
            bench_fn("map_reduce_threshold_count", "fused", "torch_mps",
                     lambda: torch.sum(data_mps > threshold), n, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("map_reduce_threshold_count", "fused", "mlx",
                 lambda: mx.eval(mx.sum(data_mx > threshold)), n, sync=_sync_mlx)
