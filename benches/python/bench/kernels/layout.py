"""Layout Transforms: transpose, AoS-to-SoA, SoA-to-AoS."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    rows, cols = 4096, 4096
    total = rows * cols

    # ── Transpose 2D ──
    mat_np = np.random.randn(rows, cols).astype(np.float32)

    bench_fn("transpose_2d", "layout", "numpy",
             lambda: np.ascontiguousarray(mat_np.T), total)
    if HAS_TORCH:
        mat_cpu = torch.from_numpy(mat_np)
        bench_fn("transpose_2d", "layout", "torch_cpu",
                 lambda: mat_cpu.T.contiguous(), total)
        if HAS_TORCH_MPS:
            mat_mps = mat_cpu.to("mps")
            bench_fn("transpose_2d", "layout", "torch_mps",
                     lambda: mat_mps.T.contiguous(), total, sync=_sync_mps)
    if HAS_MLX:
        mat_mx = mx.array(mat_np)
        bench_fn("transpose_2d", "layout", "mlx",
                 lambda: mx.eval(mx.transpose(mat_mx)), total, sync=_sync_mlx)

    # ── AoS to SoA ──
    # Simulate struct of 4 floats: (x, y, z, w) x N
    n_structs = total // 4
    aos_np = np.random.randn(n_structs, 4).astype(np.float32)

    bench_fn("repack_aos_to_soa", "layout", "numpy",
             lambda: np.ascontiguousarray(aos_np.T), n_structs * 4)
    if HAS_TORCH:
        aos_cpu = torch.from_numpy(aos_np)
        bench_fn("repack_aos_to_soa", "layout", "torch_cpu",
                 lambda: aos_cpu.T.contiguous(), n_structs * 4)
        if HAS_TORCH_MPS:
            aos_mps = aos_cpu.to("mps")
            bench_fn("repack_aos_to_soa", "layout", "torch_mps",
                     lambda: aos_mps.T.contiguous(), n_structs * 4, sync=_sync_mps)

    # ── SoA to AoS ──
    soa_np = np.random.randn(4, n_structs).astype(np.float32)
    bench_fn("repack_soa_to_aos", "layout", "numpy",
             lambda: np.ascontiguousarray(soa_np.T), n_structs * 4)
    if HAS_TORCH:
        soa_cpu = torch.from_numpy(soa_np)
        bench_fn("repack_soa_to_aos", "layout", "torch_cpu",
                 lambda: soa_cpu.T.contiguous(), n_structs * 4)
