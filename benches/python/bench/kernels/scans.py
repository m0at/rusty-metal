"""Scans & Compaction: exclusive scan, inclusive scan, compact_if, compact_mask."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    data_np = np.random.randn(n).astype(np.float32)
    data_u32 = np.random.randint(0, 1000, n, dtype=np.uint32)

    # ── Exclusive scan (prefix sum) ──
    bench_fn("scan_exclusive", "scans", "numpy",
             lambda: np.cumsum(data_u32) - data_u32, n, elem_bytes=4)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_u32.astype(np.int64))
        bench_fn("scan_exclusive", "scans", "torch_cpu",
                 lambda: torch.cumsum(data_cpu, dim=0) - data_cpu, n, elem_bytes=4)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            bench_fn("scan_exclusive", "scans", "torch_mps",
                     lambda: torch.cumsum(data_mps, dim=0) - data_mps, n, elem_bytes=4, sync=_sync_mps)
    if HAS_MLX:
        data_mx = mx.array(data_u32.astype(np.int32))
        bench_fn("scan_exclusive", "scans", "mlx",
                 lambda: mx.eval(mx.cumsum(data_mx, axis=0) - data_mx), n, elem_bytes=4, sync=_sync_mlx)

    # ── Inclusive scan ──
    bench_fn("scan_inclusive", "scans", "numpy",
             lambda: np.cumsum(data_u32), n, elem_bytes=4)
    if HAS_TORCH:
        bench_fn("scan_inclusive", "scans", "torch_cpu",
                 lambda: torch.cumsum(data_cpu, dim=0), n, elem_bytes=4)
        if HAS_TORCH_MPS:
            bench_fn("scan_inclusive", "scans", "torch_mps",
                     lambda: torch.cumsum(data_mps, dim=0), n, elem_bytes=4, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("scan_inclusive", "scans", "mlx",
                 lambda: mx.eval(mx.cumsum(data_mx, axis=0)), n, elem_bytes=4, sync=_sync_mlx)

    # ── Compact with mask (stream compaction) ──
    mask = np.random.rand(n) > 0.5
    bench_fn("compact_mask", "scans", "numpy",
             lambda: data_np[mask], n)
    if HAS_TORCH:
        data_f_cpu = torch.from_numpy(data_np)
        mask_cpu = torch.from_numpy(mask)
        bench_fn("compact_mask", "scans", "torch_cpu",
                 lambda: data_f_cpu[mask_cpu], n)
        if HAS_TORCH_MPS:
            data_f_mps = data_f_cpu.to("mps")
            mask_mps = mask_cpu.to("mps")
            bench_fn("compact_mask", "scans", "torch_mps",
                     lambda: data_f_mps[mask_mps], n, sync=_sync_mps)
    if HAS_MLX:
        data_f_mx = mx.array(data_np)
        mask_mx = mx.array(mask)
        bench_fn("compact_mask", "scans", "mlx",
                 lambda: mx.eval(data_f_mx[mask_mx]), n, sync=_sync_mlx)

    # ── Compact if (threshold-based) ──
    threshold = 0.0
    bench_fn("compact_if", "scans", "numpy",
             lambda: data_np[data_np > threshold], n)
    if HAS_TORCH:
        bench_fn("compact_if", "scans", "torch_cpu",
                 lambda: data_f_cpu[data_f_cpu > threshold], n)
