"""Sorting: top-k, radix sort, bitonic sort, bitonic KV, median select, percentile select."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    sort_n = min(n, 1_000_000)
    data_np = np.random.randn(sort_n).astype(np.float32)

    # ── Top-k ──
    k = 100
    bench_fn("topk_select", "sorting", "numpy",
             lambda: _np_topk(data_np, k), sort_n)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        bench_fn("topk_select", "sorting", "torch_cpu",
                 lambda: torch.topk(data_cpu, k), sort_n)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            bench_fn("topk_select", "sorting", "torch_mps",
                     lambda: torch.topk(data_mps, k), sort_n, sync=_sync_mps)
    if HAS_MLX:
        data_mx = mx.array(data_np)
        bench_fn("topk_select", "sorting", "mlx",
                 lambda: mx.eval(mx.topk(data_mx, k)), sort_n, sync=_sync_mlx)

    # ── Radix sort (full sort) ──
    bench_fn("sort_radix", "sorting", "numpy",
             lambda: np.sort(data_np), sort_n)
    if HAS_TORCH:
        bench_fn("sort_radix", "sorting", "torch_cpu",
                 lambda: torch.sort(data_cpu), sort_n)
        if HAS_TORCH_MPS:
            bench_fn("sort_radix", "sorting", "torch_mps",
                     lambda: torch.sort(data_mps), sort_n, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("sort_radix", "sorting", "mlx",
                 lambda: mx.eval(mx.sort(data_mx)), sort_n, sync=_sync_mlx)

    # ── Bitonic sort (power-of-2 size) ──
    bitonic_n = 1 << 15  # 32768
    bitonic_np = np.random.randn(bitonic_n).astype(np.float32)
    bench_fn("sort_bitonic", "sorting", "numpy",
             lambda: np.sort(bitonic_np), bitonic_n)
    if HAS_TORCH:
        bitonic_cpu = torch.from_numpy(bitonic_np)
        bench_fn("sort_bitonic", "sorting", "torch_cpu",
                 lambda: torch.sort(bitonic_cpu), bitonic_n)

    # ── Bitonic sort KV (sort with indices) ──
    bench_fn("sort_bitonic_kv", "sorting", "numpy",
             lambda: np.argsort(bitonic_np), bitonic_n)
    if HAS_TORCH:
        bench_fn("sort_bitonic_kv", "sorting", "torch_cpu",
                 lambda: torch.sort(bitonic_cpu), bitonic_n)

    # ── Median select ──
    bench_fn("median_select", "sorting", "numpy",
             lambda: np.median(data_np), sort_n)
    if HAS_TORCH:
        bench_fn("median_select", "sorting", "torch_cpu",
                 lambda: torch.median(data_cpu), sort_n)

    # ── Percentile select ──
    bench_fn("percentile_select", "sorting", "numpy",
             lambda: np.percentile(data_np, [25, 50, 75, 90, 99]), sort_n)
    if HAS_TORCH:
        bench_fn("percentile_select", "sorting", "torch_cpu",
                 lambda: torch.quantile(data_cpu, torch.tensor([0.25, 0.5, 0.75, 0.9, 0.99])), sort_n)


def _np_topk(data, k):
    idx = np.argpartition(data, -k)[-k:]
    return data[idx]
