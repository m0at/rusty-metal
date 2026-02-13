"""PRNG: Philox, uniform float, normal float, dropout mask."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    # ── Philox (raw u32 generation) ──
    bench_fn("prng_philox", "prng", "numpy",
             lambda: np.random.randint(0, 2**32, size=n, dtype=np.uint32), n, elem_bytes=4)
    if HAS_TORCH:
        bench_fn("prng_philox", "prng", "torch_cpu",
                 lambda: torch.randint(0, 2**31, (n,), dtype=torch.int32), n, elem_bytes=4)

    # ── Uniform [0, 1) ──
    bench_fn("prng_uniform_f32", "prng", "numpy",
             lambda: np.random.rand(n).astype(np.float32), n)
    if HAS_TORCH:
        bench_fn("prng_uniform_f32", "prng", "torch_cpu",
                 lambda: torch.rand(n), n)
        if HAS_TORCH_MPS:
            bench_fn("prng_uniform_f32", "prng", "torch_mps",
                     lambda: torch.rand(n, device="mps"), n, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("prng_uniform_f32", "prng", "mlx",
                 lambda: mx.eval(mx.random.uniform(shape=(n,))), n, sync=_sync_mlx)

    # ── Normal N(0,1) ──
    bench_fn("prng_normal_f32", "prng", "numpy",
             lambda: np.random.randn(n).astype(np.float32), n)
    if HAS_TORCH:
        bench_fn("prng_normal_f32", "prng", "torch_cpu",
                 lambda: torch.randn(n), n)
        if HAS_TORCH_MPS:
            bench_fn("prng_normal_f32", "prng", "torch_mps",
                     lambda: torch.randn(n, device="mps"), n, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("prng_normal_f32", "prng", "mlx",
                 lambda: mx.eval(mx.random.normal(shape=(n,))), n, sync=_sync_mlx)

    # ── Dropout mask ──
    keep_prob = 0.9
    bench_fn("prng_dropout_mask", "prng", "numpy",
             lambda: (np.random.rand(n) < keep_prob).astype(np.uint8), n, elem_bytes=1)
    if HAS_TORCH:
        bench_fn("prng_dropout_mask", "prng", "torch_cpu",
                 lambda: (torch.rand(n) < keep_prob).to(torch.uint8), n, elem_bytes=1)
        if HAS_TORCH_MPS:
            bench_fn("prng_dropout_mask", "prng", "torch_mps",
                     lambda: (torch.rand(n, device="mps") < keep_prob).to(torch.uint8), n, elem_bytes=1, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("prng_dropout_mask", "prng", "mlx",
                 lambda: mx.eval((mx.random.uniform(shape=(n,)) < keep_prob).astype(mx.uint8)), n, elem_bytes=1, sync=_sync_mlx)
