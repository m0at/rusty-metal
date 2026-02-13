"""Dot Products & Similarity: batched dot, cosine similarity."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    batch = n // 128
    dim = 128
    total = batch * dim

    a_np = np.random.randn(batch, dim).astype(np.float32)
    b_np = np.random.randn(batch, dim).astype(np.float32)

    # ── Batched dot product ──
    bench_fn("dot_batched", "dot_products", "numpy",
             lambda: np.sum(a_np * b_np, axis=-1), total, elem_bytes=8)
    if HAS_TORCH:
        a_cpu = torch.from_numpy(a_np)
        b_cpu = torch.from_numpy(b_np)
        bench_fn("dot_batched", "dot_products", "torch_cpu",
                 lambda: torch.sum(a_cpu * b_cpu, dim=-1), total, elem_bytes=8)
        if HAS_TORCH_MPS:
            a_mps = a_cpu.to("mps")
            b_mps = b_cpu.to("mps")
            bench_fn("dot_batched", "dot_products", "torch_mps",
                     lambda: torch.sum(a_mps * b_mps, dim=-1), total, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        bench_fn("dot_batched", "dot_products", "mlx",
                 lambda: mx.eval(mx.sum(a_mx * b_mx, axis=-1)), total, elem_bytes=8, sync=_sync_mlx)

    # ── Cosine similarity ──
    bench_fn("cosine_similarity", "dot_products", "numpy",
             lambda: _np_cosine_sim(a_np, b_np), total, elem_bytes=8)
    if HAS_TORCH:
        bench_fn("cosine_similarity", "dot_products", "torch_cpu",
                 lambda: F.cosine_similarity(a_cpu, b_cpu, dim=-1), total, elem_bytes=8)
        if HAS_TORCH_MPS:
            bench_fn("cosine_similarity", "dot_products", "torch_mps",
                     lambda: F.cosine_similarity(a_mps, b_mps, dim=-1), total, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("cosine_similarity", "dot_products", "mlx",
                 lambda: mx.eval(_mlx_cosine_sim(a_mx, b_mx)), total, elem_bytes=8, sync=_sync_mlx)


def _np_cosine_sim(a, b):
    dot = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    return dot / (norm_a * norm_b + 1e-8)

if HAS_MLX:
    def _mlx_cosine_sim(a, b):
        dot = mx.sum(a * b, axis=-1)
        norm_a = mx.sqrt(mx.sum(a * a, axis=-1))
        norm_b = mx.sqrt(mx.sum(b * b, axis=-1))
        return dot / (norm_a * norm_b + 1e-8)
