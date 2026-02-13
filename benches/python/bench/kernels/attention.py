"""Attention & Positional Encoding: SDPA, Flash, MQA, GQA, RoPE, ALiBi."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    # Attention dimensions
    batch = 4
    heads = 32
    seq_len = 512
    dim = 128

    # ── SDPA (standard dot-product attention, short seq) ──
    total = batch * heads * seq_len * dim

    q_np = np.random.randn(batch, heads, seq_len, dim).astype(np.float32)
    k_np = np.random.randn(batch, heads, seq_len, dim).astype(np.float32)
    v_np = np.random.randn(batch, heads, seq_len, dim).astype(np.float32)

    bench_fn("attention_sdpa", "attention", "numpy",
             lambda: _np_sdpa(q_np, k_np, v_np), total)

    if HAS_TORCH:
        q_cpu = torch.from_numpy(q_np)
        k_cpu = torch.from_numpy(k_np)
        v_cpu = torch.from_numpy(v_np)
        bench_fn("attention_sdpa", "attention", "torch_cpu",
                 lambda: F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu), total)
        if HAS_TORCH_MPS:
            q_mps = q_cpu.to("mps")
            k_mps = k_cpu.to("mps")
            v_mps = v_cpu.to("mps")
            bench_fn("attention_sdpa", "attention", "torch_mps",
                     lambda: F.scaled_dot_product_attention(q_mps, k_mps, v_mps), total, sync=_sync_mps)
    if HAS_MLX:
        q_mx = mx.array(q_np)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)
        bench_fn("attention_sdpa", "attention", "mlx",
                 lambda: mx.eval(mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=1.0/np.sqrt(dim))),
                 total, sync=_sync_mlx)

    # ── Flash attention (long sequence) ──
    long_seq = 2048
    total_long = batch * heads * long_seq * dim
    ql_np = np.random.randn(batch, heads, long_seq, dim).astype(np.float32)
    kl_np = np.random.randn(batch, heads, long_seq, dim).astype(np.float32)
    vl_np = np.random.randn(batch, heads, long_seq, dim).astype(np.float32)

    bench_fn("attention_flash", "attention", "numpy",
             lambda: _np_sdpa(ql_np, kl_np, vl_np), total_long)
    if HAS_TORCH:
        ql_cpu = torch.from_numpy(ql_np)
        kl_cpu = torch.from_numpy(kl_np)
        vl_cpu = torch.from_numpy(vl_np)
        bench_fn("attention_flash", "attention", "torch_cpu",
                 lambda: F.scaled_dot_product_attention(ql_cpu, kl_cpu, vl_cpu), total_long)
        if HAS_TORCH_MPS:
            ql_mps = ql_cpu.to("mps")
            kl_mps = kl_cpu.to("mps")
            vl_mps = vl_cpu.to("mps")
            bench_fn("attention_flash", "attention", "torch_mps",
                     lambda: F.scaled_dot_product_attention(ql_mps, kl_mps, vl_mps), total_long, sync=_sync_mps)
    if HAS_MLX:
        ql_mx = mx.array(ql_np)
        kl_mx = mx.array(kl_np)
        vl_mx = mx.array(vl_np)
        bench_fn("attention_flash", "attention", "mlx",
                 lambda: mx.eval(mx.fast.scaled_dot_product_attention(ql_mx, kl_mx, vl_mx, scale=1.0/np.sqrt(dim))),
                 total_long, sync=_sync_mlx)

    # ── RoPE (rotary position embedding) ──
    rope_np = np.random.randn(batch, seq_len, heads, dim).astype(np.float32)
    rope_total = batch * seq_len * heads * dim

    bench_fn("rope_apply", "attention", "numpy",
             lambda: _np_rope(rope_np, dim), rope_total)
    if HAS_TORCH:
        rope_cpu = torch.from_numpy(rope_np)
        bench_fn("rope_apply", "attention", "torch_cpu",
                 lambda: _torch_rope(rope_cpu, dim), rope_total)
        if HAS_TORCH_MPS:
            rope_mps = rope_cpu.to("mps")
            bench_fn("rope_apply", "attention", "torch_mps",
                     lambda: _torch_rope(rope_mps, dim), rope_total, sync=_sync_mps)

    # ── ALiBi bias ──
    alibi_total = heads * seq_len * seq_len
    bench_fn("alibi_bias", "attention", "numpy",
             lambda: _np_alibi(heads, seq_len), alibi_total)
    if HAS_TORCH:
        bench_fn("alibi_bias", "attention", "torch_cpu",
                 lambda: _torch_alibi(heads, seq_len), alibi_total)


def _np_sdpa(q, k, v):
    d = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    w = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(w, v)


def _np_rope(x, dim):
    seq_len = x.shape[1]
    half = dim // 2
    theta = 10000.0 ** (-np.arange(0, half, dtype=np.float32) * 2 / dim)
    pos = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(pos, theta)
    cos_f = np.cos(freqs).reshape(1, seq_len, 1, half)
    sin_f = np.sin(freqs).reshape(1, seq_len, 1, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1)


def _np_alibi(num_heads, seq_len):
    slopes = 2.0 ** (-8.0 * np.arange(1, num_heads + 1, dtype=np.float32) / num_heads)
    positions = np.arange(seq_len, dtype=np.float32)
    bias = slopes[:, None, None] * (positions[None, :, None] - positions[None, None, :])
    return bias


if HAS_TORCH:
    def _torch_rope(x, dim):
        seq_len = x.shape[1]
        half = dim // 2
        theta = 10000.0 ** (-torch.arange(0, half, dtype=torch.float32, device=x.device) * 2 / dim)
        pos = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        freqs = torch.outer(pos, theta).reshape(1, seq_len, 1, half)
        cos_f = torch.cos(freqs)
        sin_f = torch.sin(freqs)
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1)

    def _torch_alibi(num_heads, seq_len):
        slopes = 2.0 ** (-8.0 * torch.arange(1, num_heads + 1, dtype=torch.float32) / num_heads)
        pos = torch.arange(seq_len, dtype=torch.float32)
        return slopes[:, None, None] * (pos[None, :, None] - pos[None, None, :])
