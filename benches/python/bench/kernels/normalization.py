"""Normalization: LayerNorm, BatchNorm, RMSNorm, GroupNorm."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    batch, hidden = 1024, n // 1024
    if hidden < 256:
        hidden = 256
        batch = n // hidden
    total = batch * hidden

    data_np = np.random.randn(batch, hidden).astype(np.float32)
    gamma_np = np.ones(hidden, dtype=np.float32)
    beta_np = np.zeros(hidden, dtype=np.float32)

    # ── LayerNorm ──
    bench_fn("layer_norm", "normalization", "numpy",
             lambda: _np_layernorm(data_np, gamma_np, beta_np), total)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        g_cpu = torch.from_numpy(gamma_np)
        b_cpu = torch.from_numpy(beta_np)
        bench_fn("layer_norm", "normalization", "torch_cpu",
                 lambda: F.layer_norm(data_cpu, [hidden], g_cpu, b_cpu), total)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            g_mps = g_cpu.to("mps")
            b_mps = b_cpu.to("mps")
            bench_fn("layer_norm", "normalization", "torch_mps",
                     lambda: F.layer_norm(data_mps, [hidden], g_mps, b_mps), total, sync=_sync_mps)
    if HAS_MLX:
        data_mx = mx.array(data_np)
        g_mx = mx.array(gamma_np)
        b_mx = mx.array(beta_np)
        bench_fn("layer_norm", "normalization", "mlx",
                 lambda: mx.eval(_mlx_layernorm(data_mx, g_mx, b_mx)), total, sync=_sync_mlx)

    # ── BatchNorm (reshape to NCHW-like: batch x channels) ──
    bench_fn("batch_norm", "normalization", "numpy",
             lambda: _np_batchnorm(data_np), total)
    if HAS_TORCH:
        # BatchNorm needs (N, C) or (N, C, ...) format
        bn_cpu = torch.nn.BatchNorm1d(hidden, affine=False).eval()
        bench_fn("batch_norm", "normalization", "torch_cpu",
                 lambda: bn_cpu(data_cpu), total)
        if HAS_TORCH_MPS:
            bn_mps = bn_cpu.to("mps")
            bench_fn("batch_norm", "normalization", "torch_mps",
                     lambda: bn_mps(data_mps), total, sync=_sync_mps)

    # ── RMSNorm ──
    bench_fn("rms_norm", "normalization", "numpy",
             lambda: _np_rmsnorm(data_np, gamma_np), total)
    if HAS_TORCH:
        bench_fn("rms_norm", "normalization", "torch_cpu",
                 lambda: _torch_rmsnorm(data_cpu, g_cpu), total)
        if HAS_TORCH_MPS:
            bench_fn("rms_norm", "normalization", "torch_mps",
                     lambda: _torch_rmsnorm(data_mps, g_mps), total, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("rms_norm", "normalization", "mlx",
                 lambda: mx.eval(_mlx_rmsnorm(data_mx, g_mx)), total, sync=_sync_mlx)

    # ── GroupNorm ──
    groups = 32
    chan = hidden if hidden % groups == 0 else (hidden // groups) * groups
    gn_np = np.random.randn(batch, chan).astype(np.float32)
    gn_total = batch * chan
    bench_fn("group_norm", "normalization", "numpy",
             lambda: _np_groupnorm(gn_np, groups), gn_total)
    if HAS_TORCH:
        gn_cpu = torch.from_numpy(gn_np).reshape(batch, chan, 1)  # (N, C, 1) for GroupNorm
        gn_layer = torch.nn.GroupNorm(groups, chan, affine=False).eval()
        bench_fn("group_norm", "normalization", "torch_cpu",
                 lambda: gn_layer(gn_cpu), gn_total)
        if HAS_TORCH_MPS:
            gn_mps = gn_cpu.to("mps")
            gn_layer_mps = gn_layer.to("mps")
            bench_fn("group_norm", "normalization", "torch_mps",
                     lambda: gn_layer_mps(gn_mps), gn_total, sync=_sync_mps)


# ── Numpy implementations ──

def _np_layernorm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def _np_batchnorm(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def _np_rmsnorm(x, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms

def _np_groupnorm(x, groups, eps=1e-5):
    b, c = x.shape
    x = x.reshape(b, groups, c // groups)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)).reshape(b, c)

# ── Torch/MLX implementations ──

if HAS_TORCH:
    def _torch_rmsnorm(x, gamma, eps=1e-5):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return gamma * x / rms

if HAS_MLX:
    def _mlx_layernorm(x, gamma, beta, eps=1e-5):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) * mx.rsqrt(var + eps) + beta

    def _mlx_rmsnorm(x, gamma, eps=1e-5):
        rms = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
        return gamma * x * rms
