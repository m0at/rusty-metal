"""Softmax: stable softmax, log-softmax, online softmax."""

import numpy as np
from bench.harness import bench_fn, DEFAULT_N, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx


def run(n: int = DEFAULT_N):
    rows, cols = n // 1024, 1024
    data_np = np.random.randn(rows, cols).astype(np.float32)
    total = rows * cols

    # ── Stable softmax ──
    bench_fn("softmax_stable", "softmax", "numpy", lambda: _np_softmax(data_np), total)
    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)
        bench_fn("softmax_stable", "softmax", "torch_cpu",
                 lambda: F.softmax(data_cpu, dim=-1), total)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")
            bench_fn("softmax_stable", "softmax", "torch_mps",
                     lambda: F.softmax(data_mps, dim=-1), total, sync=_sync_mps)
    if HAS_MLX:
        data_mx = mx.array(data_np)
        bench_fn("softmax_stable", "softmax", "mlx",
                 lambda: mx.eval(mx.softmax(data_mx, axis=-1)), total, sync=_sync_mlx)

    # ── Log softmax ──
    bench_fn("log_softmax", "softmax", "numpy", lambda: _np_log_softmax(data_np), total)
    if HAS_TORCH:
        bench_fn("log_softmax", "softmax", "torch_cpu",
                 lambda: F.log_softmax(data_cpu, dim=-1), total)
        if HAS_TORCH_MPS:
            bench_fn("log_softmax", "softmax", "torch_mps",
                     lambda: F.log_softmax(data_mps, dim=-1), total, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("log_softmax", "softmax", "mlx",
                 lambda: mx.eval(_mlx_log_softmax(data_mx)), total, sync=_sync_mlx)

    # ── Online softmax (same op, included for naming parity) ──
    bench_fn("softmax_online", "softmax", "numpy", lambda: _np_softmax(data_np), total)
    if HAS_TORCH:
        bench_fn("softmax_online", "softmax", "torch_cpu",
                 lambda: F.softmax(data_cpu, dim=-1), total)
        if HAS_TORCH_MPS:
            bench_fn("softmax_online", "softmax", "torch_mps",
                     lambda: F.softmax(data_mps, dim=-1), total, sync=_sync_mps)


def _np_softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)

def _np_log_softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))

if HAS_MLX:
    def _mlx_log_softmax(x):
        m = mx.max(x, axis=-1, keepdims=True)
        return x - m - mx.log(mx.sum(mx.exp(x - m), axis=-1, keepdims=True))
