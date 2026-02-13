"""Elementwise unary and binary operations."""

import numpy as np
from bench.harness import bench_unary, bench_binary, DEFAULT_N, HAS_TORCH, HAS_MLX

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = DEFAULT_N):
    # ── Unary ops ──
    unary_ops = [
        ("map_exp",      np.exp,      torch.exp      if HAS_TORCH else None, mx.exp      if HAS_MLX else None),
        ("map_log",      np.log,      torch.log      if HAS_TORCH else None, mx.log      if HAS_MLX else None),
        ("map_sigmoid",  _np_sigmoid, torch.sigmoid  if HAS_TORCH else None, mx.sigmoid  if HAS_MLX else None),
        ("map_tanh",     np.tanh,     torch.tanh     if HAS_TORCH else None, mx.tanh     if HAS_MLX else None),
        ("map_softplus", _np_softplus, _torch_softplus if HAS_TORCH else None, _mlx_softplus if HAS_MLX else None),
        ("map_clamp",    _np_clamp,   _torch_clamp   if HAS_TORCH else None, _mlx_clamp   if HAS_MLX else None),
        ("map_abs",      np.abs,      torch.abs      if HAS_TORCH else None, mx.abs       if HAS_MLX else None),
    ]
    for name, np_fn, torch_fn, mlx_fn in unary_ops:
        bench_unary(name, "elementwise", np_fn, torch_fn, mlx_fn, n=n)

    # ── Binary ops ──
    binary_ops = [
        ("map_add",     np.add,      torch.add      if HAS_TORCH else None, mx.add      if HAS_MLX else None),
        ("map_mul",     np.multiply, torch.mul      if HAS_TORCH else None, mx.multiply if HAS_MLX else None),
        ("map_div",     np.divide,   torch.div      if HAS_TORCH else None, mx.divide   if HAS_MLX else None),
        ("map_compare", np.greater,  torch.gt       if HAS_TORCH else None, _mlx_gt     if HAS_MLX else None),
    ]
    for name, np_fn, torch_fn, mlx_fn in binary_ops:
        bench_binary(name, "elementwise", np_fn, torch_fn, mlx_fn, n=n)

    # ── FMA (ternary) ──
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    c_np = np.random.randn(n).astype(np.float32)
    from bench.harness import bench_fn, _sync_mps, _sync_mlx, HAS_TORCH_MPS
    bench_fn("map_fma", "elementwise", "numpy", lambda: a_np * b_np + c_np, n, elem_bytes=12)
    if HAS_TORCH:
        a_cpu, b_cpu, c_cpu = torch.from_numpy(a_np), torch.from_numpy(b_np), torch.from_numpy(c_np)
        bench_fn("map_fma", "elementwise", "torch_cpu", lambda: torch.addcmul(c_cpu, a_cpu, b_cpu), n, elem_bytes=12)
        if HAS_TORCH_MPS:
            a_m, b_m, c_m = a_cpu.to("mps"), b_cpu.to("mps"), c_cpu.to("mps")
            bench_fn("map_fma", "elementwise", "torch_mps",
                     lambda: torch.addcmul(c_m, a_m, b_m), n, elem_bytes=12, sync=_sync_mps)
    if HAS_MLX:
        a_mx, b_mx, c_mx = mx.array(a_np), mx.array(b_np), mx.array(c_np)
        bench_fn("map_fma", "elementwise", "mlx",
                 lambda: mx.eval(a_mx * b_mx + c_mx), n, elem_bytes=12, sync=_sync_mlx)


# ── Helper functions ──

def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _np_softplus(x):
    return np.log1p(np.exp(x))

def _np_clamp(x):
    return np.clip(x, -1.0, 1.0)

if HAS_TORCH:
    def _torch_softplus(x):
        return torch.nn.functional.softplus(x)
    def _torch_clamp(x):
        return torch.clamp(x, -1.0, 1.0)

if HAS_MLX:
    def _mlx_softplus(x):
        return mx.log(1 + mx.exp(x))
    def _mlx_clamp(x):
        return mx.clip(x, -1.0, 1.0)
    def _mlx_gt(a, b):
        return a > b
