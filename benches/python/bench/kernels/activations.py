"""ML Activations: ReLU, LeakyReLU, ELU, GELU, SiLU/Swish, Mish."""

import numpy as np
from bench.harness import bench_unary, DEFAULT_N, HAS_TORCH, HAS_MLX

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as mnn


def run(n: int = DEFAULT_N):
    activations = [
        ("map_relu",       _np_relu,       F.relu        if HAS_TORCH else None, mnn.relu        if HAS_MLX else None),
        ("map_leaky_relu", _np_leaky_relu, _torch_lrelu  if HAS_TORCH else None, mnn.leaky_relu  if HAS_MLX else None),
        ("map_elu",        _np_elu,        F.elu         if HAS_TORCH else None, mnn.elu         if HAS_MLX else None),
        ("map_gelu",       _np_gelu,       F.gelu        if HAS_TORCH else None, mnn.gelu        if HAS_MLX else None),
        ("map_silu",       _np_silu,       F.silu        if HAS_TORCH else None, mnn.silu        if HAS_MLX else None),
        ("map_mish",       _np_mish,       F.mish        if HAS_TORCH else None, mnn.mish        if HAS_MLX else None),
    ]
    for name, np_fn, torch_fn, mlx_fn in activations:
        bench_unary(name, "activations", np_fn, torch_fn, mlx_fn, n=n)


def _np_relu(x):
    return np.maximum(0, x)

def _np_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def _np_elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def _np_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def _np_silu(x):
    return x / (1 + np.exp(-x))

def _np_mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

if HAS_TORCH:
    def _torch_lrelu(x):
        return F.leaky_relu(x, 0.01)
