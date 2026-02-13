"""Optimizers & Training: SGD, SGD+momentum, Adam, AdamW, gradient clipping."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    lr = 0.001
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    wd = 0.01
    momentum = 0.9

    params_np = np.random.randn(n).astype(np.float32)
    grads_np = np.random.randn(n).astype(np.float32)
    m_np = np.zeros(n, dtype=np.float32)
    v_np = np.zeros(n, dtype=np.float32)
    vel_np = np.zeros(n, dtype=np.float32)

    # ── SGD ──
    bench_fn("opt_sgd", "optimizers", "numpy",
             lambda: params_np - lr * grads_np, n, elem_bytes=8)
    if HAS_TORCH:
        p_cpu = torch.from_numpy(params_np.copy())
        g_cpu = torch.from_numpy(grads_np)
        bench_fn("opt_sgd", "optimizers", "torch_cpu",
                 lambda: p_cpu - lr * g_cpu, n, elem_bytes=8)
        if HAS_TORCH_MPS:
            p_mps = p_cpu.to("mps")
            g_mps = g_cpu.to("mps")
            bench_fn("opt_sgd", "optimizers", "torch_mps",
                     lambda: p_mps - lr * g_mps, n, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        p_mx = mx.array(params_np.copy())
        g_mx = mx.array(grads_np)
        bench_fn("opt_sgd", "optimizers", "mlx",
                 lambda: mx.eval(p_mx - lr * g_mx), n, elem_bytes=8, sync=_sync_mlx)

    # ── SGD + Momentum ──
    bench_fn("opt_sgd_momentum", "optimizers", "numpy",
             lambda: _np_sgd_momentum(params_np.copy(), grads_np, vel_np.copy(), lr, momentum), n, elem_bytes=12)
    if HAS_TORCH:
        bench_fn("opt_sgd_momentum", "optimizers", "torch_cpu",
                 lambda: _torch_sgd_momentum(p_cpu, g_cpu, lr, momentum), n, elem_bytes=12)

    # ── Adam ──
    bench_fn("opt_adam", "optimizers", "numpy",
             lambda: _np_adam(params_np.copy(), grads_np, m_np.copy(), v_np.copy(), 1, lr, beta1, beta2, eps),
             n, elem_bytes=20)
    if HAS_TORCH:
        bench_fn("opt_adam", "optimizers", "torch_cpu",
                 lambda: _torch_adam(p_cpu.clone(), g_cpu, lr, beta1, beta2, eps),
                 n, elem_bytes=20)
        if HAS_TORCH_MPS:
            bench_fn("opt_adam", "optimizers", "torch_mps",
                     lambda: _torch_adam(p_mps.clone(), g_mps, lr, beta1, beta2, eps),
                     n, elem_bytes=20, sync=_sync_mps)

    # ── AdamW ──
    bench_fn("opt_adamw", "optimizers", "numpy",
             lambda: _np_adamw(params_np.copy(), grads_np, m_np.copy(), v_np.copy(), 1, lr, beta1, beta2, eps, wd),
             n, elem_bytes=20)
    if HAS_TORCH:
        bench_fn("opt_adamw", "optimizers", "torch_cpu",
                 lambda: _torch_adamw(p_cpu.clone(), g_cpu, lr, beta1, beta2, eps, wd),
                 n, elem_bytes=20)

    # ── Gradient clipping (norm) ──
    bench_fn("grad_clip_norm", "optimizers", "numpy",
             lambda: _np_clip_norm(grads_np.copy(), 1.0), n)
    if HAS_TORCH:
        g_list = [g_cpu.clone()]
        bench_fn("grad_clip_norm", "optimizers", "torch_cpu",
                 lambda: torch.nn.utils.clip_grad_norm_(g_list, 1.0), n)

    # ── Gradient clipping (value) ──
    bench_fn("grad_clip_value", "optimizers", "numpy",
             lambda: np.clip(grads_np, -1.0, 1.0), n)
    if HAS_TORCH:
        bench_fn("grad_clip_value", "optimizers", "torch_cpu",
                 lambda: torch.clamp(g_cpu, -1.0, 1.0), n)
        if HAS_TORCH_MPS:
            bench_fn("grad_clip_value", "optimizers", "torch_mps",
                     lambda: torch.clamp(g_mps, -1.0, 1.0), n, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("grad_clip_value", "optimizers", "mlx",
                 lambda: mx.eval(mx.clip(g_mx, -1.0, 1.0)), n, sync=_sync_mlx)


def _np_sgd_momentum(p, g, v, lr, mu):
    v = mu * v + g
    return p - lr * v

def _np_adam(p, g, m, v, t, lr, b1, b2, eps):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)

def _np_adamw(p, g, m, v, t, lr, b1, b2, eps, wd):
    p = p - lr * wd * p
    return _np_adam(p, g, m, v, t, lr, b1, b2, eps)

def _np_clip_norm(g, max_norm):
    norm = np.linalg.norm(g)
    if norm > max_norm:
        g *= max_norm / norm
    return g

if HAS_TORCH:
    def _torch_sgd_momentum(p, g, lr, mu):
        v = mu * torch.zeros_like(p) + g
        return p - lr * v

    def _torch_adam(p, g, lr, b1, b2, eps):
        m = (1 - b1) * g
        v = (1 - b2) * g**2
        return p - lr * m / (torch.sqrt(v) + eps)

    def _torch_adamw(p, g, lr, b1, b2, eps, wd):
        p = p - lr * wd * p
        return _torch_adam(p, g, lr, b1, b2, eps)
