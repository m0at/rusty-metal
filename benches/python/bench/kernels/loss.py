"""Loss Functions: cross-entropy, MSE, MAE, Huber, KL divergence, BCE."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn


def run(n: int = 10_000_000):
    batch = n // 100
    classes = 100
    total = batch * classes

    logits_np = np.random.randn(batch, classes).astype(np.float32)
    targets_np = np.random.randint(0, classes, size=batch)
    pred_np = np.random.randn(batch).astype(np.float32)
    target_np = np.random.randn(batch).astype(np.float32)
    prob_np = np.clip(np.random.rand(batch).astype(np.float32), 1e-7, 1 - 1e-7)
    label_np = np.random.randint(0, 2, size=batch).astype(np.float32)

    # ── Cross-entropy ──
    bench_fn("loss_cross_entropy", "loss", "numpy",
             lambda: _np_cross_entropy(logits_np, targets_np), total)
    if HAS_TORCH:
        logits_cpu = torch.from_numpy(logits_np)
        targets_cpu = torch.from_numpy(targets_np).long()
        bench_fn("loss_cross_entropy", "loss", "torch_cpu",
                 lambda: F.cross_entropy(logits_cpu, targets_cpu), total)
        if HAS_TORCH_MPS:
            logits_mps = logits_cpu.to("mps")
            targets_mps = targets_cpu.to("mps")
            bench_fn("loss_cross_entropy", "loss", "torch_mps",
                     lambda: F.cross_entropy(logits_mps, targets_mps), total, sync=_sync_mps)
    if HAS_MLX:
        logits_mx = mx.array(logits_np)
        targets_mx = mx.array(targets_np)
        bench_fn("loss_cross_entropy", "loss", "mlx",
                 lambda: mx.eval(mx.mean(nn.losses.cross_entropy(logits_mx, targets_mx))), total, sync=_sync_mlx)

    # ── MSE ──
    bench_fn("loss_mse", "loss", "numpy",
             lambda: np.mean((pred_np - target_np) ** 2), batch)
    if HAS_TORCH:
        pred_cpu = torch.from_numpy(pred_np)
        tgt_cpu = torch.from_numpy(target_np)
        bench_fn("loss_mse", "loss", "torch_cpu",
                 lambda: F.mse_loss(pred_cpu, tgt_cpu), batch)
        if HAS_TORCH_MPS:
            pred_mps = pred_cpu.to("mps")
            tgt_mps = tgt_cpu.to("mps")
            bench_fn("loss_mse", "loss", "torch_mps",
                     lambda: F.mse_loss(pred_mps, tgt_mps), batch, sync=_sync_mps)
    if HAS_MLX:
        pred_mx = mx.array(pred_np)
        tgt_mx = mx.array(target_np)
        bench_fn("loss_mse", "loss", "mlx",
                 lambda: mx.eval(mx.mean(mx.square(pred_mx - tgt_mx))), batch, sync=_sync_mlx)

    # ── MAE ──
    bench_fn("loss_mae", "loss", "numpy",
             lambda: np.mean(np.abs(pred_np - target_np)), batch)
    if HAS_TORCH:
        bench_fn("loss_mae", "loss", "torch_cpu",
                 lambda: F.l1_loss(pred_cpu, tgt_cpu), batch)
        if HAS_TORCH_MPS:
            bench_fn("loss_mae", "loss", "torch_mps",
                     lambda: F.l1_loss(pred_mps, tgt_mps), batch, sync=_sync_mps)
    if HAS_MLX:
        bench_fn("loss_mae", "loss", "mlx",
                 lambda: mx.eval(mx.mean(mx.abs(pred_mx - tgt_mx))), batch, sync=_sync_mlx)

    # ── Huber ──
    bench_fn("loss_huber", "loss", "numpy",
             lambda: _np_huber(pred_np, target_np), batch)
    if HAS_TORCH:
        bench_fn("loss_huber", "loss", "torch_cpu",
                 lambda: F.huber_loss(pred_cpu, tgt_cpu), batch)
        if HAS_TORCH_MPS:
            bench_fn("loss_huber", "loss", "torch_mps",
                     lambda: F.huber_loss(pred_mps, tgt_mps), batch, sync=_sync_mps)

    # ── KL Divergence ──
    p_np = np.clip(np.random.rand(batch, classes).astype(np.float32), 1e-7, 1)
    p_np /= p_np.sum(axis=-1, keepdims=True)
    q_np_kl = np.clip(np.random.rand(batch, classes).astype(np.float32), 1e-7, 1)
    q_np_kl /= q_np_kl.sum(axis=-1, keepdims=True)

    bench_fn("loss_kl_div", "loss", "numpy",
             lambda: np.sum(p_np * (np.log(p_np) - np.log(q_np_kl))), total)
    if HAS_TORCH:
        p_cpu = torch.from_numpy(p_np)
        q_cpu_kl = torch.from_numpy(q_np_kl)
        bench_fn("loss_kl_div", "loss", "torch_cpu",
                 lambda: F.kl_div(q_cpu_kl.log(), p_cpu, reduction="sum"), total)

    # ── BCE ──
    bench_fn("loss_bce", "loss", "numpy",
             lambda: _np_bce(prob_np, label_np), batch)
    if HAS_TORCH:
        prob_cpu = torch.from_numpy(prob_np)
        lbl_cpu = torch.from_numpy(label_np)
        bench_fn("loss_bce", "loss", "torch_cpu",
                 lambda: F.binary_cross_entropy(prob_cpu, lbl_cpu), batch)
        if HAS_TORCH_MPS:
            prob_mps = prob_cpu.to("mps")
            lbl_mps = lbl_cpu.to("mps")
            bench_fn("loss_bce", "loss", "torch_mps",
                     lambda: F.binary_cross_entropy(prob_mps, lbl_mps), batch, sync=_sync_mps)


def _np_cross_entropy(logits, targets):
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    log_softmax = logits - m - np.log(np.sum(e, axis=-1, keepdims=True))
    return -np.mean(log_softmax[np.arange(len(targets)), targets])

def _np_huber(pred, target, delta=1.0):
    diff = np.abs(pred - target)
    return np.mean(np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)))

def _np_bce(prob, label, eps=1e-7):
    p = np.clip(prob, eps, 1 - eps)
    return -np.mean(label * np.log(p) + (1 - label) * np.log(1 - p))
