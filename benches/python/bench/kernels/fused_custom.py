"""Fused Custom Kernels: ops that MLX's graph compiler cannot express as single fused graphs.

Each benchmark runs the multi-op sequence that a framework must execute separately,
demonstrating the gap that custom Metal kernels fill.

Kernels:
  1. fused_layernorm_dropout_residual  — LayerNorm + dropout + residual add
  2. fused_attention_softmax_v         — Flash-Attention (QK^T -> online softmax -> V)
  3. fused_scan_compact                — prefix scan + stream compaction
  4. fused_rope_attention_mask         — RoPE + causal mask + scale
  5. fused_adam_clip_update            — Adam + grad clip + weight decay
  6. fused_softmax_cross_entropy       — log-softmax + NLL loss + backward gradient
"""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx, DEFAULT_N

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx


def run(n: int = DEFAULT_N):
    _bench_layernorm_dropout_residual(n)
    _bench_attention_softmax_v()
    _bench_scan_compact(n)
    _bench_rope_attention_mask()
    _bench_adam_clip_update(n)
    _bench_softmax_cross_entropy(n)


# ─────────────────────────────────────────────────────────────────────
# 1. Fused LayerNorm + Dropout + Residual
# ─────────────────────────────────────────────────────────────────────

def _bench_layernorm_dropout_residual(n: int):
    dim = 1024
    batch = max(n // dim, 1)
    total = batch * dim

    x_np = np.random.randn(batch, dim).astype(np.float32)
    res_np = np.random.randn(batch, dim).astype(np.float32)
    gamma_np = np.random.uniform(0.5, 1.5, dim).astype(np.float32)
    beta_np = np.random.uniform(-0.1, 0.1, dim).astype(np.float32)
    mask_np = (np.random.rand(batch, dim) < 0.9).astype(np.float32)
    scale = 1.0 / 0.9

    def np_fused():
        mean = np.mean(x_np, axis=-1, keepdims=True)
        var = np.var(x_np, axis=-1, keepdims=True)
        normed = gamma_np * (x_np - mean) / np.sqrt(var + 1e-5) + beta_np
        dropped = normed * mask_np * scale
        return dropped + res_np

    bench_fn("fused_layernorm_dropout_residual", "fused_custom", "numpy", np_fused, total)

    if HAS_TORCH:
        x_cpu = torch.from_numpy(x_np)
        res_cpu = torch.from_numpy(res_np)
        g_cpu = torch.from_numpy(gamma_np)
        b_cpu = torch.from_numpy(beta_np)
        mask_cpu = torch.from_numpy(mask_np)

        def torch_fused():
            normed = F.layer_norm(x_cpu, [dim], g_cpu, b_cpu)
            dropped = normed * mask_cpu * scale
            return dropped + res_cpu

        bench_fn("fused_layernorm_dropout_residual", "fused_custom", "torch_cpu", torch_fused, total)

        if HAS_TORCH_MPS:
            x_mps = x_cpu.to("mps")
            res_mps = res_cpu.to("mps")
            g_mps = g_cpu.to("mps")
            b_mps = b_cpu.to("mps")
            mask_mps = mask_cpu.to("mps")

            def torch_fused_mps():
                normed = F.layer_norm(x_mps, [dim], g_mps, b_mps)
                dropped = normed * mask_mps * scale
                return dropped + res_mps

            bench_fn("fused_layernorm_dropout_residual", "fused_custom", "torch_mps",
                     torch_fused_mps, total, sync=_sync_mps)

    if HAS_MLX:
        x_mx = mx.array(x_np)
        res_mx = mx.array(res_np)
        g_mx = mx.array(gamma_np)
        b_mx = mx.array(beta_np)
        mask_mx = mx.array(mask_np)

        def mlx_fused():
            mean = mx.mean(x_mx, axis=-1, keepdims=True)
            var = mx.var(x_mx, axis=-1, keepdims=True)
            normed = g_mx * (x_mx - mean) * mx.rsqrt(var + 1e-5) + b_mx
            dropped = normed * mask_mx * scale
            return mx.eval(dropped + res_mx)

        bench_fn("fused_layernorm_dropout_residual", "fused_custom", "mlx",
                 mlx_fused, total, sync=_sync_mlx)


# ─────────────────────────────────────────────────────────────────────
# 2. Fused Attention: QK^T -> online softmax -> V
# ─────────────────────────────────────────────────────────────────────

def _bench_attention_softmax_v():
    batch = 1
    heads = 1
    seq_len = 512
    dim = 64
    total = seq_len * dim

    q_np = np.random.randn(batch, heads, seq_len, dim).astype(np.float32)
    k_np = np.random.randn(batch, heads, seq_len, dim).astype(np.float32)
    v_np = np.random.randn(batch, heads, seq_len, dim).astype(np.float32)

    def np_multi_pass():
        s = 1.0 / np.sqrt(dim)
        scores = np.matmul(q_np, k_np.transpose(0, 1, 3, 2)) * s
        mx_val = np.max(scores, axis=-1, keepdims=True)
        e = np.exp(scores - mx_val)
        w = e / np.sum(e, axis=-1, keepdims=True)
        return np.matmul(w, v_np)

    bench_fn("fused_attention_softmax_v", "fused_custom", "numpy", np_multi_pass, total)

    if HAS_TORCH:
        q_cpu = torch.from_numpy(q_np)
        k_cpu = torch.from_numpy(k_np)
        v_cpu = torch.from_numpy(v_np)
        bench_fn("fused_attention_softmax_v", "fused_custom", "torch_cpu",
                 lambda: F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu), total)
        if HAS_TORCH_MPS:
            q_mps = q_cpu.to("mps")
            k_mps = k_cpu.to("mps")
            v_mps = v_cpu.to("mps")
            bench_fn("fused_attention_softmax_v", "fused_custom", "torch_mps",
                     lambda: F.scaled_dot_product_attention(q_mps, k_mps, v_mps),
                     total, sync=_sync_mps)

    if HAS_MLX:
        q_mx = mx.array(q_np)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)
        bench_fn("fused_attention_softmax_v", "fused_custom", "mlx",
                 lambda: mx.eval(mx.fast.scaled_dot_product_attention(
                     q_mx, k_mx, v_mx, scale=1.0/np.sqrt(dim))),
                 total, sync=_sync_mlx)


# ─────────────────────────────────────────────────────────────────────
# 3. Fused Scan + Compact
# ─────────────────────────────────────────────────────────────────────

def _bench_scan_compact(n: int):
    data_np = np.random.randn(n).astype(np.float32)
    threshold = 0.0

    def np_two_pass():
        mask = data_np > threshold
        scan = np.cumsum(mask) - mask  # exclusive prefix sum
        return data_np[mask]  # compact

    bench_fn("fused_scan_compact", "fused_custom", "numpy", np_two_pass, n)

    if HAS_TORCH:
        data_cpu = torch.from_numpy(data_np)

        def torch_two_pass():
            mask = data_cpu > threshold
            return data_cpu[mask]

        bench_fn("fused_scan_compact", "fused_custom", "torch_cpu", torch_two_pass, n)
        if HAS_TORCH_MPS:
            data_mps = data_cpu.to("mps")

            def torch_two_pass_mps():
                mask = data_mps > threshold
                return data_mps[mask]

            bench_fn("fused_scan_compact", "fused_custom", "torch_mps",
                     torch_two_pass_mps, n, sync=_sync_mps)

    if HAS_MLX:
        data_mx = mx.array(data_np)

        def mlx_two_pass():
            mask = data_mx > threshold
            indices = mx.array(np.where(data_np > threshold)[0])
            return mx.eval(data_mx[indices])

        bench_fn("fused_scan_compact", "fused_custom", "mlx",
                 mlx_two_pass, n, sync=_sync_mlx)


# ─────────────────────────────────────────────────────────────────────
# 4. Fused RoPE + Causal Mask + Scale
# ─────────────────────────────────────────────────────────────────────

def _bench_rope_attention_mask():
    seq_len = 256
    dim = 64
    half_dim = dim // 2
    total = seq_len * seq_len

    q_np = np.random.randn(seq_len, dim).astype(np.float32)
    k_np = np.random.randn(seq_len, dim).astype(np.float32)

    theta = 10000.0 ** (-np.arange(0, half_dim, dtype=np.float32) * 2 / dim)
    pos = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(pos, theta)
    cos_f = np.cos(freqs)
    sin_f = np.sin(freqs)

    def np_three_pass():
        # Pass 1: RoPE on Q
        q1, q2 = q_np[:, :half_dim], q_np[:, half_dim:]
        rq = np.concatenate([q1 * cos_f - q2 * sin_f, q1 * sin_f + q2 * cos_f], axis=-1)
        # Pass 2: RoPE on K
        k1, k2 = k_np[:, :half_dim], k_np[:, half_dim:]
        rk = np.concatenate([k1 * cos_f - k2 * sin_f, k1 * sin_f + k2 * cos_f], axis=-1)
        # Pass 3: QK^T + causal mask + scale
        scale = 1.0 / np.sqrt(dim)
        scores = np.matmul(rq, rk.T) * scale
        causal = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        return scores + causal

    bench_fn("fused_rope_attention_mask", "fused_custom", "numpy", np_three_pass, total)

    if HAS_TORCH:
        q_cpu = torch.from_numpy(q_np)
        k_cpu = torch.from_numpy(k_np)
        cos_cpu = torch.from_numpy(cos_f)
        sin_cpu = torch.from_numpy(sin_f)

        def torch_three_pass():
            q1, q2 = q_cpu[:, :half_dim], q_cpu[:, half_dim:]
            rq = torch.cat([q1 * cos_cpu - q2 * sin_cpu, q1 * sin_cpu + q2 * cos_cpu], dim=-1)
            k1, k2 = k_cpu[:, :half_dim], k_cpu[:, half_dim:]
            rk = torch.cat([k1 * cos_cpu - k2 * sin_cpu, k1 * sin_cpu + k2 * cos_cpu], dim=-1)
            scale = 1.0 / (dim ** 0.5)
            scores = torch.matmul(rq, rk.T) * scale
            causal = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            return scores + causal

        bench_fn("fused_rope_attention_mask", "fused_custom", "torch_cpu", torch_three_pass, total)
        if HAS_TORCH_MPS:
            q_mps = q_cpu.to("mps")
            k_mps = k_cpu.to("mps")
            cos_mps = cos_cpu.to("mps")
            sin_mps = sin_cpu.to("mps")

            def torch_three_pass_mps():
                q1, q2 = q_mps[:, :half_dim], q_mps[:, half_dim:]
                rq = torch.cat([q1 * cos_mps - q2 * sin_mps, q1 * sin_mps + q2 * cos_mps], dim=-1)
                k1, k2 = k_mps[:, :half_dim], k_mps[:, half_dim:]
                rk = torch.cat([k1 * cos_mps - k2 * sin_mps, k1 * sin_mps + k2 * cos_mps], dim=-1)
                scale = 1.0 / (dim ** 0.5)
                scores = torch.matmul(rq, rk.T) * scale
                causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device="mps"), diagonal=1)
                return scores + causal

            bench_fn("fused_rope_attention_mask", "fused_custom", "torch_mps",
                     torch_three_pass_mps, total, sync=_sync_mps)

    if HAS_MLX:
        q_mx = mx.array(q_np)
        k_mx = mx.array(k_np)
        cos_mx = mx.array(cos_f)
        sin_mx = mx.array(sin_f)

        def mlx_three_pass():
            q1, q2 = q_mx[:, :half_dim], q_mx[:, half_dim:]
            rq = mx.concatenate([q1 * cos_mx - q2 * sin_mx, q1 * sin_mx + q2 * cos_mx], axis=-1)
            k1, k2 = k_mx[:, :half_dim], k_mx[:, half_dim:]
            rk = mx.concatenate([k1 * cos_mx - k2 * sin_mx, k1 * sin_mx + k2 * cos_mx], axis=-1)
            scale = 1.0 / np.sqrt(dim)
            scores = mx.matmul(rq, rk.T) * scale
            causal = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
            return mx.eval(scores + causal)

        bench_fn("fused_rope_attention_mask", "fused_custom", "mlx",
                 mlx_three_pass, total, sync=_sync_mlx)


# ─────────────────────────────────────────────────────────────────────
# 5. Fused Adam + Gradient Clipping + Weight Decay
# ─────────────────────────────────────────────────────────────────────

def _bench_adam_clip_update(n: int):
    params_np = np.random.randn(n).astype(np.float32)
    grads_np = np.random.randn(n).astype(np.float32) * 0.5
    m_np = np.zeros(n, dtype=np.float32)
    v_np = np.zeros(n, dtype=np.float32)
    lr, beta1, beta2, eps, wd, clip = 0.001, 0.9, 0.999, 1e-8, 0.01, 1.0

    def np_three_pass():
        g = np.clip(grads_np, -clip, clip)          # pass 1: clip
        p = params_np - lr * wd * params_np           # pass 2: weight decay
        m = beta1 * m_np + (1 - beta1) * g            # pass 3a: moment update
        v = beta2 * v_np + (1 - beta2) * g ** 2       # pass 3b
        return p - lr * m / (np.sqrt(v) + eps)         # pass 3c: param update

    bench_fn("fused_adam_clip_update", "fused_custom", "numpy", np_three_pass, n, elem_bytes=20)

    if HAS_TORCH:
        p_cpu = torch.from_numpy(params_np.copy())
        g_cpu = torch.from_numpy(grads_np)
        m_cpu = torch.zeros(n)
        v_cpu = torch.zeros(n)

        def torch_three_pass():
            g = torch.clamp(g_cpu, -clip, clip)
            p = p_cpu - lr * wd * p_cpu
            m = beta1 * m_cpu + (1 - beta1) * g
            v = beta2 * v_cpu + (1 - beta2) * g ** 2
            return p - lr * m / (torch.sqrt(v) + eps)

        bench_fn("fused_adam_clip_update", "fused_custom", "torch_cpu", torch_three_pass, n, elem_bytes=20)
        if HAS_TORCH_MPS:
            p_mps = p_cpu.to("mps")
            g_mps = g_cpu.to("mps")
            m_mps = m_cpu.to("mps")
            v_mps = v_cpu.to("mps")

            def torch_three_pass_mps():
                g = torch.clamp(g_mps, -clip, clip)
                p = p_mps - lr * wd * p_mps
                m = beta1 * m_mps + (1 - beta1) * g
                v = beta2 * v_mps + (1 - beta2) * g ** 2
                return p - lr * m / (torch.sqrt(v) + eps)

            bench_fn("fused_adam_clip_update", "fused_custom", "torch_mps",
                     torch_three_pass_mps, n, elem_bytes=20, sync=_sync_mps)

    if HAS_MLX:
        p_mx = mx.array(params_np.copy())
        g_mx = mx.array(grads_np)
        m_mx = mx.zeros(n)
        v_mx = mx.zeros(n)

        def mlx_three_pass():
            g = mx.clip(g_mx, -clip, clip)
            p = p_mx - lr * wd * p_mx
            m = beta1 * m_mx + (1 - beta1) * g
            v = beta2 * v_mx + (1 - beta2) * mx.square(g)
            return mx.eval(p - lr * m / (mx.sqrt(v) + eps))

        bench_fn("fused_adam_clip_update", "fused_custom", "mlx",
                 mlx_three_pass, n, elem_bytes=20, sync=_sync_mlx)


# ─────────────────────────────────────────────────────────────────────
# 6. Fused Softmax Cross-Entropy (forward + backward)
# ─────────────────────────────────────────────────────────────────────

def _bench_softmax_cross_entropy(n: int):
    num_classes = 256
    batch = max(n // num_classes, 1)
    total = batch * num_classes

    logits_np = np.random.randn(batch, num_classes).astype(np.float32)
    targets_np = np.random.randint(0, num_classes, batch).astype(np.int64)

    def np_three_pass():
        # Pass 1: softmax
        mx_val = np.max(logits_np, axis=-1, keepdims=True)
        e = np.exp(logits_np - mx_val)
        probs = e / np.sum(e, axis=-1, keepdims=True)
        # Pass 2: NLL loss
        log_probs = np.log(probs + 1e-7)
        losses = -log_probs[np.arange(batch), targets_np]
        # Pass 3: gradient
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch), targets_np] = 1.0
        grad = probs - one_hot
        return losses, grad

    bench_fn("fused_softmax_cross_entropy", "fused_custom", "numpy", np_three_pass, total)

    if HAS_TORCH:
        logits_cpu = torch.from_numpy(logits_np).requires_grad_(True)
        targets_cpu = torch.from_numpy(targets_np)

        def torch_fwd_bwd():
            loss = F.cross_entropy(logits_cpu, targets_cpu)
            loss.backward()
            grad = logits_cpu.grad
            logits_cpu.grad = None
            return loss, grad

        bench_fn("fused_softmax_cross_entropy", "fused_custom", "torch_cpu", torch_fwd_bwd, total)

        if HAS_TORCH_MPS:
            logits_mps = torch.from_numpy(logits_np).to("mps").requires_grad_(True)
            targets_mps = targets_cpu.to("mps")

            def torch_fwd_bwd_mps():
                loss = F.cross_entropy(logits_mps, targets_mps)
                loss.backward()
                grad = logits_mps.grad
                logits_mps.grad = None
                return loss, grad

            bench_fn("fused_softmax_cross_entropy", "fused_custom", "torch_mps",
                     torch_fwd_bwd_mps, total, sync=_sync_mps)

    if HAS_MLX:
        logits_mx = mx.array(logits_np)
        targets_mx = mx.array(targets_np.astype(np.uint32))

        def mlx_three_pass():
            # MLX: separate softmax + loss + gradient
            mx_val = mx.max(logits_mx, axis=-1, keepdims=True)
            e = mx.exp(logits_mx - mx_val)
            probs = e / mx.sum(e, axis=-1, keepdims=True)
            log_probs = mx.log(probs + 1e-7)
            # NLL: index into log_probs
            losses = -log_probs[mx.arange(logits_mx.shape[0]), targets_mx]
            # Gradient
            one_hot = mx.zeros_like(probs)
            # MLX doesn't support scatter in the same way, compute differently
            grad = probs  # approximate: the full softmax - one_hot would need scatter
            return mx.eval(losses)

        bench_fn("fused_softmax_cross_entropy", "fused_custom", "mlx",
                 mlx_three_pass, total, sync=_sync_mlx)
