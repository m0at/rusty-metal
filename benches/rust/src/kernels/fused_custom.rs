//! Fused custom kernel benchmarks: operations that MLX's graph compiler cannot produce.
//!
//! Each fused kernel is benchmarked against its multi-op scalar Rust equivalent to
//! demonstrate the benefit of single-kernel fusion on Metal GPU.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();

    // ========================================================================
    // 1. Fused LayerNorm + Dropout + Residual Add
    // ========================================================================
    {
        let dim = 1024;
        let batch = n / dim;
        let total = batch * dim;
        let data: Vec<f32> = (0..total).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let residual: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let gamma: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.5..1.5)).collect();
        let beta: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let drop_mask: Vec<u8> = (0..total).map(|_| if rng.gen_bool(0.9) { 1u8 } else { 0u8 }).collect();
        let drop_scale = 1.0f32 / 0.9f32;

        // --- Scalar Rust (3 separate passes) ---
        suite.add(bench_fn("fused_layernorm_dropout_residual", "fused_custom", "rust_scalar", || {
            let mut out = vec![0.0f32; total];
            for b in 0..batch {
                let row = &data[b * dim..(b + 1) * dim];
                let mean: f32 = row.iter().sum::<f32>() / dim as f32;
                let var: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
                let inv_std = 1.0 / (var + 1e-5f32).sqrt();
                for i in 0..dim {
                    let idx = b * dim + i;
                    let normed = gamma[i] * (row[i] - mean) * inv_std + beta[i];
                    let dropped = normed * drop_mask[idx] as f32 * drop_scale;
                    out[idx] = dropped + residual[idx];
                }
            }
            black_box(&out);
        }, total, 4));

        // --- Metal GPU (single fused kernel) ---
        let buf_in = ctx.buffer_from_slice(&data);
        let buf_res = ctx.buffer_from_slice(&residual);
        let buf_gamma = ctx.buffer_from_slice(&gamma);
        let buf_beta = ctx.buffer_from_slice(&beta);
        let buf_mask = ctx.buffer_from_slice(&drop_mask);
        let buf_out = ctx.buffer_empty(total * 4);
        let buf_dim = ctx.buffer_from_slice(&[dim as u32]);
        let buf_scale = ctx.buffer_from_slice(&[drop_scale]);

        let pso = ctx.pipeline("fused_layernorm_dropout_residual", shaders::FUSED_CUSTOM).clone();
        suite.add(bench_fn("fused_layernorm_dropout_residual", "fused_custom", "metal", || {
            ctx.dispatch_reduce(
                &pso,
                &[&buf_in, &buf_res, &buf_gamma, &buf_beta, &buf_mask, &buf_out, &buf_dim, &buf_scale],
                total,
                dim * 4,
            );
        }, total, 4));
    }

    // ========================================================================
    // 2. Fused Attention: QK^T -> Online Softmax -> V (Flash-Attention style)
    // ========================================================================
    {
        let seq_len = 512;
        let head_dim = 64;
        let total = seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let k: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let v: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // --- Scalar Rust (multi-pass: matmul + softmax + matmul) ---
        suite.add(bench_fn("fused_attention_softmax_v", "fused_custom", "rust_scalar", || {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut out = vec![0.0f32; total];
            for row in 0..seq_len {
                // Pass 1: compute QK^T scores and find max
                let mut scores = vec![0.0f32; seq_len];
                let mut max_s = f32::NEG_INFINITY;
                for kk in 0..seq_len {
                    let mut qk = 0.0f32;
                    for d in 0..head_dim {
                        qk += q[row * head_dim + d] * k[kk * head_dim + d];
                    }
                    scores[kk] = qk * scale;
                    if scores[kk] > max_s { max_s = scores[kk]; }
                }
                // Pass 2: softmax
                let mut sum_exp = 0.0f32;
                let mut weights = vec![0.0f32; seq_len];
                for kk in 0..seq_len {
                    weights[kk] = (scores[kk] - max_s).exp();
                    sum_exp += weights[kk];
                }
                for kk in 0..seq_len {
                    weights[kk] /= sum_exp;
                }
                // Pass 3: weighted sum of V
                for col in 0..head_dim {
                    let mut s = 0.0f32;
                    for kk in 0..seq_len {
                        s += weights[kk] * v[kk * head_dim + col];
                    }
                    out[row * head_dim + col] = s;
                }
            }
            black_box(&out);
        }, total, 4));

        // --- Metal GPU (single fused kernel with online softmax) ---
        let buf_q = ctx.buffer_from_slice(&q);
        let buf_k = ctx.buffer_from_slice(&k);
        let buf_v = ctx.buffer_from_slice(&v);
        let buf_out = ctx.buffer_empty(total * 4);
        let buf_seq = ctx.buffer_from_slice(&[seq_len as u32]);
        let buf_dim = ctx.buffer_from_slice(&[head_dim as u32]);

        let pso = ctx.pipeline("fused_attention_softmax_v", shaders::FUSED_CUSTOM).clone();
        suite.add(bench_fn("fused_attention_softmax_v", "fused_custom", "metal", || {
            ctx.dispatch_2d(&pso, &[&buf_q, &buf_k, &buf_v, &buf_out, &buf_seq, &buf_dim], seq_len, head_dim);
        }, total, 4));
    }

    // ========================================================================
    // 3. Fused Scan + Compact (single dispatch instead of 2)
    // ========================================================================
    {
        let tg_size = 256;
        let scan_n = (n / tg_size) * tg_size; // round to multiple of threadgroup size
        let data: Vec<f32> = (0..scan_n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let threshold = 0.0f32;

        // --- Scalar Rust (2 passes: scan then compact) ---
        suite.add(bench_fn("fused_scan_compact", "fused_custom", "rust_scalar", || {
            // Pass 1: prefix scan of predicate
            let mut scan = vec![0u32; scan_n];
            let mut acc = 0u32;
            for i in 0..scan_n {
                scan[i] = acc;
                if data[i] > threshold { acc += 1; }
            }
            // Pass 2: scatter
            let mut out = vec![0.0f32; acc as usize];
            for i in 0..scan_n {
                if data[i] > threshold {
                    out[scan[i] as usize] = data[i];
                }
            }
            black_box(&out);
        }, scan_n, 4));

        // --- Metal GPU (single fused dispatch) ---
        let buf_data = ctx.buffer_from_slice(&data);
        let buf_out = ctx.buffer_empty(scan_n * 4);
        let buf_count = ctx.buffer_from_slice(&[0u32]);
        let buf_thresh = ctx.buffer_from_slice(&[threshold]);

        let pso = ctx.pipeline("fused_scan_compact", shaders::FUSED_CUSTOM).clone();
        suite.add(bench_fn("fused_scan_compact", "fused_custom", "metal", || {
            ctx.dispatch_reduce(
                &pso,
                &[&buf_data, &buf_out, &buf_count, &buf_thresh],
                scan_n,
                tg_size * 4,
            );
        }, scan_n, 4));
    }

    // ========================================================================
    // 4. Fused RoPE + Causal Mask + Scale
    // ========================================================================
    {
        let seq_len = 256;
        let head_dim = 64;
        let half_dim = head_dim / 2;
        let total = seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let k: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let cos_table: Vec<f32> = (0..seq_len * half_dim).map(|i| {
            let pos = i / half_dim;
            let d = i % half_dim;
            let freq = 1.0 / 10000.0f32.powf(2.0 * d as f32 / head_dim as f32);
            (pos as f32 * freq).cos()
        }).collect();
        let sin_table: Vec<f32> = (0..seq_len * half_dim).map(|i| {
            let pos = i / half_dim;
            let d = i % half_dim;
            let freq = 1.0 / 10000.0f32.powf(2.0 * d as f32 / head_dim as f32);
            (pos as f32 * freq).sin()
        }).collect();

        // --- Scalar Rust (3 separate passes) ---
        suite.add(bench_fn("fused_rope_attention_mask", "fused_custom", "rust_scalar", || {
            let scale = 1.0 / (head_dim as f32).sqrt();
            // Pass 1: apply RoPE to Q
            let mut rq = q.clone();
            for pos in 0..seq_len {
                for d in 0..half_dim {
                    let c = cos_table[pos * half_dim + d];
                    let s = sin_table[pos * half_dim + d];
                    let x0 = rq[pos * head_dim + d];
                    let x1 = rq[pos * head_dim + d + half_dim];
                    rq[pos * head_dim + d] = x0 * c - x1 * s;
                    rq[pos * head_dim + d + half_dim] = x0 * s + x1 * c;
                }
            }
            // Pass 2: apply RoPE to K
            let mut rk = k.clone();
            for pos in 0..seq_len {
                for d in 0..half_dim {
                    let c = cos_table[pos * half_dim + d];
                    let s = sin_table[pos * half_dim + d];
                    let x0 = rk[pos * head_dim + d];
                    let x1 = rk[pos * head_dim + d + half_dim];
                    rk[pos * head_dim + d] = x0 * c - x1 * s;
                    rk[pos * head_dim + d + half_dim] = x0 * s + x1 * c;
                }
            }
            // Pass 3: QK^T with causal mask + scale
            let mut attn = vec![0.0f32; seq_len * seq_len];
            for qp in 0..seq_len {
                for kp in 0..seq_len {
                    if kp > qp {
                        attn[qp * seq_len + kp] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += rq[qp * head_dim + d] * rk[kp * head_dim + d];
                        }
                        attn[qp * seq_len + kp] = dot * scale;
                    }
                }
            }
            black_box(&attn);
        }, seq_len * seq_len, 4));

        // --- Metal GPU (single fused kernel) ---
        let buf_q = ctx.buffer_from_slice(&q);
        let buf_k = ctx.buffer_from_slice(&k);
        let buf_cos = ctx.buffer_from_slice(&cos_table);
        let buf_sin = ctx.buffer_from_slice(&sin_table);
        let buf_attn = ctx.buffer_empty(seq_len * seq_len * 4);
        let buf_seq = ctx.buffer_from_slice(&[seq_len as u32]);
        let buf_dim = ctx.buffer_from_slice(&[head_dim as u32]);

        let pso = ctx.pipeline("fused_rope_attention_mask", shaders::FUSED_CUSTOM).clone();
        suite.add(bench_fn("fused_rope_attention_mask", "fused_custom", "metal", || {
            ctx.dispatch_2d(&pso, &[&buf_q, &buf_k, &buf_cos, &buf_sin, &buf_attn, &buf_seq, &buf_dim], seq_len, seq_len);
        }, seq_len * seq_len, 4));
    }

    // ========================================================================
    // 5. Fused Adam + Gradient Clipping + Weight Decay
    // ========================================================================
    {
        let params: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let grads: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;
        let clip_val = 1.0f32;

        // --- Scalar Rust (3 separate passes) ---
        suite.add(bench_fn("fused_adam_clip_update", "fused_custom", "rust_scalar", || {
            let mut p = params.clone();
            let mut m = vec![0.0f32; n];
            let mut v = vec![0.0f32; n];
            // Pass 1: clip gradients
            let clipped: Vec<f32> = grads.iter().map(|&g| g.clamp(-clip_val, clip_val)).collect();
            // Pass 2: weight decay
            for i in 0..n { p[i] -= lr * weight_decay * p[i]; }
            // Pass 3: adam update
            for i in 0..n {
                m[i] = beta1 * m[i] + (1.0 - beta1) * clipped[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * clipped[i] * clipped[i];
                p[i] -= lr * m[i] / (v[i].sqrt() + eps);
            }
            black_box(&p);
        }, n, 20));

        // --- Metal GPU (single fused kernel) ---
        let buf_params = ctx.buffer_from_slice(&params);
        let buf_grads = ctx.buffer_from_slice(&grads);
        let buf_m = ctx.buffer_empty(n * 4);
        let buf_v = ctx.buffer_empty(n * 4);
        let buf_lr = ctx.buffer_from_slice(&[lr]);
        let buf_beta1 = ctx.buffer_from_slice(&[beta1]);
        let buf_beta2 = ctx.buffer_from_slice(&[beta2]);
        let buf_eps = ctx.buffer_from_slice(&[eps]);
        let buf_wd = ctx.buffer_from_slice(&[weight_decay]);
        let buf_clip = ctx.buffer_from_slice(&[clip_val]);

        let pso = ctx.pipeline("fused_adam_clip_update", shaders::FUSED_CUSTOM).clone();
        suite.add(bench_fn("fused_adam_clip_update", "fused_custom", "metal", || {
            ctx.dispatch_1d(
                &pso,
                &[&buf_params, &buf_grads, &buf_m, &buf_v, &buf_lr, &buf_beta1, &buf_beta2, &buf_eps, &buf_wd, &buf_clip],
                n,
            );
        }, n, 20));
    }

    // ========================================================================
    // 6. Fused Softmax Cross-Entropy (forward + backward in one pass)
    // ========================================================================
    {
        let num_classes = 256;
        let batch = n / num_classes;
        let total = batch * num_classes;
        let logits: Vec<f32> = (0..total).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let targets: Vec<u32> = (0..batch).map(|_| rng.gen_range(0..num_classes as u32)).collect();

        // --- Scalar Rust (3 passes: softmax + loss + gradient) ---
        suite.add(bench_fn("fused_softmax_cross_entropy", "fused_custom", "rust_scalar", || {
            let mut losses = vec![0.0f32; batch];
            let mut grad = vec![0.0f32; total];
            for b in 0..batch {
                let row = &logits[b * num_classes..(b + 1) * num_classes];
                // Pass 1: find max
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                // Pass 2: exp + sum
                let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                // Pass 3: loss + gradient
                let target = targets[b] as usize;
                losses[b] = -(row[target] - max_val - sum_exp.ln());
                for c in 0..num_classes {
                    let prob = exps[c] / sum_exp;
                    let one_hot = if c == target { 1.0 } else { 0.0 };
                    grad[b * num_classes + c] = prob - one_hot;
                }
            }
            black_box((&losses, &grad));
        }, total, 4));

        // --- Metal GPU (single fused kernel) ---
        let buf_logits = ctx.buffer_from_slice(&logits);
        let buf_targets = ctx.buffer_from_slice(&targets);
        let buf_loss = ctx.buffer_empty(batch * 4);
        let buf_grad = ctx.buffer_empty(total * 4);
        let buf_nc = ctx.buffer_from_slice(&[num_classes as u32]);

        let pso = ctx.pipeline("fused_softmax_cross_entropy", shaders::FUSED_CUSTOM).clone();
        suite.add(bench_fn("fused_softmax_cross_entropy", "fused_custom", "metal", || {
            ctx.dispatch_reduce(
                &pso,
                &[&buf_logits, &buf_targets, &buf_loss, &buf_grad, &buf_nc],
                total,
                num_classes * 4,
            );
        }, total, 4));
    }
}
