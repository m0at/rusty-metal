//! Attention benchmarks: SDPA, RoPE, ALiBi.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, _n: usize) {
    let seq_len = 512;
    let head_dim = 64;
    let total = seq_len * head_dim;
    let mut rng = rand::thread_rng();

    let q: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let k: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // --- Scalar Rust: SDPA ---
    suite.add(bench_fn("attention_sdpa", "attention", "rust_scalar", || {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; total];
        for row in 0..seq_len {
            for col in 0..head_dim {
                let mut sum = 0.0f32;
                for kk in 0..seq_len {
                    let mut qk = 0.0f32;
                    for d in 0..head_dim {
                        qk += q[row * head_dim + d] * k[kk * head_dim + d];
                    }
                    sum += (qk * scale).exp() * v[kk * head_dim + col];
                }
                out[row * head_dim + col] = sum;
            }
        }
        black_box(out);
    }, total, 4));

    // --- Scalar Rust: RoPE ---
    let half_dim = head_dim / 2;
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

    suite.add(bench_fn("rope_apply", "attention", "rust_scalar", || {
        let mut x = q.clone();
        for pos in 0..seq_len {
            for d in 0..half_dim {
                let c = cos_table[pos * half_dim + d];
                let s = sin_table[pos * half_dim + d];
                let x0 = x[pos * head_dim + d];
                let x1 = x[pos * head_dim + d + half_dim];
                x[pos * head_dim + d] = x0 * c - x1 * s;
                x[pos * head_dim + d + half_dim] = x0 * s + x1 * c;
            }
        }
        black_box(x);
    }, total, 4));

    // --- Scalar Rust: ALiBi ---
    let attn_scores: Vec<f32> = (0..seq_len * seq_len).map(|_| rng.gen_range(-1.0..1.0)).collect();
    suite.add(bench_fn("alibi_bias", "attention", "rust_scalar", || {
        let slope = 0.125f32;
        let mut scores = attn_scores.clone();
        for q_pos in 0..seq_len {
            for k_pos in 0..seq_len {
                scores[q_pos * seq_len + k_pos] += slope * (k_pos as f32 - q_pos as f32);
            }
        }
        black_box(scores);
    }, seq_len * seq_len, 4));

    // --- Metal GPU ---
    let buf_q = ctx.buffer_from_slice(&q);
    let buf_k = ctx.buffer_from_slice(&k);
    let buf_v = ctx.buffer_from_slice(&v);
    let buf_out = ctx.buffer_empty(total * 4);
    let buf_seq = ctx.buffer_from_slice(&[seq_len as u32]);
    let buf_dim = ctx.buffer_from_slice(&[head_dim as u32]);

    let pso = ctx.pipeline("attention_sdpa_f32", shaders::ATTENTION).clone();
    suite.add(bench_fn("attention_sdpa", "attention", "metal", || {
        ctx.dispatch_2d(&pso, &[&buf_q, &buf_k, &buf_v, &buf_out, &buf_seq, &buf_dim], seq_len, head_dim);
    }, total, 4));

    let buf_cos = ctx.buffer_from_slice(&cos_table);
    let buf_sin = ctx.buffer_from_slice(&sin_table);
    let rope_data = q.clone();
    let buf_rope = ctx.buffer_from_slice(&rope_data);

    let pso_rope = ctx.pipeline("rope_apply_f32", shaders::ATTENTION).clone();
    suite.add(bench_fn("rope_apply", "attention", "metal", || {
        ctx.dispatch_1d(&pso_rope, &[&buf_rope, &buf_cos, &buf_sin, &buf_dim], total / 2);
    }, total, 4));
}
