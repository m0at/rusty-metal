//! Dot product benchmarks: batched dot, cosine similarity.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let dim = 128;
    let batch = n / dim;
    let total = batch * dim;

    let a: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // --- Scalar Rust: batched dot ---
    suite.add(bench_fn("dot_batched", "dot_products", "rust_scalar", || {
        let mut out = vec![0.0f32; batch];
        for i in 0..batch {
            let base = i * dim;
            let mut sum = 0.0f32;
            for d in 0..dim {
                sum += a[base + d] * b[base + d];
            }
            out[i] = sum;
        }
        let _ = out;
    }, total, 8));

    // --- Scalar Rust: cosine similarity ---
    suite.add(bench_fn("cosine_similarity", "dot_products", "rust_scalar", || {
        let mut out = vec![0.0f32; batch];
        for i in 0..batch {
            let base = i * dim;
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for d in 0..dim {
                let ai = a[base + d];
                let bi = b[base + d];
                dot += ai * bi;
                norm_a += ai * ai;
                norm_b += bi * bi;
            }
            out[i] = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-8);
        }
        let _ = out;
    }, total, 8));

    // --- NEON SIMD: single large dot product ---
    suite.add(bench_fn("dot_batched", "dot_products", "neon_simd", || {
        let _ = neon::dot_f32(&a, &b);
    }, total, 8));

    // --- Metal GPU ---
    let buf_a = ctx.buffer_from_slice(&a);
    let buf_b = ctx.buffer_from_slice(&b);
    let buf_out = ctx.buffer_empty(batch * 4);
    let buf_dim = ctx.buffer_from_slice(&[dim as u32]);

    let pso_dot = ctx.pipeline("dot_batched_f32", shaders::DOT_PRODUCTS).clone();
    suite.add(bench_fn("dot_batched", "dot_products", "metal", || {
        ctx.dispatch_1d(&pso_dot, &[&buf_a, &buf_b, &buf_out, &buf_dim], batch);
    }, total, 8));

    let pso_cos = ctx.pipeline("cosine_similarity_batched", shaders::DOT_PRODUCTS).clone();
    suite.add(bench_fn("cosine_similarity", "dot_products", "metal", || {
        ctx.dispatch_1d(&pso_cos, &[&buf_a, &buf_b, &buf_out, &buf_dim], batch);
    }, total, 8));
}
