//! Correlation & Covariance benchmarks.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

    // --- Scalar Rust ---
    suite.add(bench_fn("reduce_correlation", "correlation", "rust_scalar", || {
        let mx = x.iter().sum::<f32>() / n as f32;
        let my = y.iter().sum::<f32>() / n as f32;
        let cov: f32 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sx: f32 = x.iter().map(|a| (a - mx).powi(2)).sum::<f32>().sqrt();
        let sy: f32 = y.iter().map(|b| (b - my).powi(2)).sum::<f32>().sqrt();
        let _ = cov / (sx * sy + 1e-8);
    }, n, 8));

    suite.add(bench_fn("reduce_covariance", "correlation", "rust_scalar", || {
        let mx = x.iter().sum::<f32>() / n as f32;
        let my = y.iter().sum::<f32>() / n as f32;
        let _ : f32 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum::<f32>() / n as f32;
    }, n, 8));

    suite.add(bench_fn("reduce_weighted_sum", "correlation", "rust_scalar", || {
        let _: f32 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
    }, n, 8));

    suite.add(bench_fn("reduce_weighted_mean", "correlation", "rust_scalar", || {
        let ws: f32 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
        let wt: f32 = w.iter().sum();
        let _ = ws / (wt + 1e-8);
    }, n, 8));

    // --- NEON SIMD ---
    suite.add(bench_fn("reduce_weighted_sum", "correlation", "neon_simd", || {
        let _ = neon::dot_f32(&x, &w);
    }, n, 8));

    // --- Metal GPU ---
    let buf_x = ctx.buffer_from_slice(&x);
    let buf_y = ctx.buffer_from_slice(&y);
    let buf_w = ctx.buffer_from_slice(&w);
    let tg_size = 256;
    let n_groups = (n + tg_size - 1) / tg_size;
    let buf_out = ctx.buffer_empty(n_groups * 4);

    let pso = ctx.pipeline("reduce_weighted_sum_f32", shaders::CORRELATION).clone();
    suite.add(bench_fn("reduce_weighted_sum", "correlation", "metal", || {
        ctx.dispatch_reduce(&pso, &[&buf_x, &buf_w, &buf_out], n, tg_size * 4);
    }, n, 8));

    let pso_corr = ctx.pipeline("reduce_correlation_f32", shaders::CORRELATION).clone();
    let buf_n = ctx.buffer_from_slice(&[n as u32]);
    suite.add(bench_fn("reduce_correlation", "correlation", "metal", || {
        ctx.dispatch_reduce(&pso_corr, &[&buf_x, &buf_y, &buf_out, &buf_n], n, tg_size * 4);
    }, n, 8));
}
