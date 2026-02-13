//! Loss function benchmarks: cross-entropy, MSE, MAE, Huber, KL div, BCE.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let pred: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let target: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let prob_pred: Vec<f32> = (0..n).map(|_| rng.gen_range(0.01..0.99)).collect();
    let prob_target: Vec<f32> = (0..n).map(|_| rng.gen_range(0.01..0.99)).collect();
    let binary_target: Vec<f32> = (0..n).map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 }).collect();

    // --- Scalar Rust ---
    suite.add(bench_fn("loss_mse", "loss", "rust_scalar", || {
        black_box::<f32>(pred.iter().zip(&target).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / n as f32);
    }, n, 8));

    suite.add(bench_fn("loss_mae", "loss", "rust_scalar", || {
        black_box::<f32>(pred.iter().zip(&target).map(|(p, t)| (p - t).abs()).sum::<f32>() / n as f32);
    }, n, 8));

    suite.add(bench_fn("loss_huber", "loss", "rust_scalar", || {
        let delta = 1.0f32;
        black_box::<f32>(pred.iter().zip(&target).map(|(p, t)| {
            let d = (p - t).abs();
            if d <= delta { 0.5 * d * d } else { delta * (d - 0.5 * delta) }
        }).sum::<f32>() / n as f32);
    }, n, 8));

    suite.add(bench_fn("loss_kl_div", "loss", "rust_scalar", || {
        black_box::<f32>(prob_pred.iter().zip(&prob_target).map(|(&p, &q)| {
            let p = p.max(1e-7);
            let q = q.max(1e-7);
            p * (p / q).ln()
        }).sum::<f32>() / n as f32);
    }, n, 8));

    suite.add(bench_fn("loss_bce", "loss", "rust_scalar", || {
        black_box::<f32>(prob_pred.iter().zip(&binary_target).map(|(&p, &t)| {
            let p = p.clamp(1e-7, 1.0 - 1e-7);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        }).sum::<f32>() / n as f32);
    }, n, 8));

    // --- NEON SIMD ---
    suite.add(bench_fn("loss_mse", "loss", "neon_simd", || {
        black_box(neon::mse_f32(&pred, &target));
    }, n, 8));

    suite.add(bench_fn("loss_mae", "loss", "neon_simd", || {
        black_box(neon::mae_f32(&pred, &target));
    }, n, 8));

    // --- Metal GPU ---
    let buf_pred = ctx.buffer_from_slice(&pred);
    let buf_target = ctx.buffer_from_slice(&target);
    let buf_out = ctx.buffer_empty(n * 4);

    let pso_mse = ctx.pipeline("loss_mse_f32", shaders::LOSS).clone();
    suite.add(bench_fn("loss_mse", "loss", "metal", || {
        ctx.dispatch_1d(&pso_mse, &[&buf_pred, &buf_target, &buf_out], n);
    }, n, 8));

    let pso_mae = ctx.pipeline("loss_mae_f32", shaders::LOSS).clone();
    suite.add(bench_fn("loss_mae", "loss", "metal", || {
        ctx.dispatch_1d(&pso_mae, &[&buf_pred, &buf_target, &buf_out], n);
    }, n, 8));

    let buf_delta = ctx.buffer_from_slice(&[1.0f32]);
    let pso_huber = ctx.pipeline("loss_huber_f32", shaders::LOSS).clone();
    suite.add(bench_fn("loss_huber", "loss", "metal", || {
        ctx.dispatch_1d(&pso_huber, &[&buf_pred, &buf_target, &buf_out, &buf_delta], n);
    }, n, 8));

    let buf_pp = ctx.buffer_from_slice(&prob_pred);
    let buf_pt = ctx.buffer_from_slice(&prob_target);
    let pso_kl = ctx.pipeline("loss_kl_divergence_f32", shaders::LOSS).clone();
    suite.add(bench_fn("loss_kl_div", "loss", "metal", || {
        ctx.dispatch_1d(&pso_kl, &[&buf_pp, &buf_pt, &buf_out], n);
    }, n, 8));

    let buf_bt = ctx.buffer_from_slice(&binary_target);
    let pso_bce = ctx.pipeline("loss_binary_cross_entropy_f32", shaders::LOSS).clone();
    suite.add(bench_fn("loss_bce", "loss", "metal", || {
        ctx.dispatch_1d(&pso_bce, &[&buf_pp, &buf_bt, &buf_out], n);
    }, n, 8));
}
