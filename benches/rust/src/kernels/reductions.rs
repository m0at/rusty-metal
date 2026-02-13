//! Reduction benchmarks: sum, mean, min, max, L2, var, stddev, histogram, argmax, argmin.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // --- Scalar Rust ---
    suite.add(bench_fn("reduce_sum", "reductions", "rust_scalar", || {
        let _: f32 = data.iter().sum();
    }, n, 4));

    suite.add(bench_fn("reduce_mean", "reductions", "rust_scalar", || {
        let _: f32 = data.iter().sum::<f32>() / data.len() as f32;
    }, n, 4));

    suite.add(bench_fn("reduce_min", "reductions", "rust_scalar", || {
        let _ = data.iter().cloned().fold(f32::MAX, f32::min);
    }, n, 4));

    suite.add(bench_fn("reduce_max", "reductions", "rust_scalar", || {
        let _ = data.iter().cloned().fold(f32::MIN, f32::max);
    }, n, 4));

    suite.add(bench_fn("reduce_l2", "reductions", "rust_scalar", || {
        let _ = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    }, n, 4));

    suite.add(bench_fn("reduce_var", "reductions", "rust_scalar", || {
        let mean = data.iter().sum::<f32>() / n as f32;
        let _ = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    }, n, 4));

    suite.add(bench_fn("reduce_stddev", "reductions", "rust_scalar", || {
        let mean = data.iter().sum::<f32>() / n as f32;
        let _ = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    }, n, 4));

    suite.add(bench_fn("reduce_argmax", "reductions", "rust_scalar", || {
        let _ = data.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    }, n, 4));

    suite.add(bench_fn("reduce_argmin", "reductions", "rust_scalar", || {
        let _ = data.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    }, n, 4));

    suite.add(bench_fn("reduce_histogram", "reductions", "rust_scalar", || {
        let mut bins = [0u32; 256];
        for &v in &data {
            let bin = ((v * 128.0 + 128.0).clamp(0.0, 255.0)) as usize;
            bins[bin] += 1;
        }
        let _ = bins;
    }, n, 4));

    // --- NEON SIMD ---
    suite.add(bench_fn("reduce_sum", "reductions", "neon_simd", || {
        let _ = neon::sum_f32(&data);
    }, n, 4));

    suite.add(bench_fn("reduce_min", "reductions", "neon_simd", || {
        let _ = neon::min_f32(&data);
    }, n, 4));

    suite.add(bench_fn("reduce_max", "reductions", "neon_simd", || {
        let _ = neon::max_f32(&data);
    }, n, 4));

    suite.add(bench_fn("reduce_l2", "reductions", "neon_simd", || {
        let _ = neon::l2_squared_f32(&data).sqrt();
    }, n, 4));

    // --- Metal GPU ---
    let buf_in = ctx.buffer_from_slice(&data);
    let tg_size = 256;
    let n_groups = (n + tg_size - 1) / tg_size;
    let buf_out = ctx.buffer_empty(n_groups * 4);

    let pso_sum = ctx.pipeline("reduce_sum", shaders::REDUCTIONS).clone();
    suite.add(bench_fn("reduce_sum", "reductions", "metal", || {
        ctx.dispatch_reduce(&pso_sum, &[&buf_in, &buf_out], n, tg_size * 4);
    }, n, 4));

    let pso_min = ctx.pipeline("reduce_min", shaders::REDUCTIONS).clone();
    suite.add(bench_fn("reduce_min", "reductions", "metal", || {
        ctx.dispatch_reduce(&pso_min, &[&buf_in, &buf_out], n, tg_size * 4);
    }, n, 4));

    let pso_max = ctx.pipeline("reduce_max", shaders::REDUCTIONS).clone();
    suite.add(bench_fn("reduce_max", "reductions", "metal", || {
        ctx.dispatch_reduce(&pso_max, &[&buf_in, &buf_out], n, tg_size * 4);
    }, n, 4));

    let pso_l2 = ctx.pipeline("reduce_l2", shaders::REDUCTIONS).clone();
    suite.add(bench_fn("reduce_l2", "reductions", "metal", || {
        ctx.dispatch_reduce(&pso_l2, &[&buf_in, &buf_out], n, tg_size * 4);
    }, n, 4));
}
