//! Fused map-reduce benchmarks: square+sum, abs+sum, masked sum, threshold count.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mask: Vec<f32> = (0..n).map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 }).collect();

    // --- Scalar Rust ---
    suite.add(bench_fn("map_reduce_square_sum", "fused", "rust_scalar", || {
        let _: f32 = data.iter().map(|x| x * x).sum();
    }, n, 4));

    suite.add(bench_fn("map_reduce_abs_sum", "fused", "rust_scalar", || {
        let _: f32 = data.iter().map(|x| x.abs()).sum();
    }, n, 4));

    suite.add(bench_fn("map_reduce_masked_sum", "fused", "rust_scalar", || {
        let _: f32 = data.iter().zip(&mask).map(|(d, m)| d * m).sum();
    }, n, 8));

    suite.add(bench_fn("map_reduce_threshold_count", "fused", "rust_scalar", || {
        let _: usize = data.iter().filter(|&&x| x > 0.0).count();
    }, n, 4));

    // --- NEON SIMD ---
    suite.add(bench_fn("map_reduce_square_sum", "fused", "neon_simd", || {
        let _ = neon::l2_squared_f32(&data);
    }, n, 4));

    suite.add(bench_fn("map_reduce_masked_sum", "fused", "neon_simd", || {
        let _ = neon::dot_f32(&data, &mask);
    }, n, 8));

    // --- Metal GPU ---
    let buf_data = ctx.buffer_from_slice(&data);
    let buf_mask = ctx.buffer_from_slice(&mask);
    let tg_size = 256;
    let n_groups = (n + tg_size - 1) / tg_size;
    let buf_out = ctx.buffer_empty(n_groups * 4);

    let pso_sq = ctx.pipeline("map_reduce_square_sum", shaders::FUSED).clone();
    suite.add(bench_fn("map_reduce_square_sum", "fused", "metal", || {
        ctx.dispatch_reduce(&pso_sq, &[&buf_data, &buf_out], n, tg_size * 4);
    }, n, 4));

    let pso_abs = ctx.pipeline("map_reduce_abs_sum", shaders::FUSED).clone();
    suite.add(bench_fn("map_reduce_abs_sum", "fused", "metal", || {
        ctx.dispatch_reduce(&pso_abs, &[&buf_data, &buf_out], n, tg_size * 4);
    }, n, 4));

    let pso_masked = ctx.pipeline("map_reduce_masked_sum", shaders::FUSED).clone();
    suite.add(bench_fn("map_reduce_masked_sum", "fused", "metal", || {
        ctx.dispatch_reduce(&pso_masked, &[&buf_data, &buf_mask, &buf_out], n, tg_size * 4);
    }, n, 8));

    let buf_out_u32 = ctx.buffer_empty(n_groups * 4);
    let buf_threshold = ctx.buffer_from_slice(&[0.0f32]);
    let pso_thresh = ctx.pipeline("map_reduce_threshold_count", shaders::FUSED).clone();
    suite.add(bench_fn("map_reduce_threshold_count", "fused", "metal", || {
        ctx.dispatch_reduce(&pso_thresh, &[&buf_data, &buf_out_u32, &buf_threshold], n, tg_size * 4);
    }, n, 4));
}
