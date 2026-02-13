//! PRNG benchmarks: Philox, uniform f32, normal f32, dropout mask.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();

    // --- Scalar Rust (using rand crate) ---
    suite.add(bench_fn("prng_philox", "prng", "rust_scalar", || {
        let _: Vec<[u32; 4]> = (0..n).map(|_| [rng.gen(), rng.gen(), rng.gen(), rng.gen()]).collect();
    }, n, 16));

    suite.add(bench_fn("prng_uniform_f32", "prng", "rust_scalar", || {
        let _: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
    }, n, 4));

    suite.add(bench_fn("prng_normal_f32", "prng", "rust_scalar", || {
        // Box-Muller
        let _: Vec<f32> = (0..n).map(|_| {
            let u1: f32 = rng.gen_range(1e-7..1.0);
            let u2: f32 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }).collect();
    }, n, 4));

    suite.add(bench_fn("prng_dropout_mask", "prng", "rust_scalar", || {
        let keep_prob = 0.9f32;
        let _: Vec<u8> = (0..n).map(|_| if rng.gen::<f32>() < keep_prob { 1 } else { 0 }).collect();
    }, n, 1));

    // --- Metal GPU ---
    let key = [42u32, 0u32];
    let buf_key = ctx.buffer_from_slice(&key);

    let buf_philox_out = ctx.buffer_empty(n * 16); // u32x4 per element
    let pso_philox = ctx.pipeline("prng_philox_u32x4", shaders::PRNG).clone();
    suite.add(bench_fn("prng_philox", "prng", "metal", || {
        ctx.dispatch_1d(&pso_philox, &[&buf_philox_out, &buf_key], n);
    }, n, 16));

    let buf_uniform_out = ctx.buffer_empty(n * 4);
    let pso_uniform = ctx.pipeline("prng_uniform_f32", shaders::PRNG).clone();
    suite.add(bench_fn("prng_uniform_f32", "prng", "metal", || {
        ctx.dispatch_1d(&pso_uniform, &[&buf_uniform_out, &buf_key], n);
    }, n, 4));

    let buf_normal_out = ctx.buffer_empty(n * 4);
    let pso_normal = ctx.pipeline("prng_normal_f32", shaders::PRNG).clone();
    suite.add(bench_fn("prng_normal_f32", "prng", "metal", || {
        ctx.dispatch_1d(&pso_normal, &[&buf_normal_out, &buf_key], n);
    }, n, 4));

    let buf_dropout_out = ctx.buffer_empty(n);
    let buf_keep = ctx.buffer_from_slice(&[0.9f32]);
    let pso_dropout = ctx.pipeline("prng_dropout_mask", shaders::PRNG).clone();
    suite.add(bench_fn("prng_dropout_mask", "prng", "metal", || {
        ctx.dispatch_1d(&pso_dropout, &[&buf_dropout_out, &buf_key, &buf_keep], n);
    }, n, 1));
}
