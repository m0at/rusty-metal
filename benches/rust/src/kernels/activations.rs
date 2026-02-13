//! Activation benchmarks: relu, leaky_relu, elu, gelu, silu, mish.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
    let mut out = vec![0.0f32; n];

    // --- Scalar Rust ---
    suite.add(bench_fn("map_relu", "activations", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|&x| x.max(0.0)).collect());
    }, n, 4));

    suite.add(bench_fn("map_leaky_relu", "activations", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|&x| if x > 0.0 { x } else { 0.01 * x }).collect());
    }, n, 4));

    suite.add(bench_fn("map_elu", "activations", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect());
    }, n, 4));

    suite.add(bench_fn("map_gelu", "activations", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|&x| {
            0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x) as f64).tanh() as f32)
        }).collect());
    }, n, 4));

    suite.add(bench_fn("map_silu", "activations", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|&x| x / (1.0 + (-x).exp())).collect());
    }, n, 4));

    suite.add(bench_fn("map_mish", "activations", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|&x| x * (1.0f32 + x.exp()).ln().tanh()).collect());
    }, n, 4));

    // --- NEON SIMD ---
    suite.add(bench_fn("map_relu", "activations", "neon_simd", || {
        neon::relu_f32(&data, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_leaky_relu", "activations", "neon_simd", || {
        neon::leaky_relu_f32(&data, 0.01, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_gelu", "activations", "neon_simd", || {
        neon::gelu_f32(&data, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_silu", "activations", "neon_simd", || {
        neon::silu_f32(&data, &mut out);
    }, n, 4));

    // --- Metal GPU ---
    let buf_in = ctx.buffer_from_slice(&data);
    let buf_out = ctx.buffer_empty(n * 4);

    for name in ["map_relu", "map_leaky_relu", "map_elu", "map_gelu", "map_silu", "map_mish"] {
        let pso = ctx.pipeline(name, shaders::ACTIVATIONS).clone();
        suite.add(bench_fn(name, "activations", "metal", || {
            ctx.dispatch_1d(&pso, &[&buf_in, &buf_out], n);
        }, n, 4));
    }
}
