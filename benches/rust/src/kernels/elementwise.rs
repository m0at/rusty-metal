//! Elementwise benchmarks: exp, log, sigmoid, tanh, softplus, clamp, abs, add, mul, div, fma, compare.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.gen_range(0.1..2.0)).collect();
    let c: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut out = vec![0.0f32; n];

    // --- Scalar Rust (unary) ---
    for (name, op) in [
        ("map_exp", (|x: f32| x.exp()) as fn(f32) -> f32),
        ("map_log", |x: f32| x.abs().max(1e-7).ln()),
        ("map_sigmoid", |x: f32| 1.0 / (1.0 + (-x).exp())),
        ("map_tanh", |x: f32| x.tanh()),
        ("map_softplus", |x: f32| (1.0 + x.exp()).ln()),
    ] {
        let data = a.clone();
        suite.add(bench_fn(name, "elementwise", "rust_scalar", || {
            black_box::<Vec<f32>>(data.iter().map(|&x| op(x)).collect());
        }, n, 4));
    }

    // --- Scalar Rust (clamp, abs) ---
    suite.add(bench_fn("map_clamp", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().map(|&x| x.clamp(-1.0, 1.0)).collect());
    }, n, 4));

    suite.add(bench_fn("map_abs", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().map(|&x| x.abs()).collect());
    }, n, 4));

    // --- Scalar Rust (binary) ---
    suite.add(bench_fn("map_add", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().zip(&b).map(|(x, y)| x + y).collect());
    }, n, 8));

    suite.add(bench_fn("map_mul", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().zip(&b).map(|(x, y)| x * y).collect());
    }, n, 8));

    suite.add(bench_fn("map_div", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().zip(&b).map(|(x, y)| x / y).collect());
    }, n, 8));

    suite.add(bench_fn("map_fma", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().zip(&b).zip(&c).map(|((x, y), z)| x * y + z).collect());
    }, n, 12));

    suite.add(bench_fn("map_compare", "elementwise", "rust_scalar", || {
        black_box::<Vec<f32>>(a.iter().zip(&b).map(|(x, y)| if x > y { 1.0 } else { 0.0 }).collect());
    }, n, 8));

    // --- NEON SIMD ---
    suite.add(bench_fn("map_abs", "elementwise", "neon_simd", || {
        neon::abs_f32(&a, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_clamp", "elementwise", "neon_simd", || {
        neon::clamp_f32(&a, -1.0, 1.0, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_add", "elementwise", "neon_simd", || {
        neon::add_f32(&a, &b, &mut out);
    }, n, 8));

    suite.add(bench_fn("map_mul", "elementwise", "neon_simd", || {
        neon::mul_f32(&a, &b, &mut out);
    }, n, 8));

    suite.add(bench_fn("map_fma", "elementwise", "neon_simd", || {
        neon::fma_f32(&a, &b, &c, &mut out);
    }, n, 12));

    suite.add(bench_fn("map_exp", "elementwise", "neon_simd", || {
        neon::exp_f32(&a, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_log", "elementwise", "neon_simd", || {
        neon::log_f32(&b, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_sigmoid", "elementwise", "neon_simd", || {
        neon::sigmoid_f32(&a, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_tanh", "elementwise", "neon_simd", || {
        neon::tanh_f32(&a, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_softplus", "elementwise", "neon_simd", || {
        neon::softplus_f32(&a, &mut out);
    }, n, 4));

    suite.add(bench_fn("map_div", "elementwise", "neon_simd", || {
        neon::div_f32(&a, &b, &mut out);
    }, n, 8));

    suite.add(bench_fn("map_compare", "elementwise", "neon_simd", || {
        neon::compare_gt_f32(&a, &b, &mut out);
    }, n, 8));

    // --- Metal GPU ---
    let buf_a = ctx.buffer_from_slice(&a);
    let buf_b = ctx.buffer_from_slice(&b);
    let buf_c = ctx.buffer_from_slice(&c);
    let buf_out = ctx.buffer_empty(n * 4);

    for name in ["map_exp", "map_log", "map_sigmoid", "map_tanh", "map_softplus", "map_clamp", "map_abs"] {
        let pso = ctx.pipeline(name, shaders::ELEMENTWISE).clone();
        suite.add(bench_fn(name, "elementwise", "metal", || {
            ctx.dispatch_1d(&pso, &[&buf_a, &buf_out], n);
        }, n, 4));
    }

    for name in ["map_add", "map_mul", "map_div", "map_compare"] {
        let pso = ctx.pipeline(name, shaders::ELEMENTWISE).clone();
        suite.add(bench_fn(name, "elementwise", "metal", || {
            ctx.dispatch_1d(&pso, &[&buf_a, &buf_b, &buf_out], n);
        }, n, 8));
    }

    let pso_fma = ctx.pipeline("map_fma", shaders::ELEMENTWISE).clone();
    suite.add(bench_fn("map_fma", "elementwise", "metal", || {
        ctx.dispatch_1d(&pso_fma, &[&buf_a, &buf_b, &buf_c, &buf_out], n);
    }, n, 12));
}
