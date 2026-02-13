//! Sorting benchmarks: bitonic sort, radix sort (CPU), top-k, median, percentile.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, _n: usize) {
    let mut rng = rand::thread_rng();

    // Power-of-2 for bitonic
    let n = 1 << 16; // 65536
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-100.0..100.0)).collect();

    // --- Scalar Rust: sort (std) ---
    suite.add(bench_fn("sort_radix", "sorting", "rust_scalar", || {
        let mut d = data.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let _ = d;
    }, n, 4));

    // --- Scalar Rust: top-k ---
    let k = 100;
    suite.add(bench_fn("topk_select", "sorting", "rust_scalar", || {
        let mut d = data.clone();
        d.select_nth_unstable_by(k, |a, b| b.partial_cmp(a).unwrap());
        let _ = &d[..k];
    }, n, 4));

    // --- Scalar Rust: median ---
    suite.add(bench_fn("median_select", "sorting", "rust_scalar", || {
        let mut d = data.clone();
        let mid = d.len() / 2;
        d.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        let _ = d[mid];
    }, n, 4));

    // --- Scalar Rust: percentile ---
    suite.add(bench_fn("percentile_select", "sorting", "rust_scalar", || {
        let mut d = data.clone();
        let idx = (d.len() as f64 * 0.95) as usize;
        d.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap());
        let _ = d[idx];
    }, n, 4));

    // --- Metal GPU: bitonic sort ---
    let buf_data = ctx.buffer_from_slice(&data);
    let pso_bitonic = ctx.pipeline("sort_bitonic_f32", shaders::SORTING).clone();
    let stages = (n as f32).log2() as u32;

    suite.add(bench_fn("sort_bitonic", "sorting", "metal", || {
        for stage in 0..stages {
            for substage in (0..=stage).rev() {
                let buf_stage = ctx.buffer_from_slice(&[stage]);
                let buf_sub = ctx.buffer_from_slice(&[substage]);
                ctx.dispatch_1d(&pso_bitonic, &[&buf_data, &buf_stage, &buf_sub], n / 2);
            }
        }
    }, n, 4));

    // --- Metal GPU: bitonic KV ---
    let values: Vec<u32> = (0..n as u32).collect();
    let buf_keys = ctx.buffer_from_slice(&data);
    let buf_vals = ctx.buffer_from_slice(&values);
    let pso_kv = ctx.pipeline("sort_bitonic_kv_f32", shaders::SORTING).clone();

    suite.add(bench_fn("sort_bitonic_kv", "sorting", "metal", || {
        for stage in 0..stages {
            for substage in (0..=stage).rev() {
                let buf_stage = ctx.buffer_from_slice(&[stage]);
                let buf_sub = ctx.buffer_from_slice(&[substage]);
                ctx.dispatch_1d(&pso_kv, &[&buf_keys, &buf_vals, &buf_stage, &buf_sub], n / 2);
            }
        }
    }, n, 8));
}
