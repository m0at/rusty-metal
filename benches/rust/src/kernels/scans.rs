//! Scan & compaction benchmarks: exclusive scan, inclusive scan, compact_mask, compact_if.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let data_u32: Vec<u32> = (0..n).map(|_| rng.gen_range(0..1000)).collect();
    let data_f32: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mask: Vec<u32> = (0..n).map(|_| if rng.gen_bool(0.5) { 1 } else { 0 }).collect();

    // --- Scalar Rust: exclusive scan ---
    suite.add(bench_fn("scan_exclusive", "scans", "rust_scalar", || {
        let mut out = vec![0u32; n];
        let mut acc = 0u32;
        for i in 0..n {
            out[i] = acc;
            acc = acc.wrapping_add(data_u32[i]);
        }
        black_box(out);
    }, n, 4));

    // --- Scalar Rust: inclusive scan ---
    suite.add(bench_fn("scan_inclusive", "scans", "rust_scalar", || {
        let mut out = vec![0u32; n];
        let mut acc = 0u32;
        for i in 0..n {
            acc = acc.wrapping_add(data_u32[i]);
            out[i] = acc;
        }
        black_box(out);
    }, n, 4));

    // --- Scalar Rust: compact_mask ---
    suite.add(bench_fn("compact_mask", "scans", "rust_scalar", || {
        black_box::<Vec<f32>>(data_f32.iter().zip(&mask).filter(|(_, &m)| m == 1).map(|(&v, _)| v).collect());
    }, n, 4));

    // --- Scalar Rust: compact_if ---
    let threshold = 0.0f32;
    suite.add(bench_fn("compact_if", "scans", "rust_scalar", || {
        black_box::<Vec<f32>>(data_f32.iter().filter(|&&v| v > threshold).cloned().collect());
    }, n, 4));

    // --- Metal GPU (scans need power-of-2 threadgroup sizes) ---
    let tg_size = 256;
    let scan_n = (n / tg_size) * tg_size; // round down to multiple of tg_size
    let scan_data: Vec<u32> = data_u32[..scan_n].to_vec();
    let buf_scan = ctx.buffer_from_slice(&scan_data);

    let pso_excl = ctx.pipeline("scan_exclusive_u32", shaders::SCANS).clone();
    suite.add(bench_fn("scan_exclusive", "scans", "metal", || {
        ctx.dispatch_reduce(&pso_excl, &[&buf_scan], scan_n, tg_size * 4);
    }, scan_n, 4));

    let buf_scan_inc = ctx.buffer_from_slice(&scan_data);
    let pso_incl = ctx.pipeline("scan_inclusive_u32", shaders::SCANS).clone();
    suite.add(bench_fn("scan_inclusive", "scans", "metal", || {
        ctx.dispatch_reduce(&pso_incl, &[&buf_scan_inc], scan_n, tg_size * 4);
    }, scan_n, 4));
}
