//! Softmax benchmarks: stable softmax, log-softmax, online softmax.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let rows = 4096;
    let cols = n.min(4096);
    let total = rows * cols;
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..total).map(|_| rng.gen_range(-2.0..2.0)).collect();

    // --- Scalar Rust ---
    suite.add(bench_fn("softmax_stable", "softmax", "rust_scalar", || {
        let mut out = vec![0.0f32; total];
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            for (i, &x) in row.iter().enumerate() {
                out[r * cols + i] = (x - max).exp() / sum;
            }
        }
        let _ = out;
    }, total, 4));

    suite.add(bench_fn("log_softmax", "softmax", "rust_scalar", || {
        let mut out = vec![0.0f32; total];
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            let log_sum = sum.ln();
            for (i, &x) in row.iter().enumerate() {
                out[r * cols + i] = (x - max) - log_sum;
            }
        }
        let _ = out;
    }, total, 4));

    suite.add(bench_fn("softmax_online", "softmax", "rust_scalar", || {
        let mut out = vec![0.0f32; total];
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let mut max = f32::MIN;
            let mut sum = 0.0f32;
            for &x in row {
                if x > max {
                    sum = sum * (max - x).exp() + 1.0;
                    max = x;
                } else {
                    sum += (x - max).exp();
                }
            }
            for (i, &x) in row.iter().enumerate() {
                out[r * cols + i] = (x - max).exp() / sum;
            }
        }
        let _ = out;
    }, total, 4));

    // --- Metal GPU ---
    let buf_in = ctx.buffer_from_slice(&data);
    let buf_out = ctx.buffer_empty(total * 4);
    let buf_row_len = ctx.buffer_from_slice(&[cols as u32]);

    let pso = ctx.pipeline("softmax_stable_f32", shaders::SOFTMAX).clone();
    suite.add(bench_fn("softmax_stable", "softmax", "metal", || {
        ctx.dispatch_reduce(&pso, &[&buf_in, &buf_out, &buf_row_len], total, cols * 4);
    }, total, 4));

    let pso_log = ctx.pipeline("log_softmax_f32", shaders::SOFTMAX).clone();
    suite.add(bench_fn("log_softmax", "softmax", "metal", || {
        ctx.dispatch_reduce(&pso_log, &[&buf_in, &buf_out, &buf_row_len], total, cols * 4);
    }, total, 4));
}
