//! Signal processing benchmarks: conv1d, window apply, FIR filter.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let signal: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let kernel_size = 32;
    let kernel_data: Vec<f32> = (0..kernel_size).map(|_| rng.gen_range(-0.5..0.5)).collect();

    // --- Scalar Rust: conv1d ---
    suite.add(bench_fn("conv1d", "signal", "rust_scalar", || {
        let mut out = vec![0.0f32; n];
        let half = kernel_size / 2;
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let idx = i as isize - half as isize + k as isize;
                if idx >= 0 && (idx as usize) < n {
                    sum += signal[idx as usize] * kernel_data[k];
                }
            }
            out[i] = sum;
        }
        let _ = out;
    }, n, 4));

    // --- Scalar Rust: window apply (Hanning) ---
    suite.add(bench_fn("window_apply", "signal", "rust_scalar", || {
        let _: Vec<f32> = signal.iter().enumerate().map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            x * w
        }).collect();
    }, n, 4));

    // --- Scalar Rust: FIR filter ---
    suite.add(bench_fn("fir_filter", "signal", "rust_scalar", || {
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for t in 0..kernel_size {
                if i >= t {
                    sum += signal[i - t] * kernel_data[t];
                }
            }
            out[i] = sum;
        }
        let _ = out;
    }, n, 4));

    // --- Metal GPU ---
    let buf_signal = ctx.buffer_from_slice(&signal);
    let buf_kernel = ctx.buffer_from_slice(&kernel_data);
    let buf_out = ctx.buffer_empty(n * 4);
    let buf_n = ctx.buffer_from_slice(&[n as u32]);
    let buf_ksize = ctx.buffer_from_slice(&[kernel_size as u32]);

    let pso_conv = ctx.pipeline("conv1d_f32", shaders::SIGNAL).clone();
    suite.add(bench_fn("conv1d", "signal", "metal", || {
        ctx.dispatch_1d(&pso_conv, &[&buf_signal, &buf_kernel, &buf_out, &buf_n, &buf_ksize], n);
    }, n, 4));

    let pso_window = ctx.pipeline("window_apply_f32", shaders::SIGNAL).clone();
    suite.add(bench_fn("window_apply", "signal", "metal", || {
        ctx.dispatch_1d(&pso_window, &[&buf_signal, &buf_out, &buf_n], n);
    }, n, 4));

    let pso_fir = ctx.pipeline("fir_filter_f32", shaders::SIGNAL).clone();
    suite.add(bench_fn("fir_filter", "signal", "metal", || {
        ctx.dispatch_1d(&pso_fir, &[&buf_signal, &buf_kernel, &buf_out, &buf_n, &buf_ksize], n);
    }, n, 4));
}
