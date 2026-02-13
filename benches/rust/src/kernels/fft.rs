//! FFT benchmarks: radix-2, radix-4, IFFT, spectral power.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;
use std::f32::consts::PI;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, _n: usize) {
    let n = 1 << 20; // 1M elements, power of 2 for FFT
    let mut rng = rand::thread_rng();
    // Complex data as [re, im] pairs (interleaved as f32x2)
    let data: Vec<[f32; 2]> = (0..n).map(|_| [rng.gen_range(-1.0..1.0), 0.0]).collect();

    // --- Scalar Rust: radix-2 FFT ---
    suite.add(bench_fn("fft_radix2", "fft", "rust_scalar", || {
        let mut buf: Vec<[f32; 2]> = data.clone();
        // Bit-reversal permutation
        let bits = (n as f32).log2() as usize;
        for i in 0..n {
            let j = bit_reverse(i, bits);
            if i < j { buf.swap(i, j); }
        }
        // Butterfly stages
        let mut half = 1;
        while half < n {
            let full = half * 2;
            for group in (0..n).step_by(full) {
                for pair in 0..half {
                    let angle = -2.0 * PI * pair as f32 / full as f32;
                    let w = [angle.cos(), angle.sin()];
                    let i = group + pair;
                    let j = i + half;
                    let a = buf[i];
                    let b = buf[j];
                    let wb = [w[0]*b[0]-w[1]*b[1], w[0]*b[1]+w[1]*b[0]];
                    buf[i] = [a[0]+wb[0], a[1]+wb[1]];
                    buf[j] = [a[0]-wb[0], a[1]-wb[1]];
                }
            }
            half *= 2;
        }
        black_box(buf);
    }, n, 8));

    // --- Scalar Rust: spectral power ---
    suite.add(bench_fn("spectral_power", "fft", "rust_scalar", || {
        black_box::<Vec<f32>>(data.iter().map(|c| c[0]*c[0] + c[1]*c[1]).collect());
    }, n, 8));

    // --- Metal GPU: FFT radix-2 ---
    let flat: Vec<f32> = data.iter().flat_map(|c| c.iter().cloned()).collect();
    let buf_data = ctx.buffer_from_slice(&flat);
    let buf_n = ctx.buffer_from_slice(&[n as u32]);

    let pso_r2 = ctx.pipeline("fft_radix2_f32", shaders::FFT).clone();
    let stages = (n as f32).log2() as u32;
    suite.add(bench_fn("fft_radix2", "fft", "metal", || {
        for stage in 0..stages {
            let buf_stage = ctx.buffer_from_slice(&[stage]);
            ctx.dispatch_1d(&pso_r2, &[&buf_data, &buf_n, &buf_stage], n / 2);
        }
    }, n, 8));

    // --- Metal GPU: spectral power ---
    let buf_power_out = ctx.buffer_empty(n * 4);
    let pso_power = ctx.pipeline("spectral_power_density_f32", shaders::FFT).clone();
    suite.add(bench_fn("spectral_power", "fft", "metal", || {
        ctx.dispatch_1d(&pso_power, &[&buf_data, &buf_power_out], n);
    }, n, 8));
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
