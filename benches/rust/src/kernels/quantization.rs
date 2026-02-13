//! Quantization benchmarks: f32-to-f16, f16-to-i8, i8-to-f16.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let data_f32: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // --- Scalar Rust: f32 to f16 ---
    suite.add(bench_fn("quantize_f32_to_f16", "quantization", "rust_scalar", || {
        black_box::<Vec<u16>>(data_f32.iter().map(|&v| {
            let bits = v.to_bits();
            let sign = (bits >> 16) & 0x8000;
            let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
            let mantissa = bits & 0x7FFFFF;
            if exp <= 0 { sign as u16 }
            else if exp >= 31 { (sign | 0x7C00) as u16 }
            else { (sign | (exp as u32) << 10 | (mantissa >> 13)) as u16 }
        }).collect());
    }, n, 4));

    // --- Scalar Rust: f16 to i8 (per-tensor) ---
    let scale = data_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 127.0;
    let data_i8: Vec<i8> = data_f32.iter().map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8).collect();

    suite.add(bench_fn("quantize_f16_to_i8", "quantization", "rust_scalar", || {
        black_box::<Vec<i8>>(data_f32.iter().map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8).collect());
    }, n, 2));

    // --- Scalar Rust: i8 to f16 (dequantize) ---
    suite.add(bench_fn("dequantize_i8_to_f16", "quantization", "rust_scalar", || {
        black_box::<Vec<f32>>(data_i8.iter().map(|&v| v as f32 * scale).collect());
    }, n, 1));

    // --- NEON SIMD: f32 to f16 ---
    let mut out_f16 = vec![0u16; n];
    suite.add(bench_fn("quantize_f32_to_f16", "quantization", "neon_simd", || {
        neon::f32_to_f16(&data_f32, &mut out_f16);
    }, n, 4));

    // --- Metal GPU ---
    let buf_f32 = ctx.buffer_from_slice(&data_f32);
    let buf_f16_out = ctx.buffer_empty(n * 2);

    let pso_f32_f16 = ctx.pipeline("quantize_f32_to_f16", shaders::QUANTIZATION).clone();
    suite.add(bench_fn("quantize_f32_to_f16", "quantization", "metal", || {
        ctx.dispatch_1d(&pso_f32_f16, &[&buf_f32, &buf_f16_out], n);
    }, n, 4));
}
