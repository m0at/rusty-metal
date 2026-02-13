//! Normalization benchmarks: LayerNorm, BatchNorm, RMSNorm, GroupNorm.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let dim = 1024;
    let batch = n / dim;
    let total = batch * dim;
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..total).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let gamma: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.5..1.5)).collect();
    let beta: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();

    // --- Scalar Rust ---
    suite.add(bench_fn("layer_norm", "normalization", "rust_scalar", || {
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            let row = &data[b * dim..(b + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let var: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for (i, &x) in row.iter().enumerate() {
                out[b * dim + i] = gamma[i] * (x - mean) * inv_std + beta[i];
            }
        }
        let _ = out;
    }, total, 4));

    suite.add(bench_fn("rms_norm", "normalization", "rust_scalar", || {
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            let row = &data[b * dim..(b + 1) * dim];
            let rms: f32 = (row.iter().map(|&x| x * x).sum::<f32>() / dim as f32 + 1e-5).sqrt();
            let inv_rms = 1.0 / rms;
            for (i, &x) in row.iter().enumerate() {
                out[b * dim + i] = gamma[i] * x * inv_rms;
            }
        }
        let _ = out;
    }, total, 4));

    suite.add(bench_fn("batch_norm", "normalization", "rust_scalar", || {
        let mean: f32 = data.iter().sum::<f32>() / total as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / total as f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let _: Vec<f32> = data.iter().map(|&x| gamma[0] * (x - mean) * inv_std + beta[0]).collect();
    }, total, 4));

    suite.add(bench_fn("group_norm", "normalization", "rust_scalar", || {
        let group_size = 256;
        let mut out = vec![0.0f32; total];
        for g in (0..total).step_by(group_size) {
            let end = (g + group_size).min(total);
            let group = &data[g..end];
            let len = group.len() as f32;
            let mean: f32 = group.iter().sum::<f32>() / len;
            let var: f32 = group.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / len;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for (i, &x) in group.iter().enumerate() {
                out[g + i] = gamma[0] * (x - mean) * inv_std + beta[0];
            }
        }
        let _ = out;
    }, total, 4));

    // --- Metal GPU ---
    let buf_in = ctx.buffer_from_slice(&data);
    let buf_gamma = ctx.buffer_from_slice(&gamma);
    let buf_beta = ctx.buffer_from_slice(&beta);
    let buf_out = ctx.buffer_empty(total * 4);
    let buf_dim = ctx.buffer_from_slice(&[dim as u32]);

    let pso_ln = ctx.pipeline("layer_norm_f32", shaders::NORMALIZATION).clone();
    suite.add(bench_fn("layer_norm", "normalization", "metal", || {
        ctx.dispatch_reduce(&pso_ln, &[&buf_in, &buf_gamma, &buf_beta, &buf_out, &buf_dim], total, dim * 4);
    }, total, 4));

    let pso_rms = ctx.pipeline("rms_norm_f32", shaders::NORMALIZATION).clone();
    suite.add(bench_fn("rms_norm", "normalization", "metal", || {
        ctx.dispatch_reduce(&pso_rms, &[&buf_in, &buf_gamma, &buf_out, &buf_dim], total, dim * 4);
    }, total, 4));
}
