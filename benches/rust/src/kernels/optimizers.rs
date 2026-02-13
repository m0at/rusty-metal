//! Optimizer benchmarks: SGD, SGD+momentum, Adam, AdamW, grad clip.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::neon;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, n: usize) {
    let mut rng = rand::thread_rng();
    let params: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let grads: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();
    let lr = 0.001f32;

    // --- Scalar Rust ---
    suite.add(bench_fn("opt_sgd", "optimizers", "rust_scalar", || {
        let mut p = params.clone();
        for i in 0..n { p[i] -= lr * grads[i]; }
        let _ = p;
    }, n, 8));

    suite.add(bench_fn("opt_sgd_momentum", "optimizers", "rust_scalar", || {
        let mut p = params.clone();
        let mut vel = vec![0.0f32; n];
        let momentum = 0.9f32;
        for i in 0..n {
            vel[i] = momentum * vel[i] + grads[i];
            p[i] -= lr * vel[i];
        }
        let _ = p;
    }, n, 12));

    suite.add(bench_fn("opt_adam", "optimizers", "rust_scalar", || {
        let mut p = params.clone();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            p[i] -= lr * m[i] / (v[i].sqrt() + eps);
        }
        let _ = p;
    }, n, 20));

    suite.add(bench_fn("opt_adamw", "optimizers", "rust_scalar", || {
        let mut p = params.clone();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let wd = 0.01f32;
        for i in 0..n {
            p[i] -= lr * wd * p[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            p[i] -= lr * m[i] / (v[i].sqrt() + eps);
        }
        let _ = p;
    }, n, 20));

    suite.add(bench_fn("grad_clip_value", "optimizers", "rust_scalar", || {
        let _: Vec<f32> = grads.iter().map(|&g| g.clamp(-1.0, 1.0)).collect();
    }, n, 4));

    suite.add(bench_fn("grad_clip_norm", "optimizers", "rust_scalar", || {
        let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        let max_norm = 1.0f32;
        if norm > max_norm {
            let _: Vec<f32> = grads.iter().map(|&g| g * max_norm / norm).collect();
        }
    }, n, 4));

    // --- NEON SIMD (SGD) ---
    suite.add(bench_fn("opt_sgd", "optimizers", "neon_simd", || {
        let mut p = params.clone();
        neon::sgd_f32(&mut p, &grads, lr);
        let _ = p;
    }, n, 8));

    // --- Metal GPU ---
    let buf_params = ctx.buffer_from_slice(&params);
    let buf_grads = ctx.buffer_from_slice(&grads);
    let buf_lr = ctx.buffer_from_slice(&[lr]);

    let pso_sgd = ctx.pipeline("opt_sgd_f32", shaders::OPTIMIZERS).clone();
    suite.add(bench_fn("opt_sgd", "optimizers", "metal", || {
        ctx.dispatch_1d(&pso_sgd, &[&buf_params, &buf_grads, &buf_lr], n);
    }, n, 8));

    let buf_vel = ctx.buffer_empty(n * 4);
    let buf_momentum = ctx.buffer_from_slice(&[0.9f32]);
    let pso_sgdm = ctx.pipeline("opt_sgd_momentum_f32", shaders::OPTIMIZERS).clone();
    suite.add(bench_fn("opt_sgd_momentum", "optimizers", "metal", || {
        ctx.dispatch_1d(&pso_sgdm, &[&buf_params, &buf_grads, &buf_vel, &buf_lr, &buf_momentum], n);
    }, n, 12));

    let buf_m = ctx.buffer_empty(n * 4);
    let buf_v = ctx.buffer_empty(n * 4);
    let buf_beta1 = ctx.buffer_from_slice(&[0.9f32]);
    let buf_beta2 = ctx.buffer_from_slice(&[0.999f32]);
    let buf_eps = ctx.buffer_from_slice(&[1e-8f32]);
    let pso_adam = ctx.pipeline("opt_adam_f32", shaders::OPTIMIZERS).clone();
    suite.add(bench_fn("opt_adam", "optimizers", "metal", || {
        ctx.dispatch_1d(&pso_adam, &[&buf_params, &buf_grads, &buf_m, &buf_v, &buf_lr, &buf_beta1, &buf_beta2, &buf_eps], n);
    }, n, 20));

    let buf_clip = ctx.buffer_from_slice(&[1.0f32]);
    let pso_clip_val = ctx.pipeline("grad_clip_by_value_f32", shaders::OPTIMIZERS).clone();
    suite.add(bench_fn("grad_clip_value", "optimizers", "metal", || {
        ctx.dispatch_1d(&pso_clip_val, &[&buf_grads, &buf_clip], n);
    }, n, 4));
}
