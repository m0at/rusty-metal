//! Simulation & physics benchmarks: heat eq, wave eq, RK4, Verlet, Monte Carlo.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, _n: usize) {
    let mut rng = rand::thread_rng();

    // --- 2D stencils ---
    let width = 1024;
    let total = width * width;
    let grid: Vec<f32> = (0..total).map(|_| rng.gen_range(0.0..1.0)).collect();
    let alpha = 0.1f32;

    // Heat equation
    suite.add(bench_fn("sim_heat_equation", "simulation", "rust_scalar", || {
        let mut next = vec![0.0f32; total];
        for y in 1..width - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let center = grid[idx];
                let left = grid[idx - 1];
                let right = grid[idx + 1];
                let up = grid[idx - width];
                let down = grid[idx + width];
                next[idx] = center + alpha * (left + right + up + down - 4.0 * center);
            }
        }
        let _ = next;
    }, total, 4));

    // Wave equation
    let grid_prev: Vec<f32> = (0..total).map(|_| rng.gen_range(0.0..1.0)).collect();
    let c2 = 0.01f32;
    suite.add(bench_fn("sim_wave_equation", "simulation", "rust_scalar", || {
        let mut next = vec![0.0f32; total];
        for y in 1..width - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let laplacian = grid[idx - 1] + grid[idx + 1] + grid[idx - width] + grid[idx + width] - 4.0 * grid[idx];
                next[idx] = 2.0 * grid[idx] - grid_prev[idx] + c2 * laplacian;
            }
        }
        let _ = next;
    }, total, 4));

    // --- 1D ODE ---
    let ode_n = 1_000_000;
    let y: Vec<f32> = (0..ode_n).map(|_| rng.gen_range(0.1..2.0)).collect();
    let dt = 0.001f32;

    // RK4
    suite.add(bench_fn("integrate_rk4", "simulation", "rust_scalar", || {
        let _: Vec<f32> = y.iter().map(|&yi| {
            let k1 = -yi;
            let k2 = -(yi + 0.5 * dt * k1);
            let k3 = -(yi + 0.5 * dt * k2);
            let k4 = -(yi + dt * k3);
            yi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        }).collect();
    }, ode_n, 4));

    // Verlet
    let vel: Vec<f32> = (0..ode_n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let acc: Vec<f32> = y.iter().map(|&p| -p).collect(); // simple harmonic
    suite.add(bench_fn("integrate_verlet", "simulation", "rust_scalar", || {
        let mut pos = y.clone();
        let mut v = vel.clone();
        for i in 0..ode_n {
            let a = acc[i];
            pos[i] += v[i] * dt + 0.5 * a * dt * dt;
            let a_new = -pos[i];
            v[i] += 0.5 * (a + a_new) * dt;
        }
        let _ = (pos, v);
    }, ode_n, 12));

    // Monte Carlo (pi estimation)
    let random_x: Vec<f32> = (0..ode_n).map(|_| rng.gen_range(0.0..1.0)).collect();
    suite.add(bench_fn("monte_carlo_integrate", "simulation", "rust_scalar", || {
        let _: Vec<f32> = random_x.iter().map(|&x| 4.0 / (1.0 + x * x)).collect();
    }, ode_n, 4));

    // --- Metal GPU ---
    let buf_grid = ctx.buffer_from_slice(&grid);
    let buf_next = ctx.buffer_empty(total * 4);
    let buf_width = ctx.buffer_from_slice(&[width as u32]);
    let buf_alpha = ctx.buffer_from_slice(&[alpha]);

    let pso_heat = ctx.pipeline("sim_heat_equation_f32", shaders::SIMULATION).clone();
    suite.add(bench_fn("sim_heat_equation", "simulation", "metal", || {
        ctx.dispatch_2d(&pso_heat, &[&buf_grid, &buf_next, &buf_width, &buf_alpha], width, width);
    }, total, 4));

    let buf_prev = ctx.buffer_from_slice(&grid_prev);
    let buf_wave_next = ctx.buffer_empty(total * 4);
    let buf_c2 = ctx.buffer_from_slice(&[c2]);
    let pso_wave = ctx.pipeline("sim_wave_equation_f32", shaders::SIMULATION).clone();
    suite.add(bench_fn("sim_wave_equation", "simulation", "metal", || {
        ctx.dispatch_2d(&pso_wave, &[&buf_grid, &buf_prev, &buf_wave_next, &buf_width, &buf_c2], width, width);
    }, total, 4));

    let buf_y = ctx.buffer_from_slice(&y);
    let buf_y_next = ctx.buffer_empty(ode_n * 4);
    let buf_dt = ctx.buffer_from_slice(&[dt]);
    let pso_rk4 = ctx.pipeline("integrate_rk4_f32", shaders::SIMULATION).clone();
    suite.add(bench_fn("integrate_rk4", "simulation", "metal", || {
        ctx.dispatch_1d(&pso_rk4, &[&buf_y, &buf_y_next, &buf_dt], ode_n);
    }, ode_n, 4));

    let buf_pos = ctx.buffer_from_slice(&y);
    let buf_vel = ctx.buffer_from_slice(&vel);
    let buf_acc = ctx.buffer_from_slice(&acc);
    let pso_verlet = ctx.pipeline("integrate_velocity_verlet_f32", shaders::SIMULATION).clone();
    suite.add(bench_fn("integrate_verlet", "simulation", "metal", || {
        ctx.dispatch_1d(&pso_verlet, &[&buf_pos, &buf_vel, &buf_acc, &buf_dt], ode_n);
    }, ode_n, 12));

    let buf_rx = ctx.buffer_from_slice(&random_x);
    let buf_mc_out = ctx.buffer_empty(ode_n * 4);
    let pso_mc = ctx.pipeline("monte_carlo_integrate_f32", shaders::SIMULATION).clone();
    suite.add(bench_fn("monte_carlo_integrate", "simulation", "metal", || {
        ctx.dispatch_1d(&pso_mc, &[&buf_rx, &buf_mc_out], ode_n);
    }, ode_n, 4));
}
