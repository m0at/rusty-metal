//! Layout transform benchmarks: transpose 2D, AoS-to-SoA, SoA-to-AoS.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, _n: usize) {
    let mut rng = rand::thread_rng();
    let rows = 4096;
    let cols = 4096;
    let total = rows * cols;

    // --- Transpose ---
    let mat: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();

    suite.add(bench_fn("transpose_2d", "layout", "rust_scalar", || {
        let mut out = vec![0.0f32; total];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = mat[r * cols + c];
            }
        }
        black_box(out);
    }, total, 4));

    // --- AoS to SoA ---
    let n_structs = total / 4;
    let aos: Vec<[f32; 4]> = (0..n_structs).map(|_| {
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()]
    }).collect();

    suite.add(bench_fn("repack_aos_to_soa", "layout", "rust_scalar", || {
        let mut x = vec![0.0f32; n_structs];
        let mut y = vec![0.0f32; n_structs];
        let mut z = vec![0.0f32; n_structs];
        let mut w = vec![0.0f32; n_structs];
        for (i, v) in aos.iter().enumerate() {
            x[i] = v[0]; y[i] = v[1]; z[i] = v[2]; w[i] = v[3];
        }
        black_box((x, y, z, w));
    }, n_structs * 4, 4));

    // --- SoA to AoS ---
    let soa_x: Vec<f32> = (0..n_structs).map(|_| rng.gen()).collect();
    let soa_y: Vec<f32> = (0..n_structs).map(|_| rng.gen()).collect();
    let soa_z: Vec<f32> = (0..n_structs).map(|_| rng.gen()).collect();
    let soa_w: Vec<f32> = (0..n_structs).map(|_| rng.gen()).collect();

    suite.add(bench_fn("repack_soa_to_aos", "layout", "rust_scalar", || {
        let mut out = vec![[0.0f32; 4]; n_structs];
        for i in 0..n_structs {
            out[i] = [soa_x[i], soa_y[i], soa_z[i], soa_w[i]];
        }
        black_box(out);
    }, n_structs * 4, 4));

    // --- Metal GPU ---
    let buf_mat = ctx.buffer_from_slice(&mat);
    let buf_out = ctx.buffer_empty(total * 4);
    let buf_rows = ctx.buffer_from_slice(&[rows as u32]);
    let buf_cols = ctx.buffer_from_slice(&[cols as u32]);

    let pso_transpose = ctx.pipeline("transpose_2d", shaders::LAYOUT).clone();
    suite.add(bench_fn("transpose_2d", "layout", "metal", || {
        ctx.dispatch_2d(&pso_transpose, &[&buf_mat, &buf_out, &buf_rows, &buf_cols], rows, cols);
    }, total, 4));

    // AoS -> SoA on Metal
    let aos_flat: Vec<f32> = aos.iter().flat_map(|v| v.iter().cloned()).collect();
    let buf_aos = ctx.buffer_from_slice(&aos_flat);
    let buf_sx = ctx.buffer_empty(n_structs * 4);
    let buf_sy = ctx.buffer_empty(n_structs * 4);
    let buf_sz = ctx.buffer_empty(n_structs * 4);
    let buf_sw = ctx.buffer_empty(n_structs * 4);

    let pso_a2s = ctx.pipeline("repack_aos_to_soa", shaders::LAYOUT).clone();
    suite.add(bench_fn("repack_aos_to_soa", "layout", "metal", || {
        ctx.dispatch_1d(&pso_a2s, &[&buf_aos, &buf_sx, &buf_sy, &buf_sz, &buf_sw], n_structs);
    }, n_structs * 4, 4));

    // SoA -> AoS on Metal
    let buf_soa_x = ctx.buffer_from_slice(&soa_x);
    let buf_soa_y = ctx.buffer_from_slice(&soa_y);
    let buf_soa_z = ctx.buffer_from_slice(&soa_z);
    let buf_soa_w = ctx.buffer_from_slice(&soa_w);
    let buf_aos_out = ctx.buffer_empty(n_structs * 16);

    let pso_s2a = ctx.pipeline("repack_soa_to_aos", shaders::LAYOUT).clone();
    suite.add(bench_fn("repack_soa_to_aos", "layout", "metal", || {
        ctx.dispatch_1d(&pso_s2a, &[&buf_soa_x, &buf_soa_y, &buf_soa_z, &buf_soa_w, &buf_aos_out], n_structs);
    }, n_structs * 4, 4));
}
