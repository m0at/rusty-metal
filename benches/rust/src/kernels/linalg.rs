//! Linear algebra benchmarks: SpMV CSR, batched matmul, matvec, outer product.

use crate::harness::{bench_fn, BenchSuite};
use crate::metal_ctx::MetalCtx;
use crate::shaders;
use rand::Rng;
use std::hint::black_box;

pub fn run(suite: &mut BenchSuite, ctx: &mut MetalCtx, _n: usize) {
    let mut rng = rand::thread_rng();

    // --- Matvec ---
    let rows = 4096;
    let cols = 4096;
    let total = rows * cols;
    let mat: Vec<f32> = (0..total).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let vec_x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();

    suite.add(bench_fn("matvec", "linalg", "rust_scalar", || {
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += mat[r * cols + c] * vec_x[c];
            }
            y[r] = sum;
        }
        black_box(y);
    }, total, 4));

    // --- Outer product ---
    let op_n = 2048;
    let a: Vec<f32> = (0..op_n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..op_n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    suite.add(bench_fn("outer_product", "linalg", "rust_scalar", || {
        let mut out = vec![0.0f32; op_n * op_n];
        for i in 0..op_n {
            for j in 0..op_n {
                out[i * op_n + j] = a[i] * b[j];
            }
        }
        black_box(out);
    }, op_n * op_n, 4));

    // --- Batched matmul (small) ---
    let m = 128;
    let k = 128;
    let nn = 128;
    let mat_a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mat_b: Vec<f32> = (0..k * nn).map(|_| rng.gen_range(-1.0..1.0)).collect();

    suite.add(bench_fn("matmul_batched", "linalg", "rust_scalar", || {
        let mut c = vec![0.0f32; m * nn];
        for i in 0..m {
            for j in 0..nn {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += mat_a[i * k + kk] * mat_b[kk * nn + j];
                }
                c[i * nn + j] = sum;
            }
        }
        black_box(c);
    }, m * nn * k, 4));

    // --- SpMV CSR ---
    let spmv_rows = 4096;
    let spmv_cols = 4096;
    let nnz_per_row = 32;
    let nnz = spmv_rows * nnz_per_row;
    let mut row_ptr = vec![0u32; spmv_rows + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    for r in 0..spmv_rows {
        row_ptr[r] = (r * nnz_per_row) as u32;
        for j in 0..nnz_per_row {
            col_idx.push(((r * 7 + j * 131) % spmv_cols) as u32);
            values.push(rng.gen_range(-1.0..1.0f32));
        }
    }
    row_ptr[spmv_rows] = nnz as u32;
    let spmv_x: Vec<f32> = (0..spmv_cols).map(|_| rng.gen_range(-1.0..1.0)).collect();

    suite.add(bench_fn("spmv_csr", "linalg", "rust_scalar", || {
        let mut y = vec![0.0f32; spmv_rows];
        for r in 0..spmv_rows {
            let start = row_ptr[r] as usize;
            let end = row_ptr[r + 1] as usize;
            let mut sum = 0.0f32;
            for j in start..end {
                sum += values[j] * spmv_x[col_idx[j] as usize];
            }
            y[r] = sum;
        }
        black_box(y);
    }, nnz, 4));

    // --- Metal GPU ---
    let buf_mat = ctx.buffer_from_slice(&mat);
    let buf_x = ctx.buffer_from_slice(&vec_x);
    let buf_y = ctx.buffer_empty(rows * 4);
    let buf_cols = ctx.buffer_from_slice(&[cols as u32]);

    let pso_matvec = ctx.pipeline("matvec_f32", shaders::LINALG).clone();
    suite.add(bench_fn("matvec", "linalg", "metal", || {
        ctx.dispatch_1d(&pso_matvec, &[&buf_mat, &buf_x, &buf_y, &buf_cols], rows);
    }, total, 4));

    let buf_a = ctx.buffer_from_slice(&a);
    let buf_b = ctx.buffer_from_slice(&b);
    let buf_outer = ctx.buffer_empty(op_n * op_n * 4);
    let buf_op_n = ctx.buffer_from_slice(&[op_n as u32]);
    let pso_outer = ctx.pipeline("outer_product_f32", shaders::LINALG).clone();
    suite.add(bench_fn("outer_product", "linalg", "metal", || {
        ctx.dispatch_2d(&pso_outer, &[&buf_a, &buf_b, &buf_outer, &buf_op_n], op_n, op_n);
    }, op_n * op_n, 4));

    let buf_spmv_vals = ctx.buffer_from_slice(&values);
    let buf_spmv_cols = ctx.buffer_from_slice(&col_idx);
    let buf_spmv_rows = ctx.buffer_from_slice(&row_ptr);
    let buf_spmv_x = ctx.buffer_from_slice(&spmv_x);
    let buf_spmv_y = ctx.buffer_empty(spmv_rows * 4);
    let pso_spmv = ctx.pipeline("spmv_csr_f32", shaders::LINALG).clone();
    suite.add(bench_fn("spmv_csr", "linalg", "metal", || {
        ctx.dispatch_1d(&pso_spmv, &[&buf_spmv_vals, &buf_spmv_cols, &buf_spmv_rows, &buf_spmv_x, &buf_spmv_y], spmv_rows);
    }, nnz, 4));
}
