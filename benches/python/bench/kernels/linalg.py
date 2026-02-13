"""Linear Algebra: SpMV CSR, batched matmul, matvec, triangular solve, outer product."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    # ── SpMV CSR ──
    rows, cols = 10000, 10000
    density = 0.01
    nnz = int(rows * cols * density)
    row_idx = np.sort(np.random.randint(0, rows, nnz))
    col_idx = np.random.randint(0, cols, nnz)
    vals = np.random.randn(nnz).astype(np.float32)
    vec = np.random.randn(cols).astype(np.float32)

    from scipy import sparse
    try:
        A_csr = sparse.csr_matrix((vals, (row_idx, col_idx)), shape=(rows, cols))
        bench_fn("spmv_csr", "linalg", "numpy",
                 lambda: A_csr.dot(vec), rows)
    except ImportError:
        pass  # scipy not available

    if HAS_TORCH:
        indices = np.vstack([row_idx, col_idx])
        A_sparse = torch.sparse_coo_tensor(
            torch.from_numpy(indices).long(),
            torch.from_numpy(vals),
            (rows, cols)
        ).to_sparse_csr()
        vec_cpu = torch.from_numpy(vec)
        bench_fn("spmv_csr", "linalg", "torch_cpu",
                 lambda: torch.mv(A_sparse, vec_cpu), rows)

    # ── Batched matmul ──
    batch, m, k, nn = 64, 128, 128, 128
    total = batch * m * nn
    a_np = np.random.randn(batch, m, k).astype(np.float32)
    b_np = np.random.randn(batch, k, nn).astype(np.float32)

    bench_fn("matmul_batched", "linalg", "numpy",
             lambda: np.matmul(a_np, b_np), total, elem_bytes=12)
    if HAS_TORCH:
        a_cpu = torch.from_numpy(a_np)
        b_cpu = torch.from_numpy(b_np)
        bench_fn("matmul_batched", "linalg", "torch_cpu",
                 lambda: torch.bmm(a_cpu, b_cpu), total, elem_bytes=12)
        if HAS_TORCH_MPS:
            a_mps = a_cpu.to("mps")
            b_mps = b_cpu.to("mps")
            bench_fn("matmul_batched", "linalg", "torch_mps",
                     lambda: torch.bmm(a_mps, b_mps), total, elem_bytes=12, sync=_sync_mps)
    if HAS_MLX:
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        bench_fn("matmul_batched", "linalg", "mlx",
                 lambda: mx.eval(a_mx @ b_mx), total, elem_bytes=12, sync=_sync_mlx)

    # ── Matvec ──
    mat_n = 4096
    A_np = np.random.randn(mat_n, mat_n).astype(np.float32)
    x_np = np.random.randn(mat_n).astype(np.float32)

    bench_fn("matvec", "linalg", "numpy",
             lambda: A_np @ x_np, mat_n * mat_n)
    if HAS_TORCH:
        A_cpu = torch.from_numpy(A_np)
        x_cpu = torch.from_numpy(x_np)
        bench_fn("matvec", "linalg", "torch_cpu",
                 lambda: torch.mv(A_cpu, x_cpu), mat_n * mat_n)
        if HAS_TORCH_MPS:
            A_mps = A_cpu.to("mps")
            x_mps = x_cpu.to("mps")
            bench_fn("matvec", "linalg", "torch_mps",
                     lambda: torch.mv(A_mps, x_mps), mat_n * mat_n, sync=_sync_mps)
    if HAS_MLX:
        A_mx = mx.array(A_np)
        x_mx = mx.array(x_np)
        bench_fn("matvec", "linalg", "mlx",
                 lambda: mx.eval(A_mx @ x_mx), mat_n * mat_n, sync=_sync_mlx)

    # ── Triangular solve ──
    L_np = np.tril(np.random.randn(mat_n, mat_n).astype(np.float32))
    np.fill_diagonal(L_np, np.abs(np.diag(L_np)) + 1.0)
    b_np_solve = np.random.randn(mat_n).astype(np.float32)

    bench_fn("triangular_solve", "linalg", "numpy",
             lambda: np.linalg.solve(L_np, b_np_solve), mat_n * mat_n)
    if HAS_TORCH:
        L_cpu = torch.from_numpy(L_np)
        b_cpu_solve = torch.from_numpy(b_np_solve).unsqueeze(1)
        bench_fn("triangular_solve", "linalg", "torch_cpu",
                 lambda: torch.linalg.solve_triangular(L_cpu, b_cpu_solve, upper=False), mat_n * mat_n)

    # ── Outer product ──
    u_np = np.random.randn(mat_n).astype(np.float32)
    v_np = np.random.randn(mat_n).astype(np.float32)

    bench_fn("outer_product", "linalg", "numpy",
             lambda: np.outer(u_np, v_np), mat_n * mat_n)
    if HAS_TORCH:
        u_cpu = torch.from_numpy(u_np)
        v_cpu = torch.from_numpy(v_np)
        bench_fn("outer_product", "linalg", "torch_cpu",
                 lambda: torch.outer(u_cpu, v_cpu), mat_n * mat_n)
        if HAS_TORCH_MPS:
            u_mps = u_cpu.to("mps")
            v_mps = v_cpu.to("mps")
            bench_fn("outer_product", "linalg", "torch_mps",
                     lambda: torch.outer(u_mps, v_mps), mat_n * mat_n, sync=_sync_mps)
    if HAS_MLX:
        u_mx = mx.array(u_np)
        v_mx = mx.array(v_np)
        bench_fn("outer_product", "linalg", "mlx",
                 lambda: mx.eval(mx.outer(u_mx, v_mx)), mat_n * mat_n, sync=_sync_mlx)
