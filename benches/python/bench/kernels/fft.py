"""FFT & Spectral: radix-2, radix-4, inverse FFT, real FFT, spectral power density."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    # Use power-of-2 sizes for FFT
    fft_n = 1 << 20  # 1M points

    real_np = np.random.randn(fft_n).astype(np.float32)
    complex_np = np.random.randn(fft_n).astype(np.float32) + 1j * np.random.randn(fft_n).astype(np.float32)
    complex_np = complex_np.astype(np.complex64)

    # ── FFT radix-2 (complex) ──
    bench_fn("fft_radix2", "fft", "numpy",
             lambda: np.fft.fft(complex_np), fft_n, elem_bytes=8)
    if HAS_TORCH:
        complex_cpu = torch.from_numpy(complex_np)
        bench_fn("fft_radix2", "fft", "torch_cpu",
                 lambda: torch.fft.fft(complex_cpu), fft_n, elem_bytes=8)
        if HAS_TORCH_MPS:
            complex_mps = complex_cpu.to("mps")
            bench_fn("fft_radix2", "fft", "torch_mps",
                     lambda: torch.fft.fft(complex_mps), fft_n, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        complex_mx = mx.array(complex_np)
        bench_fn("fft_radix2", "fft", "mlx",
                 lambda: mx.eval(mx.fft.fft(complex_mx)), fft_n, elem_bytes=8, sync=_sync_mlx)

    # ── FFT radix-4 (power-of-4 size) ──
    fft_r4 = 1 << 20  # 4^10 = 1048576
    complex_r4 = complex_np[:fft_r4]
    bench_fn("fft_radix4", "fft", "numpy",
             lambda: np.fft.fft(complex_r4), fft_r4, elem_bytes=8)
    if HAS_TORCH:
        complex_r4_cpu = torch.from_numpy(complex_r4)
        bench_fn("fft_radix4", "fft", "torch_cpu",
                 lambda: torch.fft.fft(complex_r4_cpu), fft_r4, elem_bytes=8)

    # ── Inverse FFT ──
    freq_np = np.fft.fft(complex_np)
    bench_fn("ifft", "fft", "numpy",
             lambda: np.fft.ifft(freq_np), fft_n, elem_bytes=8)
    if HAS_TORCH:
        freq_cpu = torch.from_numpy(freq_np)
        bench_fn("ifft", "fft", "torch_cpu",
                 lambda: torch.fft.ifft(freq_cpu), fft_n, elem_bytes=8)
        if HAS_TORCH_MPS:
            freq_mps = freq_cpu.to("mps")
            bench_fn("ifft", "fft", "torch_mps",
                     lambda: torch.fft.ifft(freq_mps), fft_n, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        freq_mx = mx.array(freq_np)
        bench_fn("ifft", "fft", "mlx",
                 lambda: mx.eval(mx.fft.ifft(freq_mx)), fft_n, elem_bytes=8, sync=_sync_mlx)

    # ── Real FFT ──
    bench_fn("fft_real", "fft", "numpy",
             lambda: np.fft.rfft(real_np), fft_n)
    if HAS_TORCH:
        real_cpu = torch.from_numpy(real_np)
        bench_fn("fft_real", "fft", "torch_cpu",
                 lambda: torch.fft.rfft(real_cpu), fft_n)
        if HAS_TORCH_MPS:
            real_mps = real_cpu.to("mps")
            bench_fn("fft_real", "fft", "torch_mps",
                     lambda: torch.fft.rfft(real_mps), fft_n, sync=_sync_mps)
    if HAS_MLX:
        real_mx = mx.array(real_np)
        bench_fn("fft_real", "fft", "mlx",
                 lambda: mx.eval(mx.fft.rfft(real_mx)), fft_n, sync=_sync_mlx)

    # ── Spectral power density ──
    bench_fn("spectral_power", "fft", "numpy",
             lambda: np.abs(np.fft.rfft(real_np)) ** 2, fft_n)
    if HAS_TORCH:
        bench_fn("spectral_power", "fft", "torch_cpu",
                 lambda: torch.abs(torch.fft.rfft(real_cpu)) ** 2, fft_n)
        if HAS_TORCH_MPS:
            bench_fn("spectral_power", "fft", "torch_mps",
                     lambda: torch.abs(torch.fft.rfft(real_mps)) ** 2, fft_n, sync=_sync_mps)
