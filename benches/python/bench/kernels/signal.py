"""Signal Processing: conv1d, FFT conv1d, autocorrelation, cross-correlation, windowing, FIR."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    sig_len = min(n, 1_000_000)
    signal_np = np.random.randn(sig_len).astype(np.float32)
    signal2_np = np.random.randn(sig_len).astype(np.float32)

    # ── Conv1d (small kernel, direct) ──
    kernel_small = np.random.randn(32).astype(np.float32)
    bench_fn("conv1d", "signal", "numpy",
             lambda: np.convolve(signal_np, kernel_small, mode="same"), sig_len)
    if HAS_TORCH:
        sig_cpu = torch.from_numpy(signal_np).reshape(1, 1, -1)
        k_cpu = torch.from_numpy(kernel_small).reshape(1, 1, -1)
        bench_fn("conv1d", "signal", "torch_cpu",
                 lambda: F.conv1d(sig_cpu, k_cpu, padding=16), sig_len)
        if HAS_TORCH_MPS:
            sig_mps = sig_cpu.to("mps")
            k_mps = k_cpu.to("mps")
            bench_fn("conv1d", "signal", "torch_mps",
                     lambda: F.conv1d(sig_mps, k_mps, padding=16), sig_len, sync=_sync_mps)

    # ── FFT Conv1d (large kernel) ──
    kernel_large = np.random.randn(256).astype(np.float32)
    bench_fn("fft_conv1d", "signal", "numpy",
             lambda: _np_fft_conv(signal_np, kernel_large), sig_len)
    if HAS_TORCH:
        bench_fn("fft_conv1d", "signal", "torch_cpu",
                 lambda: _torch_fft_conv(signal_np, kernel_large), sig_len)

    # ── Autocorrelation ──
    bench_fn("autocorrelation", "signal", "numpy",
             lambda: _np_autocorr(signal_np), sig_len)
    if HAS_TORCH:
        bench_fn("autocorrelation", "signal", "torch_cpu",
                 lambda: _torch_autocorr(signal_np), sig_len)

    # ── Cross-correlation ──
    bench_fn("cross_correlation", "signal", "numpy",
             lambda: np.correlate(signal_np[:10000], signal2_np[:10000], mode="full"), 10000)

    # ── Window apply (Hann) ──
    hann_np = np.hanning(sig_len).astype(np.float32)
    bench_fn("window_apply", "signal", "numpy",
             lambda: signal_np * hann_np, sig_len, elem_bytes=8)
    if HAS_TORCH:
        sig1d = torch.from_numpy(signal_np)
        hann_cpu = torch.from_numpy(hann_np)
        bench_fn("window_apply", "signal", "torch_cpu",
                 lambda: sig1d * hann_cpu, sig_len, elem_bytes=8)
        if HAS_TORCH_MPS:
            sig1d_mps = sig1d.to("mps")
            hann_mps = hann_cpu.to("mps")
            bench_fn("window_apply", "signal", "torch_mps",
                     lambda: sig1d_mps * hann_mps, sig_len, elem_bytes=8, sync=_sync_mps)
    if HAS_MLX:
        sig_mx = mx.array(signal_np)
        hann_mx = mx.array(hann_np)
        bench_fn("window_apply", "signal", "mlx",
                 lambda: mx.eval(sig_mx * hann_mx), sig_len, elem_bytes=8, sync=_sync_mlx)

    # ── FIR filter (same as small conv1d but with reversed coefficients) ──
    fir_coeffs = kernel_small[::-1].copy()
    bench_fn("fir_filter", "signal", "numpy",
             lambda: np.convolve(signal_np, fir_coeffs, mode="same"), sig_len)


def _np_fft_conv(signal, kernel):
    n = len(signal) + len(kernel) - 1
    n_fft = 1 << int(np.ceil(np.log2(n)))
    return np.real(np.fft.ifft(np.fft.fft(signal, n_fft) * np.fft.fft(kernel, n_fft)))[:len(signal)]

def _np_autocorr(signal):
    n = len(signal)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.fft(signal, n_fft)
    return np.real(np.fft.ifft(f * np.conj(f)))[:n]

if HAS_TORCH:
    def _torch_fft_conv(signal, kernel):
        n = len(signal) + len(kernel) - 1
        n_fft = 1 << int(np.ceil(np.log2(n)))
        s = torch.from_numpy(signal)
        k = torch.from_numpy(kernel)
        return torch.fft.ifft(torch.fft.fft(s, n_fft) * torch.fft.fft(k, n_fft)).real[:len(signal)]

    def _torch_autocorr(signal):
        n = len(signal)
        n_fft = 1 << int(np.ceil(np.log2(2 * n)))
        s = torch.from_numpy(signal)
        f = torch.fft.fft(s, n_fft)
        return torch.fft.ifft(f * f.conj()).real[:n]
