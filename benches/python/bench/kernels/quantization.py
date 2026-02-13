"""Quantization: f16-to-i8, i8-to-f16, f32-to-f16."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    data_f32 = np.random.randn(n).astype(np.float32)
    data_f16 = data_f32.astype(np.float16)

    # ── F32 to F16 ──
    bench_fn("quantize_f32_to_f16", "quantization", "numpy",
             lambda: data_f32.astype(np.float16), n, elem_bytes=4)
    if HAS_TORCH:
        t_f32 = torch.from_numpy(data_f32)
        bench_fn("quantize_f32_to_f16", "quantization", "torch_cpu",
                 lambda: t_f32.half(), n, elem_bytes=4)
        if HAS_TORCH_MPS:
            t_mps = t_f32.to("mps")
            bench_fn("quantize_f32_to_f16", "quantization", "torch_mps",
                     lambda: t_mps.half(), n, elem_bytes=4, sync=_sync_mps)
    if HAS_MLX:
        m_f32 = mx.array(data_f32)
        bench_fn("quantize_f32_to_f16", "quantization", "mlx",
                 lambda: mx.eval(m_f32.astype(mx.float16)), n, elem_bytes=4, sync=_sync_mlx)

    # ── F16 to I8 (per-tensor quantization) ──
    bench_fn("quantize_f16_to_i8", "quantization", "numpy",
             lambda: _np_quant_i8(data_f16), n, elem_bytes=2)
    if HAS_TORCH:
        t_f16 = torch.from_numpy(data_f16)
        bench_fn("quantize_f16_to_i8", "quantization", "torch_cpu",
                 lambda: _torch_quant_i8(t_f16), n, elem_bytes=2)

    # ── I8 to F16 (dequantize) ──
    scale = np.max(np.abs(data_f16)) / 127.0
    data_i8 = np.clip(np.round(data_f16 / scale), -127, 127).astype(np.int8)
    bench_fn("dequantize_i8_to_f16", "quantization", "numpy",
             lambda: (data_i8.astype(np.float16) * np.float16(scale)), n, elem_bytes=1)
    if HAS_TORCH:
        t_i8 = torch.from_numpy(data_i8)
        bench_fn("dequantize_i8_to_f16", "quantization", "torch_cpu",
                 lambda: t_i8.half() * scale, n, elem_bytes=1)


def _np_quant_i8(data):
    scale = np.max(np.abs(data)) / 127.0
    return np.clip(np.round(data / scale), -127, 127).astype(np.int8)

if HAS_TORCH:
    def _torch_quant_i8(data):
        scale = torch.max(torch.abs(data)) / 127.0
        return torch.clamp(torch.round(data / scale), -127, 127).to(torch.int8)
