//! ARM NEON SIMD implementations for CPU benchmarks.
//! Uses std::arch::aarch64 intrinsics on Apple Silicon.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Sum f32 array using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn sum_f32(data: &[f32]) -> f32 {
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        let mut result = vaddvq_f32(acc);
        for i in (chunks * 4)..data.len() {
            result += data[i];
        }
        result
    }
}

/// Min f32 array using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn min_f32(data: &[f32]) -> f32 {
    unsafe {
        let mut acc = vdupq_n_f32(f32::MAX);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            acc = vminq_f32(acc, v);
        }
        let mut result = vminnmvq_f32(acc);
        for i in (chunks * 4)..data.len() {
            result = result.min(data[i]);
        }
        result
    }
}

/// Max f32 array using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn max_f32(data: &[f32]) -> f32 {
    unsafe {
        let mut acc = vdupq_n_f32(f32::MIN);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            acc = vmaxq_f32(acc, v);
        }
        let mut result = vmaxnmvq_f32(acc);
        for i in (chunks * 4)..data.len() {
            result = result.max(data[i]);
        }
        result
    }
}

/// Dot product using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let mut result = vaddvq_f32(acc);
        for i in (chunks * 4)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

/// Elementwise add using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    unsafe {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vaddq_f32(va, vb));
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] + b[i];
        }
    }
}

/// Elementwise multiply using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    unsafe {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vmulq_f32(va, vb));
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] * b[i];
        }
    }
}

/// FMA: a * b + c using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let vc = vld1q_f32(c.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vfmaq_f32(vc, va, vb));
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

/// Abs using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn abs_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vabsq_f32(v));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = data[i].abs();
        }
    }
}

/// Clamp using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn clamp_f32(data: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    unsafe {
        let v_lo = vdupq_n_f32(lo);
        let v_hi = vdupq_n_f32(hi);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let clamped = vminq_f32(vmaxq_f32(v, v_lo), v_hi);
            vst1q_f32(out.as_mut_ptr().add(i * 4), clamped);
        }
        for i in (chunks * 4)..data.len() {
            out[i] = data[i].clamp(lo, hi);
        }
    }
}

/// ReLU using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn relu_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vmaxq_f32(v, zero));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = data[i].max(0.0);
        }
    }
}

/// L2 norm squared using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn l2_squared_f32(data: &[f32]) -> f32 {
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        let mut result = vaddvq_f32(acc);
        for i in (chunks * 4)..data.len() {
            result += data[i] * data[i];
        }
        result
    }
}

/// SGD update: params -= lr * grads, using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn sgd_f32(params: &mut [f32], grads: &[f32], lr: f32) {
    unsafe {
        let v_lr = vdupq_n_f32(lr);
        let chunks = params.len() / 4;
        for i in 0..chunks {
            let vp = vld1q_f32(params.as_ptr().add(i * 4));
            let vg = vld1q_f32(grads.as_ptr().add(i * 4));
            let updated = vmlsq_f32(vp, vg, v_lr);
            vst1q_f32(params.as_mut_ptr().add(i * 4), updated);
        }
        for i in (chunks * 4)..params.len() {
            params[i] -= lr * grads[i];
        }
    }
}

/// f32 to f16 conversion using scalar bit manipulation (vectorized by LLVM on aarch64).
#[cfg(target_arch = "aarch64")]
pub fn f32_to_f16(data: &[f32], out: &mut [u16]) {
    for i in 0..data.len() {
        out[i] = f16_from_f32(data[i]);
    }
}

/// Scalar f32 to f16.
fn f16_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x7FFFFF;
    if exp <= 0 {
        sign as u16
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        (sign | (exp as u32) << 10 | (mantissa >> 13)) as u16
    }
}

// Scalar fallback for non-aarch64 (won't be used on Apple Silicon)
#[cfg(not(target_arch = "aarch64"))]
pub fn sum_f32(data: &[f32]) -> f32 { data.iter().sum() }
#[cfg(not(target_arch = "aarch64"))]
pub fn min_f32(data: &[f32]) -> f32 { data.iter().cloned().fold(f32::MAX, f32::min) }
#[cfg(not(target_arch = "aarch64"))]
pub fn max_f32(data: &[f32]) -> f32 { data.iter().cloned().fold(f32::MIN, f32::max) }
#[cfg(not(target_arch = "aarch64"))]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
#[cfg(not(target_arch = "aarch64"))]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) { for i in 0..a.len() { out[i] = a[i]+b[i]; } }
#[cfg(not(target_arch = "aarch64"))]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) { for i in 0..a.len() { out[i] = a[i]*b[i]; } }
#[cfg(not(target_arch = "aarch64"))]
pub fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) { for i in 0..a.len() { out[i] = a[i]*b[i]+c[i]; } }
#[cfg(not(target_arch = "aarch64"))]
pub fn abs_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i].abs(); } }
#[cfg(not(target_arch = "aarch64"))]
pub fn clamp_f32(data: &[f32], lo: f32, hi: f32, out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i].clamp(lo, hi); } }
#[cfg(not(target_arch = "aarch64"))]
pub fn relu_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i].max(0.0); } }
#[cfg(not(target_arch = "aarch64"))]
pub fn l2_squared_f32(data: &[f32]) -> f32 { data.iter().map(|x| x*x).sum() }
#[cfg(not(target_arch = "aarch64"))]
pub fn sgd_f32(params: &mut [f32], grads: &[f32], lr: f32) { for i in 0..params.len() { params[i] -= lr * grads[i]; } }
#[cfg(not(target_arch = "aarch64"))]
pub fn f32_to_f16(data: &[f32], out: &mut [u16]) { for i in 0..data.len() { out[i] = f16_from_f32(data[i]); } }
