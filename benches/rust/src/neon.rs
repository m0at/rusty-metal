//! ARM NEON SIMD implementations for CPU benchmarks.
//! Uses std::arch::aarch64 intrinsics on Apple Silicon.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// Helper: NEON fast exp approximation (12-bit precision)
// Uses the classic integer bit-cast trick: exp(x) ~ 2^(x/ln2)
// Split into integer part (bit shift) and fractional part (polynomial).
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_exp_f32(x: float32x4_t) -> float32x4_t {
    let ln2_inv = vdupq_n_f32(1.442695041); // 1/ln(2)
    let ln2 = vdupq_n_f32(0.6931471806);
    let one = vdupq_n_f32(1.0);
    let c1 = vdupq_n_f32(0.5);
    let c2 = vdupq_n_f32(0.166666667);
    let c3 = vdupq_n_f32(0.041666667);
    let c4 = vdupq_n_f32(0.008333333);

    // Clamp to avoid overflow/underflow
    let x = vmaxq_f32(vdupq_n_f32(-88.0), vminq_f32(vdupq_n_f32(88.0), x));

    // t = x / ln(2)
    let t = vmulq_f32(x, ln2_inv);

    // Integer part: n = floor(t)
    let n = vrndmq_f32(t); // floor

    // Fractional part: f = x - n * ln(2)
    let f = vmlsq_f32(x, n, ln2);

    // Polynomial approximation of exp(f) for f in [0, ln2):
    // 1 + f + f^2/2 + f^3/6 + f^4/24 + f^5/120
    let f2 = vmulq_f32(f, f);
    let f3 = vmulq_f32(f2, f);
    let f4 = vmulq_f32(f2, f2);
    let f5 = vmulq_f32(f4, f);
    let poly = vaddq_f32(one,
        vaddq_f32(f,
            vaddq_f32(vmulq_f32(c1, f2),
                vaddq_f32(vmulq_f32(c2, f3),
                    vaddq_f32(vmulq_f32(c3, f4),
                        vmulq_f32(c4, f5))))));

    // Reconstruct: 2^n * poly
    // 2^n via integer bit manipulation: add n*2^23 to the float exponent bits
    let n_i32 = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n_i32, vdupq_n_s32(127)), 23));

    vmulq_f32(poly, pow2n)
}

// ============================================================================
// Helper: NEON fast log approximation
// Uses bit extraction: log(x) = (exponent - 127) * ln(2) + log(mantissa)
// where mantissa in [1,2) and log(m) is approximated with a polynomial.
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_log_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let ln2 = vdupq_n_f32(0.6931471806);
    let min_positive = vdupq_n_f32(1.17549435e-38);

    // Clamp to minimum positive to avoid log(0) / log(negative)
    let x = vmaxq_f32(x, min_positive);

    // Extract exponent: reinterpret as int, shift right 23, subtract 127
    let xi = vreinterpretq_s32_f32(x);
    let exponent = vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(xi, 23), vdupq_n_s32(127)));

    // Extract mantissa in [1, 2): mask mantissa bits, set exponent to 127
    let mantissa_mask = vdupq_n_s32(0x007FFFFF);
    let exp_127 = vdupq_n_s32(0x3F800000);
    let m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(xi, mantissa_mask), exp_127));

    // log(m) for m in [1,2) via polynomial: let p = m - 1
    let p = vsubq_f32(m, one);

    // Minimax polynomial for ln(1+p), p in [0,1):
    // ln(1+p) ~ p - p^2/2 + p^3/3 - p^4/4 + p^5/5
    let p2 = vmulq_f32(p, p);
    let p3 = vmulq_f32(p2, p);
    let p4 = vmulq_f32(p2, p2);
    let p5 = vmulq_f32(p4, p);

    let c2 = vdupq_n_f32(-0.5);
    let c3 = vdupq_n_f32(0.333333333);
    let c4 = vdupq_n_f32(-0.25);
    let c5 = vdupq_n_f32(0.2);

    let log_m = vaddq_f32(p,
        vaddq_f32(vmulq_f32(c2, p2),
            vaddq_f32(vmulq_f32(c3, p3),
                vaddq_f32(vmulq_f32(c4, p4),
                    vmulq_f32(c5, p5)))));

    // result = exponent * ln(2) + log(mantissa)
    vfmaq_f32(log_m, exponent, ln2)
}

// ============================================================================
// Helper: NEON fast sigmoid: 1 / (1 + exp(-x))
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_sigmoid_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let neg_x = vnegq_f32(x);
    let exp_neg_x = neon_exp_f32(neg_x);
    // Approximate reciprocal then refine with Newton step
    let denom = vaddq_f32(one, exp_neg_x);
    let recip = vrecpeq_f32(denom);
    let recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
    recip
}

// ============================================================================
// Helper: NEON fast tanh: (exp(2x) - 1) / (exp(2x) + 1)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_tanh_f32(x: float32x4_t) -> float32x4_t {
    let two = vdupq_n_f32(2.0);
    let one = vdupq_n_f32(1.0);
    let two_x = vmulq_f32(x, two);
    let exp2x = neon_exp_f32(two_x);
    let num = vsubq_f32(exp2x, one);
    let den = vaddq_f32(exp2x, one);
    let recip = vrecpeq_f32(den);
    let recip = vmulq_f32(recip, vrecpsq_f32(den, recip));
    vmulq_f32(num, recip)
}

/// Scalar exp helper for tail elements.
#[inline(always)]
fn scalar_exp_approx(x: f32) -> f32 {
    x.exp()
}

/// Scalar log helper for tail elements.
#[inline(always)]
fn scalar_log_approx(x: f32) -> f32 {
    x.max(1.17549435e-38).ln()
}

// ============================================================================
// EXISTING FUNCTIONS
// ============================================================================

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

// ============================================================================
// NEW: Reductions
// ============================================================================

/// Mean f32 array using NEON SIMD: sum then divide.
#[cfg(target_arch = "aarch64")]
pub fn mean_f32(data: &[f32]) -> f32 {
    sum_f32(data) / data.len() as f32
}

/// Variance f32 array using NEON SIMD (two-pass: mean, then sum-of-squared-diffs).
#[cfg(target_arch = "aarch64")]
pub fn variance_f32(data: &[f32]) -> f32 {
    let mean = mean_f32(data);
    unsafe {
        let v_mean = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let diff = vsubq_f32(v, v_mean);
            acc = vfmaq_f32(acc, diff, diff);
        }
        let mut result = vaddvq_f32(acc);
        for i in (chunks * 4)..data.len() {
            let diff = data[i] - mean;
            result += diff * diff;
        }
        result / data.len() as f32
    }
}

/// Argmax f32 array using NEON SIMD with index tracking.
/// Returns (max_value, index_of_max).
#[cfg(target_arch = "aarch64")]
pub fn argmax_f32(data: &[f32]) -> (f32, usize) {
    if data.is_empty() {
        return (f32::MIN, 0);
    }
    unsafe {
        let chunks = data.len() / 4;
        // Track 4 lane-wise maxima and their indices
        let mut max_vals = vdupq_n_f32(f32::MIN);
        let mut max_idxs = vdupq_n_u32(0);

        for i in 0..chunks {
            let offset = (i * 4) as u32;
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let idx = vld1q_u32([offset, offset + 1, offset + 2, offset + 3].as_ptr());
            // Compare: which lanes have new max?
            let mask = vcgtq_f32(v, max_vals);
            max_vals = vbslq_f32(mask, v, max_vals);
            max_idxs = vbslq_u32(mask, idx, max_idxs);
        }

        // Reduce 4 lanes to scalar
        let mut best_val = f32::MIN;
        let mut best_idx = 0usize;

        let vals: [f32; 4] = core::mem::transmute(max_vals);
        let idxs: [u32; 4] = core::mem::transmute(max_idxs);
        for lane in 0..4 {
            if vals[lane] > best_val {
                best_val = vals[lane];
                best_idx = idxs[lane] as usize;
            }
        }

        // Handle tail
        for i in (chunks * 4)..data.len() {
            if data[i] > best_val {
                best_val = data[i];
                best_idx = i;
            }
        }

        (best_val, best_idx)
    }
}

/// Argmin f32 array using NEON SIMD with index tracking.
/// Returns (min_value, index_of_min).
#[cfg(target_arch = "aarch64")]
pub fn argmin_f32(data: &[f32]) -> (f32, usize) {
    if data.is_empty() {
        return (f32::MAX, 0);
    }
    unsafe {
        let chunks = data.len() / 4;
        let mut min_vals = vdupq_n_f32(f32::MAX);
        let mut min_idxs = vdupq_n_u32(0);

        for i in 0..chunks {
            let offset = (i * 4) as u32;
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let idx = vld1q_u32([offset, offset + 1, offset + 2, offset + 3].as_ptr());
            let mask = vcltq_f32(v, min_vals);
            min_vals = vbslq_f32(mask, v, min_vals);
            min_idxs = vbslq_u32(mask, idx, min_idxs);
        }

        let mut best_val = f32::MAX;
        let mut best_idx = 0usize;

        let vals: [f32; 4] = core::mem::transmute(min_vals);
        let idxs: [u32; 4] = core::mem::transmute(min_idxs);
        for lane in 0..4 {
            if vals[lane] < best_val {
                best_val = vals[lane];
                best_idx = idxs[lane] as usize;
            }
        }

        for i in (chunks * 4)..data.len() {
            if data[i] < best_val {
                best_val = data[i];
                best_idx = i;
            }
        }

        (best_val, best_idx)
    }
}

// ============================================================================
// NEW: Elementwise transcendentals
// ============================================================================

/// Elementwise exp using NEON polynomial approximation.
#[cfg(target_arch = "aarch64")]
pub fn exp_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), neon_exp_f32(v));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = scalar_exp_approx(data[i]);
        }
    }
}

/// Elementwise log using NEON bit-extraction + polynomial approximation.
#[cfg(target_arch = "aarch64")]
pub fn log_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), neon_log_f32(v));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = scalar_log_approx(data[i]);
        }
    }
}

/// Elementwise sigmoid: 1/(1+exp(-x)) using NEON approximation.
#[cfg(target_arch = "aarch64")]
pub fn sigmoid_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), neon_sigmoid_f32(v));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = 1.0 / (1.0 + (-data[i]).exp());
        }
    }
}

/// Elementwise tanh using NEON approximation.
#[cfg(target_arch = "aarch64")]
pub fn tanh_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), neon_tanh_f32(v));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = data[i].tanh();
        }
    }
}

/// Elementwise softplus: ln(1 + exp(x)) using NEON.
#[cfg(target_arch = "aarch64")]
pub fn softplus_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let one = vdupq_n_f32(1.0);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let exp_v = neon_exp_f32(v);
            let sum = vaddq_f32(one, exp_v);
            vst1q_f32(out.as_mut_ptr().add(i * 4), neon_log_f32(sum));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = (1.0 + data[i].exp()).ln();
        }
    }
}

/// Elementwise divide using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    unsafe {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            vst1q_f32(out.as_mut_ptr().add(i * 4), vdivq_f32(va, vb));
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] / b[i];
        }
    }
}

/// Elementwise compare (a > b ? 1.0 : 0.0) using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub fn compare_gt_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    unsafe {
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let mask = vcgtq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i * 4), vbslq_f32(mask, one, zero));
        }
        for i in (chunks * 4)..a.len() {
            out[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
        }
    }
}

// ============================================================================
// NEW: Activations
// ============================================================================

/// Leaky ReLU using NEON: max(alpha*x, x) via comparison + blend.
#[cfg(target_arch = "aarch64")]
pub fn leaky_relu_f32(data: &[f32], alpha: f32, out: &mut [f32]) {
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let v_alpha = vdupq_n_f32(alpha);
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            let scaled = vmulq_f32(v, v_alpha);
            let mask = vcgtq_f32(v, zero); // true where x > 0
            vst1q_f32(out.as_mut_ptr().add(i * 4), vbslq_f32(mask, v, scaled));
        }
        for i in (chunks * 4)..data.len() {
            out[i] = if data[i] > 0.0 { data[i] } else { alpha * data[i] };
        }
    }
}

/// GELU approximation using NEON tanh path:
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[cfg(target_arch = "aarch64")]
pub fn gelu_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let coeff = vdupq_n_f32(0.044715);
        let sqrt_2_pi = vdupq_n_f32(0.7978845608); // sqrt(2/pi)
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let x = vld1q_f32(data.as_ptr().add(i * 4));
            let x3 = vmulq_f32(vmulq_f32(x, x), x);
            let inner = vmulq_f32(sqrt_2_pi, vfmaq_f32(x, coeff, x3));
            let tanh_val = neon_tanh_f32(inner);
            let result = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val));
            vst1q_f32(out.as_mut_ptr().add(i * 4), result);
        }
        for i in (chunks * 4)..data.len() {
            let x = data[i];
            out[i] = 0.5 * x * (1.0 + (0.7978845608_f64 * (x as f64 + 0.044715 * (x as f64).powi(3))).tanh() as f32);
        }
    }
}

/// SiLU (Swish): x * sigmoid(x) using NEON sigmoid.
#[cfg(target_arch = "aarch64")]
pub fn silu_f32(data: &[f32], out: &mut [f32]) {
    unsafe {
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let x = vld1q_f32(data.as_ptr().add(i * 4));
            let sig = neon_sigmoid_f32(x);
            vst1q_f32(out.as_mut_ptr().add(i * 4), vmulq_f32(x, sig));
        }
        for i in (chunks * 4)..data.len() {
            let x = data[i];
            out[i] = x / (1.0 + (-x).exp());
        }
    }
}

// ============================================================================
// NEW: Normalization
// ============================================================================

/// Layer normalization using NEON SIMD.
/// data shape: [batch, dim], gamma/beta shape: [dim].
#[cfg(target_arch = "aarch64")]
pub fn layer_norm_f32(data: &[f32], gamma: &[f32], beta: &[f32], dim: usize, out: &mut [f32]) {
    let batch = data.len() / dim;
    let eps = 1e-5f32;
    for b in 0..batch {
        let row = &data[b * dim..(b + 1) * dim];
        let out_row = &mut out[b * dim..(b + 1) * dim];

        // NEON mean
        let mean = sum_f32(row) / dim as f32;

        // NEON variance
        unsafe {
            let v_mean = vdupq_n_f32(mean);
            let mut acc = vdupq_n_f32(0.0);
            let chunks = dim / 4;
            for i in 0..chunks {
                let v = vld1q_f32(row.as_ptr().add(i * 4));
                let diff = vsubq_f32(v, v_mean);
                acc = vfmaq_f32(acc, diff, diff);
            }
            let mut var = vaddvq_f32(acc);
            for i in (chunks * 4)..dim {
                let diff = row[i] - mean;
                var += diff * diff;
            }
            var /= dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // NEON normalize: gamma * (x - mean) * inv_std + beta
            let v_inv_std = vdupq_n_f32(inv_std);
            for i in 0..chunks {
                let v = vld1q_f32(row.as_ptr().add(i * 4));
                let vg = vld1q_f32(gamma.as_ptr().add(i * 4));
                let vb = vld1q_f32(beta.as_ptr().add(i * 4));
                let normalized = vmulq_f32(vsubq_f32(v, v_mean), v_inv_std);
                let result = vfmaq_f32(vb, vg, normalized);
                vst1q_f32(out_row.as_mut_ptr().add(i * 4), result);
            }
            for i in (chunks * 4)..dim {
                out_row[i] = gamma[i] * (row[i] - mean) * inv_std + beta[i];
            }
        }
    }
}

/// RMS normalization using NEON SIMD.
/// data shape: [batch, dim], gamma shape: [dim].
#[cfg(target_arch = "aarch64")]
pub fn rms_norm_f32(data: &[f32], gamma: &[f32], dim: usize, out: &mut [f32]) {
    let batch = data.len() / dim;
    let eps = 1e-5f32;
    for b in 0..batch {
        let row = &data[b * dim..(b + 1) * dim];
        let out_row = &mut out[b * dim..(b + 1) * dim];

        // NEON sum of squares
        let ss = l2_squared_f32(row);
        let rms = (ss / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // NEON normalize: gamma * x * inv_rms
        unsafe {
            let v_inv_rms = vdupq_n_f32(inv_rms);
            let chunks = dim / 4;
            for i in 0..chunks {
                let v = vld1q_f32(row.as_ptr().add(i * 4));
                let vg = vld1q_f32(gamma.as_ptr().add(i * 4));
                let scaled = vmulq_f32(v, v_inv_rms);
                vst1q_f32(out_row.as_mut_ptr().add(i * 4), vmulq_f32(vg, scaled));
            }
            for i in (chunks * 4)..dim {
                out_row[i] = gamma[i] * row[i] * inv_rms;
            }
        }
    }
}

// ============================================================================
// NEW: Loss functions
// ============================================================================

/// MSE loss using NEON: mean((pred - target)^2).
#[cfg(target_arch = "aarch64")]
pub fn mse_f32(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len());
    let n = pred.len();
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let chunks = n / 4;
        for i in 0..chunks {
            let vp = vld1q_f32(pred.as_ptr().add(i * 4));
            let vt = vld1q_f32(target.as_ptr().add(i * 4));
            let diff = vsubq_f32(vp, vt);
            acc = vfmaq_f32(acc, diff, diff);
        }
        let mut result = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            let diff = pred[i] - target[i];
            result += diff * diff;
        }
        result / n as f32
    }
}

/// MAE loss using NEON: mean(|pred - target|).
#[cfg(target_arch = "aarch64")]
pub fn mae_f32(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len());
    let n = pred.len();
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let chunks = n / 4;
        for i in 0..chunks {
            let vp = vld1q_f32(pred.as_ptr().add(i * 4));
            let vt = vld1q_f32(target.as_ptr().add(i * 4));
            let diff = vsubq_f32(vp, vt);
            acc = vaddq_f32(acc, vabsq_f32(diff));
        }
        let mut result = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            result += (pred[i] - target[i]).abs();
        }
        result / n as f32
    }
}

// ============================================================================
// NEW: Dot products (expand)
// ============================================================================

/// Batched cosine similarity using NEON SIMD.
/// a, b shape: [batch, dim], out shape: [batch].
#[cfg(target_arch = "aarch64")]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32], dim: usize, batch: usize, out: &mut [f32]) {
    for i in 0..batch {
        let base = i * dim;
        let row_a = &a[base..base + dim];
        let row_b = &b[base..base + dim];

        unsafe {
            let mut acc_dot = vdupq_n_f32(0.0);
            let mut acc_na = vdupq_n_f32(0.0);
            let mut acc_nb = vdupq_n_f32(0.0);
            let chunks = dim / 4;
            for j in 0..chunks {
                let va = vld1q_f32(row_a.as_ptr().add(j * 4));
                let vb = vld1q_f32(row_b.as_ptr().add(j * 4));
                acc_dot = vfmaq_f32(acc_dot, va, vb);
                acc_na = vfmaq_f32(acc_na, va, va);
                acc_nb = vfmaq_f32(acc_nb, vb, vb);
            }
            let mut dot = vaddvq_f32(acc_dot);
            let mut norm_a = vaddvq_f32(acc_na);
            let mut norm_b = vaddvq_f32(acc_nb);
            for j in (chunks * 4)..dim {
                dot += row_a[j] * row_b[j];
                norm_a += row_a[j] * row_a[j];
                norm_b += row_b[j] * row_b[j];
            }
            out[i] = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-8);
        }
    }
}

// ============================================================================
// NEW: Optimizers
// ============================================================================

/// Adam optimizer step using NEON SIMD.
/// Updates params, m (first moment), v (second moment) in-place.
#[cfg(target_arch = "aarch64")]
pub fn adam_f32(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) {
    let n = params.len();
    unsafe {
        let v_lr = vdupq_n_f32(lr);
        let v_beta1 = vdupq_n_f32(beta1);
        let v_beta2 = vdupq_n_f32(beta2);
        let v_one_minus_b1 = vdupq_n_f32(1.0 - beta1);
        let v_one_minus_b2 = vdupq_n_f32(1.0 - beta2);
        let v_eps = vdupq_n_f32(eps);
        let chunks = n / 4;

        for i in 0..chunks {
            let off = i * 4;
            let vp = vld1q_f32(params.as_ptr().add(off));
            let vg = vld1q_f32(grads.as_ptr().add(off));
            let vm = vld1q_f32(m.as_ptr().add(off));
            let vv = vld1q_f32(v.as_ptr().add(off));

            // m = beta1 * m + (1 - beta1) * g
            let new_m = vfmaq_f32(vmulq_f32(v_beta1, vm), v_one_minus_b1, vg);

            // v = beta2 * v + (1 - beta2) * g^2
            let g2 = vmulq_f32(vg, vg);
            let new_v = vfmaq_f32(vmulq_f32(v_beta2, vv), v_one_minus_b2, g2);

            // p -= lr * m / (sqrt(v) + eps)
            let sqrt_v = vsqrtq_f32(new_v);
            let denom = vaddq_f32(sqrt_v, v_eps);
            let step = vdivq_f32(vmulq_f32(v_lr, new_m), denom);
            let new_p = vsubq_f32(vp, step);

            vst1q_f32(params.as_mut_ptr().add(off), new_p);
            vst1q_f32(m.as_mut_ptr().add(off), new_m);
            vst1q_f32(v.as_mut_ptr().add(off), new_v);
        }

        for i in (chunks * 4)..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            params[i] -= lr * m[i] / (v[i].sqrt() + eps);
        }
    }
}

// ============================================================================
// Scalar fallback for non-aarch64 (won't be used on Apple Silicon)
// ============================================================================

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

// NEW scalar fallbacks

#[cfg(not(target_arch = "aarch64"))]
pub fn mean_f32(data: &[f32]) -> f32 { data.iter().sum::<f32>() / data.len() as f32 }

#[cfg(not(target_arch = "aarch64"))]
pub fn variance_f32(data: &[f32]) -> f32 {
    let mean = mean_f32(data);
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

#[cfg(not(target_arch = "aarch64"))]
pub fn argmax_f32(data: &[f32]) -> (f32, usize) {
    let mut best = (f32::MIN, 0);
    for (i, &v) in data.iter().enumerate() {
        if v > best.0 { best = (v, i); }
    }
    best
}

#[cfg(not(target_arch = "aarch64"))]
pub fn argmin_f32(data: &[f32]) -> (f32, usize) {
    let mut best = (f32::MAX, 0);
    for (i, &v) in data.iter().enumerate() {
        if v < best.0 { best = (v, i); }
    }
    best
}

#[cfg(not(target_arch = "aarch64"))]
pub fn exp_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i].exp(); } }

#[cfg(not(target_arch = "aarch64"))]
pub fn log_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i].abs().max(1e-7).ln(); } }

#[cfg(not(target_arch = "aarch64"))]
pub fn sigmoid_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = 1.0 / (1.0 + (-data[i]).exp()); } }

#[cfg(not(target_arch = "aarch64"))]
pub fn tanh_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i].tanh(); } }

#[cfg(not(target_arch = "aarch64"))]
pub fn softplus_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = (1.0 + data[i].exp()).ln(); } }

#[cfg(not(target_arch = "aarch64"))]
pub fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) { for i in 0..a.len() { out[i] = a[i] / b[i]; } }

#[cfg(not(target_arch = "aarch64"))]
pub fn compare_gt_f32(a: &[f32], b: &[f32], out: &mut [f32]) { for i in 0..a.len() { out[i] = if a[i] > b[i] { 1.0 } else { 0.0 }; } }

#[cfg(not(target_arch = "aarch64"))]
pub fn leaky_relu_f32(data: &[f32], alpha: f32, out: &mut [f32]) { for i in 0..data.len() { out[i] = if data[i] > 0.0 { data[i] } else { alpha * data[i] }; } }

#[cfg(not(target_arch = "aarch64"))]
pub fn gelu_f32(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        let x = data[i];
        out[i] = 0.5 * x * (1.0 + (0.7978845608_f64 * (x as f64 + 0.044715 * (x as f64).powi(3))).tanh() as f32);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn silu_f32(data: &[f32], out: &mut [f32]) { for i in 0..data.len() { out[i] = data[i] / (1.0 + (-data[i]).exp()); } }

#[cfg(not(target_arch = "aarch64"))]
pub fn layer_norm_f32(data: &[f32], gamma: &[f32], beta: &[f32], dim: usize, out: &mut [f32]) {
    let batch = data.len() / dim;
    for b in 0..batch {
        let row = &data[b*dim..(b+1)*dim];
        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        let var: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for (i, &x) in row.iter().enumerate() {
            out[b*dim + i] = gamma[i] * (x - mean) * inv_std + beta[i];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rms_norm_f32(data: &[f32], gamma: &[f32], dim: usize, out: &mut [f32]) {
    let batch = data.len() / dim;
    for b in 0..batch {
        let row = &data[b*dim..(b+1)*dim];
        let rms = (row.iter().map(|x| x*x).sum::<f32>() / dim as f32 + 1e-5).sqrt();
        let inv_rms = 1.0 / rms;
        for (i, &x) in row.iter().enumerate() {
            out[b*dim + i] = gamma[i] * x * inv_rms;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn mse_f32(pred: &[f32], target: &[f32]) -> f32 {
    pred.iter().zip(target).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / pred.len() as f32
}

#[cfg(not(target_arch = "aarch64"))]
pub fn mae_f32(pred: &[f32], target: &[f32]) -> f32 {
    pred.iter().zip(target).map(|(p, t)| (p - t).abs()).sum::<f32>() / pred.len() as f32
}

#[cfg(not(target_arch = "aarch64"))]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32], dim: usize, batch: usize, out: &mut [f32]) {
    for i in 0..batch {
        let base = i * dim;
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for d in 0..dim {
            dot += a[base+d] * b[base+d];
            na += a[base+d] * a[base+d];
            nb += b[base+d] * b[base+d];
        }
        out[i] = dot / (na.sqrt() * nb.sqrt() + 1e-8);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn adam_f32(params: &mut [f32], grads: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, beta1: f32, beta2: f32, eps: f32) {
    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        params[i] -= lr * m[i] / (v[i].sqrt() + eps);
    }
}
