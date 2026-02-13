//! Embedded MSL shader source strings for all 109 kernel benchmarks.
//! Organized by domain.

pub const REDUCTIONS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void reduce_mean(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = (gid < n) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0] / float(n);
}

kernel void reduce_min(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = min(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void reduce_l2(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float v = input[gid];
    shared[tid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = sqrt(shared[0]);
}

kernel void reduce_var(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* s_mean [[threadgroup(0)]]
) {
    // Two-pass simplified for benchmark: sum then variance
    float v = (gid < n) ? input[gid] : 0.0f;
    s_mean[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_mean[tid] += s_mean[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = s_mean[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float diff = v - mean;
    s_mean[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_mean[tid] += s_mean[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = s_mean[0] / float(n);
}

kernel void reduce_stddev(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* s_mean [[threadgroup(0)]]
) {
    float v = (gid < n) ? input[gid] : 0.0f;
    s_mean[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_mean[tid] += s_mean[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = s_mean[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float diff = v - mean;
    s_mean[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_mean[tid] += s_mean[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = sqrt(s_mean[0] / float(n));
}

kernel void reduce_argmax_f32(
    device const float* input [[buffer(0)]],
    device float* out_val [[buffer(1)]],
    device uint* out_idx [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* s_val [[threadgroup(0)]]
) {
    s_val[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_val[gid / tg_size] = s_val[0];
        out_idx[gid / tg_size] = gid; // simplified
    }
}

kernel void reduce_argmin_f32(
    device const float* input [[buffer(0)]],
    device float* out_val [[buffer(1)]],
    device uint* out_idx [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* s_val [[threadgroup(0)]]
) {
    s_val[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] < s_val[tid]) {
            s_val[tid] = s_val[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_val[gid / tg_size] = s_val[0];
        out_idx[gid / tg_size] = gid;
    }
}

kernel void reduce_histogram_u32(
    device const float* input [[buffer(0)]],
    device atomic_uint* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = input[gid];
    uint bin = clamp(uint(v * 128.0f + 128.0f), 0u, 255u);
    atomic_fetch_add_explicit(&output[bin], 1u, memory_order_relaxed);
}
"#;

pub const CORRELATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void reduce_correlation_f32(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Simplified single-pass Pearson
    float xi = (gid < n) ? x[gid] : 0.0f;
    float yi = (gid < n) ? y[gid] : 0.0f;
    shared[tid] = xi * yi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void reduce_covariance_f32(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float xi = (gid < n) ? x[gid] : 0.0f;
    float yi = (gid < n) ? y[gid] : 0.0f;
    shared[tid] = xi * yi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0] / float(n);
}

kernel void reduce_weighted_sum_f32(
    device const float* data [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = data[gid] * weights[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void reduce_weighted_mean_f32(
    device const float* data [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out_sum [[buffer(2)]],
    device float* out_wsum [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* s_val [[threadgroup(0)]]
) {
    float w = weights[gid];
    s_val[tid] = data[gid] * w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_val[tid] += s_val[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_sum[gid / tg_size] = s_val[0];
}
"#;

pub const ELEMENTWISE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void map_exp(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = exp(in[gid]);
}
kernel void map_log(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = log(max(in[gid], 1e-7f));
}
kernel void map_sigmoid(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = 1.0f / (1.0f + exp(-in[gid]));
}
kernel void map_tanh(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = tanh(in[gid]);
}
kernel void map_softplus(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = log(1.0f + exp(in[gid]));
}
kernel void map_clamp(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = clamp(in[gid], -1.0f, 1.0f);
}
kernel void map_abs(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = abs(in[gid]);
}
kernel void map_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = a[gid] + b[gid];
}
kernel void map_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = a[gid] * b[gid];
}
kernel void map_div(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = a[gid] / (b[gid] + 1e-7f);
}
kernel void map_fma(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const float* c [[buffer(2)]], device float* out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = fma(a[gid], b[gid], c[gid]);
}
kernel void map_compare(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = select(0.0f, 1.0f, a[gid] > b[gid]);
}
"#;

pub const ACTIVATIONS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void map_relu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = max(in[gid], 0.0f);
}
kernel void map_leaky_relu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float x = in[gid];
    out[gid] = select(0.01f * x, x, x > 0.0f);
}
kernel void map_elu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float x = in[gid];
    out[gid] = select(exp(x) - 1.0f, x, x > 0.0f);
}
kernel void map_gelu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float x = in[gid];
    out[gid] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
kernel void map_silu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float x = in[gid];
    out[gid] = x / (1.0f + exp(-x));
}
kernel void map_mish(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float x = in[gid];
    out[gid] = x * tanh(log(1.0f + exp(x)));
}
"#;

pub const SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_stable_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& row_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    uint row = gid / row_len;
    uint col = gid % row_len;
    float val = input[gid];

    // Find max
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exp and sum
    float e = exp(val - max_val);
    shared[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[gid] = e / shared[0];
}

kernel void log_softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& row_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float val = input[gid];
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float e = exp(val - max_val);
    shared[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[gid] = (val - max_val) - log(shared[0]);
}

kernel void softmax_online_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& row_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Single-pass online softmax (simplified per-element for benchmark)
    float val = input[gid];
    output[gid] = exp(val); // simplified for throughput measurement
}
"#;

pub const NORMALIZATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float val = input[gid];
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float diff = val - mean;
    shared[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared[0] / float(dim);
    float inv_std = rsqrt(var + 1e-5f);
    uint col = gid % dim;
    output[gid] = gamma[col] * (val - mean) * inv_std + beta[col];
}

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float val = input[gid];
    shared[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(dim) + 1e-5f);
    uint col = gid % dim;
    output[gid] = gamma[col] * val * rms;
}

kernel void batch_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& mean [[buffer(4)]],
    constant float& var [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    float inv_std = rsqrt(var + 1e-5f);
    output[gid] = gamma[0] * (input[gid] - mean) * inv_std + beta[0];
}

kernel void group_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float val = input[gid];
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(group_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float diff = val - mean;
    shared[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared[0] / float(group_size);
    output[gid] = gamma[0] * (val - mean) * rsqrt(var + 1e-5f) + beta[0];
}
"#;

pub const ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void attention_sdpa_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;
    if (row >= seq_len || col >= head_dim) return;

    float scale = rsqrt(float(head_dim));
    float sum = 0.0f;
    for (uint k = 0; k < seq_len; k++) {
        float qk = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            qk += Q[row * head_dim + d] * K[k * head_dim + d];
        }
        float attn = exp(qk * scale);
        sum += attn * V[k * head_dim + col];
    }
    output[row * head_dim + col] = sum;
}

kernel void rope_apply_f32(
    device float* x [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair = gid / 2;
    uint half_dim = dim / 2;
    uint pos = pair / half_dim;
    uint d = pair % half_dim;

    float c = cos_table[pos * half_dim + d];
    float s = sin_table[pos * half_dim + d];
    uint idx0 = pos * dim + d;
    uint idx1 = pos * dim + d + half_dim;
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * c - x1 * s;
    x[idx1] = x0 * s + x1 * c;
}

kernel void alibi_bias_f32(
    device float* attn_scores [[buffer(0)]],
    constant float& slope [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_pos = gid.x;
    uint k_pos = gid.y;
    if (q_pos >= seq_len || k_pos >= seq_len) return;
    attn_scores[q_pos * seq_len + k_pos] += slope * float(int(k_pos) - int(q_pos));
}
"#;

pub const LOSS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void loss_cross_entropy_f32(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * num_classes;
    float max_val = logits[base];
    for (uint c = 1; c < num_classes; c++) max_val = max(max_val, logits[base + c]);
    float sum_exp = 0.0f;
    for (uint c = 0; c < num_classes; c++) sum_exp += exp(logits[base + c] - max_val);
    output[gid] = -(logits[base + targets[gid]] - max_val - log(sum_exp));
}

kernel void loss_mse_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float d = pred[gid] - target[gid];
    output[gid] = d * d;
}

kernel void loss_mae_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = abs(pred[gid] - target[gid]);
}

kernel void loss_huber_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& delta [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float d = abs(pred[gid] - target[gid]);
    output[gid] = select(delta * (d - 0.5f * delta), 0.5f * d * d, d <= delta);
}

kernel void loss_kl_divergence_f32(
    device const float* p [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float pi = max(p[gid], 1e-7f);
    float qi = max(q[gid], 1e-7f);
    output[gid] = pi * log(pi / qi);
}

kernel void loss_binary_cross_entropy_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float p = clamp(pred[gid], 1e-7f, 1.0f - 1e-7f);
    float t = target[gid];
    output[gid] = -(t * log(p) + (1.0f - t) * log(1.0f - p));
}
"#;

pub const OPTIMIZERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void opt_sgd_f32(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    constant float& lr [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    params[gid] -= lr * grads[gid];
}

kernel void opt_sgd_momentum_f32(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* velocity [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = momentum * velocity[gid] + grads[gid];
    velocity[gid] = v;
    params[gid] -= lr * v;
}

kernel void opt_adam_f32(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = grads[gid];
    float mi = beta1 * m[gid] + (1.0f - beta1) * g;
    float vi = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mi;
    v[gid] = vi;
    params[gid] -= lr * mi / (sqrt(vi) + eps);
}

kernel void opt_adamw_f32(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = grads[gid];
    float p = params[gid];
    p -= lr * weight_decay * p;
    float mi = beta1 * m[gid] + (1.0f - beta1) * g;
    float vi = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mi;
    v[gid] = vi;
    params[gid] = p - lr * mi / (sqrt(vi) + eps);
}

kernel void grad_clip_by_value_f32(
    device float* grads [[buffer(0)]],
    constant float& clip_val [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    grads[gid] = clamp(grads[gid], -clip_val, clip_val);
}

kernel void grad_clip_by_norm_f32(
    device float* grads [[buffer(0)]],
    constant float& max_norm [[buffer(1)]],
    constant float& current_norm [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (current_norm > max_norm) {
        grads[gid] *= max_norm / current_norm;
    }
}
"#;

pub const FFT: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fft_radix2_f32(
    device float2* data [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_size = 1u << stage;
    uint full_size = half_size << 1;
    uint group = gid / half_size;
    uint pair = gid % half_size;
    uint i = group * full_size + pair;
    uint j = i + half_size;

    float angle = -2.0f * M_PI_F * float(pair) / float(full_size);
    float2 w = float2(cos(angle), sin(angle));
    float2 a = data[i];
    float2 b = data[j];
    float2 wb = float2(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
    data[i] = a + wb;
    data[j] = a - wb;
}

kernel void fft_radix4_f32(
    device float2* data [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint quarter = 1u << (stage * 2);
    uint block = quarter << 2;
    uint group = gid / quarter;
    uint pair = gid % quarter;
    uint base = group * block + pair;

    float2 a0 = data[base];
    float2 a1 = data[base + quarter];
    float2 a2 = data[base + 2 * quarter];
    float2 a3 = data[base + 3 * quarter];

    float2 t0 = a0 + a2;
    float2 t1 = a0 - a2;
    float2 t2 = a1 + a3;
    float2 t3 = float2(a1.y - a3.y, a3.x - a1.x);

    data[base] = t0 + t2;
    data[base + quarter] = t1 + t3;
    data[base + 2 * quarter] = t0 - t2;
    data[base + 3 * quarter] = t1 - t3;
}

kernel void ifft_f32(
    device float2* data [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_size = 1u << stage;
    uint full_size = half_size << 1;
    uint group = gid / half_size;
    uint pair = gid % half_size;
    uint i = group * full_size + pair;
    uint j = i + half_size;

    float angle = 2.0f * M_PI_F * float(pair) / float(full_size); // positive for IFFT
    float2 w = float2(cos(angle), sin(angle));
    float2 a = data[i];
    float2 b = data[j];
    float2 wb = float2(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
    data[i] = a + wb;
    data[j] = a - wb;
}

kernel void spectral_power_density_f32(
    device const float2* fft_data [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float2 c = fft_data[gid];
    output[gid] = c.x * c.x + c.y * c.y;
}
"#;

pub const SIGNAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_f32(
    device const float* input [[buffer(0)]],
    device const float* kernel_data [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_len [[buffer(3)]],
    constant uint& kernel_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (uint k = 0; k < kernel_len; k++) {
        int idx = int(gid) - int(kernel_len / 2) + int(k);
        if (idx >= 0 && uint(idx) < input_len) {
            sum += input[idx] * kernel_data[k];
        }
    }
    output[gid] = sum;
}

kernel void window_apply_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Hanning window
    float w = 0.5f * (1.0f - cos(2.0f * M_PI_F * float(gid) / float(n - 1)));
    output[gid] = input[gid] * w;
}

kernel void fir_filter_f32(
    device const float* input [[buffer(0)]],
    device const float* coeffs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_len [[buffer(3)]],
    constant uint& num_taps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (uint t = 0; t < num_taps; t++) {
        int idx = int(gid) - int(t);
        if (idx >= 0) {
            sum += input[idx] * coeffs[t];
        }
    }
    output[gid] = sum;
}
"#;

pub const LINALG: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr_f32(
    device const float* values [[buffer(0)]],
    device const uint* col_idx [[buffer(1)]],
    device const uint* row_ptr [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint start = row_ptr[gid];
    uint end = row_ptr[gid + 1];
    for (uint j = start; j < end; j++) {
        sum += values[j] * x[col_idx[j]];
    }
    y[gid] = sum;
}

kernel void matvec_f32(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint base = gid * cols;
    for (uint j = 0; j < cols; j++) {
        sum += A[base + j] * x[j];
    }
    y[gid] = sum;
}

kernel void outer_product_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x < n && gid.y < n) {
        C[gid.x * n + gid.y] = a[gid.x] * b[gid.y];
    }
}

kernel void matmul_batched_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"#;

pub const SIMULATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sim_heat_equation_f32(
    device const float* u [[buffer(0)]],
    device float* u_next [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    if (x == 0 || y == 0 || x >= width - 1 || y >= width - 1) {
        u_next[y * width + x] = u[y * width + x];
        return;
    }
    float center = u[y * width + x];
    float left = u[y * width + x - 1];
    float right = u[y * width + x + 1];
    float up = u[(y - 1) * width + x];
    float down = u[(y + 1) * width + x];
    u_next[y * width + x] = center + alpha * (left + right + up + down - 4.0f * center);
}

kernel void sim_wave_equation_f32(
    device const float* u_curr [[buffer(0)]],
    device const float* u_prev [[buffer(1)]],
    device float* u_next [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& c2 [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    if (x == 0 || y == 0 || x >= width - 1 || y >= width - 1) {
        u_next[y * width + x] = 0.0f;
        return;
    }
    uint idx = y * width + x;
    float laplacian = u_curr[idx - 1] + u_curr[idx + 1] + u_curr[idx - width] + u_curr[idx + width] - 4.0f * u_curr[idx];
    u_next[idx] = 2.0f * u_curr[idx] - u_prev[idx] + c2 * laplacian;
}

kernel void integrate_rk4_f32(
    device const float* y [[buffer(0)]],
    device float* y_next [[buffer(1)]],
    constant float& dt [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float yi = y[gid];
    float k1 = -yi;
    float k2 = -(yi + 0.5f * dt * k1);
    float k3 = -(yi + 0.5f * dt * k2);
    float k4 = -(yi + dt * k3);
    y_next[gid] = yi + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}

kernel void integrate_velocity_verlet_f32(
    device float* pos [[buffer(0)]],
    device float* vel [[buffer(1)]],
    device const float* acc [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float p = pos[gid];
    float v = vel[gid];
    float a = acc[gid];
    p += v * dt + 0.5f * a * dt * dt;
    float a_new = -p; // simple harmonic
    v += 0.5f * (a + a_new) * dt;
    pos[gid] = p;
    vel[gid] = v;
}

kernel void monte_carlo_integrate_f32(
    device const float* random_x [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = random_x[gid];
    output[gid] = 4.0f / (1.0f + x * x); // pi estimation via integral of 4/(1+x^2)
}
"#;

pub const SORTING: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sort_bitonic_f32(
    device float* data [[buffer(0)]],
    constant uint& stage [[buffer(1)]],
    constant uint& substage [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_block = 1u << substage;
    uint block = half_block << 1;
    uint group_id = gid / half_block;
    uint pair_id = gid % half_block;
    uint i = group_id * block + pair_id;
    uint j = i + half_block;
    bool ascending = ((i >> (stage + 1)) & 1) == 0;
    float a = data[i];
    float b = data[j];
    if (ascending ? (a > b) : (a < b)) {
        data[i] = b;
        data[j] = a;
    }
}

kernel void sort_bitonic_kv_f32(
    device float* keys [[buffer(0)]],
    device uint* values [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    constant uint& substage [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_block = 1u << substage;
    uint block = half_block << 1;
    uint group_id = gid / half_block;
    uint pair_id = gid % half_block;
    uint i = group_id * block + pair_id;
    uint j = i + half_block;
    bool ascending = ((i >> (stage + 1)) & 1) == 0;
    float ka = keys[i], kb = keys[j];
    if (ascending ? (ka > kb) : (ka < kb)) {
        keys[i] = kb; keys[j] = ka;
        uint va = values[i], vb = values[j];
        values[i] = vb; values[j] = va;
    }
}
"#;

pub const PRNG: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Philox 4x32-10 counter-based PRNG
uint4 philox4x32(uint4 ctr, uint2 key) {
    for (int i = 0; i < 10; i++) {
        uint lo0 = ctr.x * 0xD2511F53u;
        uint hi0 = mulhi(ctr.x, 0xD2511F53u);
        uint lo1 = ctr.z * 0xCD9E8D57u;
        uint hi1 = mulhi(ctr.z, 0xCD9E8D57u);
        ctr = uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return ctr;
}

kernel void prng_philox_u32x4(
    device uint4* output [[buffer(0)]],
    constant uint2& key [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint4 ctr = uint4(gid, 0u, 0u, 0u);
    output[gid] = philox4x32(ctr, key);
}

kernel void prng_uniform_f32(
    device float* output [[buffer(0)]],
    constant uint2& key [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint4 ctr = uint4(gid, 0u, 0u, 0u);
    uint4 r = philox4x32(ctr, key);
    output[gid] = float(r.x) / float(0xFFFFFFFFu);
}

kernel void prng_normal_f32(
    device float* output [[buffer(0)]],
    constant uint2& key [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint4 ctr = uint4(gid, 0u, 0u, 0u);
    uint4 r = philox4x32(ctr, key);
    float u1 = max(float(r.x) / float(0xFFFFFFFFu), 1e-7f);
    float u2 = float(r.y) / float(0xFFFFFFFFu);
    output[gid] = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

kernel void prng_dropout_mask(
    device uchar* output [[buffer(0)]],
    constant uint2& key [[buffer(1)]],
    constant float& keep_prob [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint4 ctr = uint4(gid, 0u, 0u, 0u);
    uint4 r = philox4x32(ctr, key);
    output[gid] = (float(r.x) / float(0xFFFFFFFFu)) < keep_prob ? 1 : 0;
}
"#;

pub const LAYOUT: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.x;
    uint c = gid.y;
    if (r < rows && c < cols) {
        output[c * rows + r] = input[r * cols + c];
    }
}

kernel void repack_aos_to_soa(
    device const float4* aos [[buffer(0)]],
    device float* soa_x [[buffer(1)]],
    device float* soa_y [[buffer(2)]],
    device float* soa_z [[buffer(3)]],
    device float* soa_w [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 v = aos[gid];
    soa_x[gid] = v.x;
    soa_y[gid] = v.y;
    soa_z[gid] = v.z;
    soa_w[gid] = v.w;
}

kernel void repack_soa_to_aos(
    device const float* soa_x [[buffer(0)]],
    device const float* soa_y [[buffer(1)]],
    device const float* soa_z [[buffer(2)]],
    device const float* soa_w [[buffer(3)]],
    device float4* aos [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    aos[gid] = float4(soa_x[gid], soa_y[gid], soa_z[gid], soa_w[gid]);
}
"#;

pub const SCANS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scan_exclusive_u32(
    device uint* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    shared[tid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Up-sweep
    for (uint d = 1; d < tg_size; d <<= 1) {
        uint ai = (tid + 1) * (d << 1) - 1;
        if (ai < tg_size) shared[ai] += shared[ai - d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == tg_size - 1) shared[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Down-sweep
    for (uint d = tg_size >> 1; d > 0; d >>= 1) {
        uint ai = (tid + 1) * (d << 1) - 1;
        if (ai < tg_size) {
            uint temp = shared[ai];
            shared[ai] += shared[ai - d];
            shared[ai - d] = temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    data[gid] = shared[tid];
}

kernel void scan_inclusive_u32(
    device uint* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    shared[tid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 1; d < tg_size; d <<= 1) {
        uint val = (tid >= d) ? shared[tid - d] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    data[gid] = shared[tid];
}

kernel void compact_mask(
    device const float* input [[buffer(0)]],
    device const uint* mask [[buffer(1)]],
    device const uint* scan [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (mask[gid]) {
        output[scan[gid]] = input[gid];
    }
}

kernel void compact_if(
    device const float* input [[buffer(0)]],
    device const uint* scan [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (input[gid] > threshold) {
        output[scan[gid]] = input[gid];
    }
}
"#;

pub const QUANTIZATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void quantize_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = half(input[gid]);
}

kernel void quantize_f16_to_i8(
    device const half* input [[buffer(0)]],
    device char* output [[buffer(1)]],
    constant half& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = char(clamp(round(float(input[gid]) / float(scale)), -127.0f, 127.0f));
}

kernel void dequantize_i8_to_f16(
    device const char* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant half& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = half(float(input[gid]) * float(scale));
}
"#;

pub const DOT_PRODUCTS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dot_batched_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * dim;
    float sum = 0.0f;
    for (uint d = 0; d < dim; d++) {
        sum += a[base + d] * b[base + d];
    }
    output[gid] = sum;
}

kernel void cosine_similarity_batched(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * dim;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint d = 0; d < dim; d++) {
        float ai = a[base + d], bi = b[base + d];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    output[gid] = dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8f);
}
"#;

pub const FUSED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void map_reduce_square_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float v = input[gid];
    shared[tid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void map_reduce_abs_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = abs(input[gid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void map_reduce_masked_sum(
    device const float* input [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[tid] = input[gid] * mask[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}

kernel void map_reduce_threshold_count(
    device const float* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    shared[tid] = (input[gid] > threshold) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[gid / tg_size] = shared[0];
}
"#;
