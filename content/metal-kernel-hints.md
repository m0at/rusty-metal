# Metal Kernel Optimization Reference

Production Metal GPU kernel catalog for Apple Silicon. Maps workload patterns to specific kernels with performance characteristics, routing logic, fusion opportunities, implementation details, and pipeline recipes.

Covers: reductions, elementwise ops, scans, compaction, quantization, layout, statistics, ML (activations, normalization, attention, loss, optimizers), FFT & signal processing, linear algebra, simulation & physics, PRNG.

---

## Kernel Catalog

109 production kernels organized by domain. GFLOPS estimated on M-series Apple Silicon.

### Reductions

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| REDUCE_SUM | `reduce_sum` | 450 | memory | |
| REDUCE_MEAN | `reduce_mean` | 450 | memory | |
| REDUCE_MIN | `reduce_min` | 450 | memory | |
| REDUCE_MAX | `reduce_max` | 450 | memory | |
| REDUCE_L2 | `reduce_l2` | 450 | memory | |
| REDUCE_VAR | `reduce_var` | 420 | memory | Welford's algorithm |
| REDUCE_STDDEV | `reduce_stddev` | 420 | memory | Variance + sqrt |
| REDUCE_HISTOGRAM | `reduce_histogram_u32` | 350 | memory | Atomic shared memory |
| REDUCE_ARGMAX | `reduce_argmax_f32` | 440 | memory | (value, index) pairs |
| REDUCE_ARGMIN | `reduce_argmin_f32` | 440 | memory | (value, index) pairs |

### Correlation & Covariance

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| REDUCE_CORRELATION | `reduce_correlation_f32` | 400 | memory | Single-pass Pearson |
| REDUCE_COVARIANCE | `reduce_covariance_f32` | 390 | memory | |
| REDUCE_WEIGHTED_SUM | `reduce_weighted_sum_f32` | 470 | memory | Fused weight*value |
| REDUCE_WEIGHTED_MEAN | `reduce_weighted_mean_f32` | 460 | memory | |
| REDUCE_COVARIANCE_MATRIX | `reduce_covariance_matrix_f32` | 450 | compute | Batched pairwise |
| REDUCE_PEARSON_PAIRWISE | `reduce_pearson_pairwise_f32` | 430 | compute | |

### Elementwise Unary

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| MAP_EXP | `map_exp` | 380 | compute | |
| MAP_LOG | `map_log` | 380 | compute | |
| MAP_SIGMOID | `map_sigmoid` | 380 | compute | |
| MAP_TANH | `map_tanh` | 380 | compute | |
| MAP_SOFTPLUS | `map_softplus` | 360 | compute | |
| MAP_CLAMP | `map_clamp` | 500 | memory | |
| MAP_ABS | `map_abs` | 500 | memory | |

### ML Activations

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| MAP_RELU | `map_relu` | 500 | memory | max(0,x) |
| MAP_LEAKY_RELU | `map_leaky_relu` | 490 | memory | Branchless select |
| MAP_ELU | `map_elu` | 360 | compute | exp() for x<0 |
| MAP_GELU | `map_gelu` | 340 | compute | Tanh approximation |
| MAP_SILU | `map_silu` | 370 | compute | x*sigmoid(x), LLaMA/Mistral |
| MAP_MISH | `map_mish` | 330 | compute | x*tanh(softplus(x)) |

### Elementwise Binary

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| MAP_ADD | `map_add` | 500 | memory | |
| MAP_MUL | `map_mul` | 500 | memory | |
| MAP_DIV | `map_div` | 480 | memory | |
| MAP_FMA | `map_fma` | 520 | compute | |
| MAP_COMPARE | `map_compare` | 500 | memory | |

### Scans

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| SCAN_EXCLUSIVE | `scan_exclusive_u32` | 280 | memory | |
| SCAN_INCLUSIVE | `scan_inclusive_u32` | 280 | memory | |

### Compaction

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| COMPACT_IF | `compact_if` | 300 | memory | |
| COMPACT_MASK | `compact_mask` | 300 | memory | |

### Dot Products & Similarity

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| DOT_BATCHED | `dot_batched_f32` | 520 | compute | |
| COSINE_SIMILARITY | `cosine_similarity_batched` | 500 | compute | |

### Quantization

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| QUANTIZE_F16_TO_I8 | `quantize_f16_to_i8` | 600 | memory | |
| DEQUANTIZE_I8_TO_F16 | `dequantize_i8_to_f16` | 600 | memory | |
| QUANTIZE_F32_TO_F16 | `quantize_f32_to_f16` | 650 | memory | |

### Layout Transforms

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| TRANSPOSE_2D | `transpose_2d` | 400 | memory | |
| REPACK_AOS_TO_SOA | `repack_aos_to_soa` | 350 | memory | |
| REPACK_SOA_TO_AOS | `repack_soa_to_aos` | 350 | memory | |

### Fused Map-Reduce

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| MAP_REDUCE_SQUARE_SUM | `map_reduce_square_sum` | 480 | fused | |
| MAP_REDUCE_ABS_SUM | `map_reduce_abs_sum` | 470 | fused | |
| MAP_REDUCE_MASKED_SUM | `map_reduce_masked_sum` | 460 | fused | |
| MAP_REDUCE_THRESHOLD_COUNT | `map_reduce_threshold_count` | 450 | fused | |

### Advanced Statistics & Sorting

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| TOPK_SELECT | `topk_select_f32` | 320 | compute | O(n log k) partial sort |
| SORT_RADIX | `sort_radix_u32` | 280 | memory | 4-bit radix, 8 passes for u32 |
| SORT_BITONIC | `sort_bitonic_f32` | 250 | compute | O(n log²n), power-of-2 |
| SORT_BITONIC_KV | `sort_bitonic_kv_f32` | 220 | compute | 2x memory traffic |
| MEDIAN_SELECT | `median_select_f32` | 300 | memory | |
| PERCENTILE_SELECT | `percentile_select_f32` | 300 | memory | |

### Softmax

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| SOFTMAX_STABLE | `softmax_stable_f32` | 400 | memory | Fused max-exp-sum-div |
| LOG_SOFTMAX | `log_softmax_f32` | 400 | memory | x - max - log(sum_exp) |
| SOFTMAX_ONLINE | `softmax_online_f32` | 420 | memory | Single-pass for Flash Attention |

### Normalization

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| LAYER_NORM | `layer_norm_f32` | 380 | memory | Welford mean+var + normalize+scale+shift |
| BATCH_NORM | `batch_norm_f32` | 360 | memory | Per-channel across batch |
| RMS_NORM | `rms_norm_f32` | 400 | memory | 30% fewer ops than LayerNorm |
| GROUP_NORM | `group_norm_f32` | 350 | memory | Independent per group |

### Attention & Positional Encoding

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| ATTENTION_SDPA | `attention_sdpa_f32` | 520 | compute | O(N²) memory, seq<2048 |
| ATTENTION_FLASH | `attention_flash_f16` | 580 | compute | O(N) memory, tiled |
| ATTENTION_MQA | `attention_mqa_f16` | 600 | compute | Shared K,V across heads |
| ATTENTION_GQA | `attention_gqa_f16` | 590 | compute | Grouped-query |
| ROPE_APPLY | `rope_apply_f32` | 450 | memory | Sin/cos rotation on dim pairs |
| ROPE_PRECOMPUTE | `rope_precompute_f32` | 300 | compute | One-time frequency table |
| ALIBI_BIAS | `alibi_bias_f32` | 400 | memory | Linear bias, no params |

### Loss Functions

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| LOSS_CROSS_ENTROPY | `loss_cross_entropy_f32` | 380 | compute | Fused log-softmax + NLL |
| LOSS_MSE | `loss_mse_f32` | 470 | memory | Fused (pred-target)² + reduce |
| LOSS_MAE | `loss_mae_f32` | 470 | memory | Fused abs(diff) + reduce |
| LOSS_HUBER | `loss_huber_f32` | 450 | memory | Smooth L1, branchless select |
| LOSS_KL_DIV | `loss_kl_divergence_f32` | 360 | compute | Safe log, handles zeros |
| LOSS_BCE | `loss_binary_cross_entropy_f32` | 370 | compute | Clamped log for stability |

### Optimizers & Training

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| OPT_SGD | `opt_sgd_f32` | 480 | memory | p -= lr * g |
| OPT_SGD_MOMENTUM | `opt_sgd_momentum_f32` | 440 | memory | Read/write velocity state |
| OPT_ADAM | `opt_adam_f32` | 380 | memory | 5 buffers: param,grad,m,v,out |
| OPT_ADAMW | `opt_adamw_f32` | 370 | memory | Decoupled weight decay |
| GRAD_CLIP_NORM | `grad_clip_by_norm_f32` | 440 | memory | Two-pass: L2 norm then scale |
| GRAD_CLIP_VALUE | `grad_clip_by_value_f32` | 490 | memory | Single-pass clamp |

### FFT & Spectral

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| FFT_RADIX2 | `fft_radix2_f32` | 300 | compute | O(N log N), power-of-2 |
| FFT_RADIX4 | `fft_radix4_f32` | 350 | compute | 25% fewer passes than radix-2 |
| IFFT | `ifft_f32` | 300 | compute | Conjugate twiddles + 1/N scale |
| FFT_REAL | `fft_real_f32` | 320 | compute | Hermitian symmetry, ~2x faster |
| SPECTRAL_POWER | `spectral_power_density_f32` | 480 | memory | \|X[k]\|² = re²+im² |

### Signal Processing

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| CONV1D | `conv1d_f32` | 400 | compute | Direct, kernel_size<64 |
| FFT_CONV1D | `fft_conv1d_f32` | 350 | compute | FFT-based, kernel_size>=64 |
| AUTOCORRELATION | `autocorrelation_f32` | 380 | compute | |
| CROSS_CORRELATION | `cross_correlation_f32` | 400 | compute | |
| WINDOW_APPLY | `window_apply_f32` | 480 | memory | On-the-fly cos() coefficients |
| FIR_FILTER | `fir_filter_f32` | 400 | compute | Reversed coeffs, streaming |

### Linear Algebra

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| SPMV_CSR | `spmv_csr_f32` | 200 | memory | One thread per row |
| MATMUL_BATCHED | `matmul_batched_f32` | 520 | compute | Tiled, one TG per batch |
| MATVEC | `matvec_f32` | 350 | memory | Row dot product |
| TRIANGULAR_SOLVE | `triangular_solve_f32` | 180 | memory | Level-set parallelism |
| OUTER_PRODUCT | `outer_product_f32` | 500 | memory | Trivially parallel |

### Simulation & Physics

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| SIM_HEAT_EQUATION | `sim_heat_equation_f32` | 350 | memory | 5-point stencil, explicit Euler |
| SIM_WAVE_EQUATION | `sim_wave_equation_f32` | 330 | memory | Leapfrog, two time levels |
| SIM_STENCIL_2D | `sim_stencil_2d_f32` | 370 | memory | Configurable coefficients |
| SIM_STENCIL_3D | `sim_stencil_3d_f32` | 300 | memory | High memory pressure |
| INTEGRATE_RK4 | `integrate_rk4_f32` | 400 | compute | 4 evaluations per step |
| INTEGRATE_VERLET | `integrate_velocity_verlet_f32` | 420 | compute | Symplectic, energy-conserving |
| MONTE_CARLO_INTEGRATE | `monte_carlo_integrate_f32` | 450 | compute | Embarrassingly parallel |

### Random Number Generation

| Kernel | Metal Function | GFLOPS | Bound | Notes |
|--------|---------------|--------|-------|-------|
| PRNG_PHILOX | `prng_philox_u32x4` | 550 | compute | Counter-based, 4x u32 output |
| PRNG_UNIFORM_F32 | `prng_uniform_f32` | 520 | compute | Philox -> [0,1) float |
| PRNG_NORMAL_F32 | `prng_normal_f32` | 480 | compute | Box-Muller transform |
| PRNG_DROPOUT_MASK | `prng_dropout_mask` | 500 | compute | Fused Philox + threshold |

---

## Operation Routing

Decision tables for mapping operations to optimal kernels. Conditions reference `data_size` (element count), `precision`, and operation-specific kwargs.

### Vector & Reduction Operations

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `vector_norm` | data_size > 100k | REDUCE_L2 | Direct L2 norm kernel for large vectors |
| `vector_norm` | data_size <= 100k | MAP_REDUCE_SQUARE_SUM | Fused square+sum avoids intermediate buffer; follow with CPU sqrt |
| `sum_of_squares` | any | MAP_REDUCE_SQUARE_SUM | Single fused kernel eliminates memory traffic (2x savings) |
| `cosine_similarity` | data_size > 1000 | COSINE_SIMILARITY | Batched cosine similarity with normalization |
| `cosine_similarity` | data_size <= 1000 | DOT_BATCHED + REDUCE_L2 | Dot product + separate normalization for small batch |
| `filter_threshold` | any | MAP_COMPARE -> COMPACT_MASK | Compare to generate mask, then compact (streaming) |
| `prefix_sum` | any | SCAN_EXCLUSIVE | Two-pass exclusive scan, O(n) work-efficient. Critical for stream compaction and radix sort |

### Statistics

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `histogram` | any | REDUCE_HISTOGRAM | Atomic histogram updates in shared memory. Fixed 256 bins, privatization for conflict reduction |
| `variance` | any | REDUCE_VAR | Welford's online algorithm, numerically stable. Avoids catastrophic cancellation from naive E[X²]-E[X]² |
| `standard_deviation` | any | REDUCE_STDDEV | Variance + final sqrt, single-pass Welford |
| `argmax` | any | REDUCE_ARGMAX | Reduction tree carrying (value, index) pairs. Deterministic tie-breaking (first occurrence) |
| `argmin` | any | REDUCE_ARGMIN | Reduction tree with (value, index), reversed comparator |
| `describe_stats` | any | REDUCE_MEAN + REDUCE_VAR + REDUCE_MIN + REDUCE_MAX + REDUCE_ARGMIN + REDUCE_ARGMAX + REDUCE_HISTOGRAM | Launch all stats kernels in parallel streams. Overlapped execution hides latency |

### Correlation & Covariance

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `correlation` | pairwise=True | REDUCE_PEARSON_PAIRWISE | Batch Pearson correlation matrix, single-pass per pair. Exploit symmetry: only upper triangle |
| `correlation` | pairwise=False | REDUCE_CORRELATION | Single-pass Pearson r with Welford-style stability |
| `covariance` | matrix=True | REDUCE_COVARIANCE_MATRIX | Batched pairwise covariance, symmetric output optimization |
| `covariance` | matrix=False | REDUCE_COVARIANCE | Two-pass covariance with mean precomputation |
| `weighted_sum` | any | REDUCE_WEIGHTED_SUM | Fused weight*value multiply in map phase before reduction |
| `weighted_mean` | any | REDUCE_WEIGHTED_MEAN | Parallel weighted sum and weight sum with single final division |
| `pca_prep` | any | REDUCE_MEAN + REDUCE_COVARIANCE_MATRIX | Mean-center data then compute covariance matrix for PCA |

### Advanced Statistics & Sorting

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `top_k` | k/n < 0.1 (low selectivity) | TOPK_SELECT | Heap-based top-k selection, O(n log k) |
| `top_k` | k/n >= 0.1 (high selectivity) | SORT_RADIX | Full radix sort then slice top k |
| `sort` | n <= 32768 and power-of-2 | SORT_BITONIC | Bitonic sort, in-place, O(n log²n), perfect for GPU |
| `sort` | otherwise | SORT_RADIX | 4-bit radix per pass, 8 passes for 32-bit keys, uses scan infrastructure |
| `median` | data_size > 8192 | MEDIAN_SELECT | Selection without full sort, O(n) average |
| `median` | data_size <= 8192, power-of-2 | SORT_BITONIC | Full sort then extract middle element |
| `median` | data_size <= 8192, not power-of-2 | SORT_RADIX | Full sort then extract middle element |
| `percentile` | any | PERCENTILE_SELECT | Selection for p-th percentile, O(n log k) |

### Quantization

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `quantize_weights` | precision=fp16 | QUANTIZE_F16_TO_I8 | Per-channel int8 quantization, 2x storage reduction |
| `quantize_weights` | precision!=fp16 | QUANTIZE_F32_TO_F16 | Fast fp32->fp16 downcast |

### Layout Transforms

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `transpose` | any | TRANSPOSE_2D | Tiled transpose for cache efficiency, 16x16 optimal for M-series, coalesced writes |
| `aos_to_soa` | any | REPACK_AOS_TO_SOA | Structure layout change for vectorization, better SIMD utilization |

### Activations

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `sigmoid` | any | MAP_SIGMOID | Vectorized sigmoid, consider fusing with adjacent operations |
| `tanh` | any | MAP_TANH | Vectorized tanh, consider fusing with adjacent operations |
| `softplus` | any | MAP_SOFTPLUS | Vectorized softplus, consider fusing with adjacent operations |
| `relu` | any | MAP_RELU | max(0, x). Memory-bound, fuse with preceding conv/matmul if possible |
| `gelu` | any | MAP_GELU | Tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³))). Standard in GPT-2/3, BERT, ViT |
| `silu` / `swish` | any | MAP_SILU | x*sigmoid(x), fused to avoid intermediate. Used in LLaMA, Mistral, Gemma, EfficientNet |
| `leaky_relu` | any | MAP_LEAKY_RELU | Branchless select(alpha*x, x, x>0). Common in GAN discriminators |
| `elu` | any | MAP_ELU | alpha*(exp(x)-1) for x<0, x for x>=0. Smoother than ReLU at zero |
| `mish` | any | MAP_MISH | x*tanh(softplus(x)), numerically stable for large x. YOLOv4, self-regularizing activation |

### Softmax

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `softmax` | any (seq_len <= 8192) | SOFTMAX_STABLE | Fused stable softmax (max-subtract-exp-sum-normalize). **DANGER: Never decompose into map_exp->reduce_sum->map_div (overflows!)** |
| `softmax` | seq_len > 8192 | SOFTMAX_STABLE | Same kernel, but two-kernel approach for large sequences |
| `log_softmax` | any | LOG_SOFTMAX | Fused log-softmax: x - max(x) - log(sum(exp(x-max))). More stable than log(softmax(x)) |

### Normalization

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `layer_norm` | any | LAYER_NORM | Fused LayerNorm. Single-pass Welford mean+var, then normalize+scale+shift |
| `batch_norm` | any | BATCH_NORM | Per-channel stats across batch dimension. Training: batch-wide reduction. Inference: cached running stats |
| `rms_norm` | any | RMS_NORM | 30% cheaper than LayerNorm (no mean subtraction). Formula: y = x / RMS(x) * gamma. Used in LLaMA, Mistral |
| `group_norm` | any | GROUP_NORM | Independent normalization per group. Common in vision models, batch-size independent |

### Attention

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `attention` / `self_attention` / `cross_attention` | seq_length < 2048 | ATTENTION_SDPA | Naive SDPA, O(N²) memory acceptable |
| `attention` / `self_attention` / `cross_attention` | seq_length >= 2048 | ATTENTION_FLASH | Flash attention, O(N) memory, 2-4x faster |
| _(with position_encoding=rope)_ | any | + ROPE_PRECOMPUTE + ROPE_APPLY | Precompute frequencies + apply rotation to Q,K |
| _(with position_encoding=alibi)_ | any | + ALIBI_BIAS | Add linear bias m*(i-j), parameter-free |
| `rope` | any | ROPE_PRECOMPUTE + ROPE_APPLY | Rotary position embedding: sin/cos rotation on consecutive dim pairs |

### Loss Functions

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `cross_entropy` / `cross_entropy_loss` | any | LOSS_CROSS_ENTROPY | Fused log-softmax + NLL with log-sum-exp trick. Supports class weights and ignore_index |
| `mse_loss` / `mean_squared_error` | any | LOSS_MSE | Fused (pred-target)² + reduction, no intermediate buffer |
| `mae_loss` / `l1_loss` / `mean_absolute_error` | any | LOSS_MAE | Fused abs(pred-target) + reduction |
| `huber_loss` / `smooth_l1` | any | LOSS_HUBER | Smooth L1 with branchless select(), configurable delta |
| `kl_divergence` / `kl_div` | any | LOSS_KL_DIV | target * (log(target) - log(pred)), handles zeros with epsilon clamp |
| `binary_cross_entropy` / `bce` | any | LOSS_BCE | -(y*log(p) + (1-y)*log(1-p)), clamped for log stability |

### Optimizers

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `sgd_step` | momentum > 0 | OPT_SGD_MOMENTUM | v = mu*v + g; p -= lr*v. Requires persistent velocity buffer |
| `sgd_step` | momentum = 0 | OPT_SGD | Simple SGD: p -= lr * g |
| `adam_step` | weight_decay > 0 | OPT_ADAMW | Decoupled weight decay + Adam update in single kernel. Fused: read param+grad+m+v, write param+m+v |
| `adam_step` | weight_decay = 0 | OPT_ADAM | Fused m/v update + bias correction + param step. Fused: read param+grad+m+v, write param+m+v |
| `gradient_clip` | clip_type=norm | REDUCE_L2 + GRAD_CLIP_NORM | Two-pass: compute global L2 norm, then scale gradients. Can fuse with optimizer step if norm pre-computed |
| `gradient_clip` | clip_type=value | GRAD_CLIP_VALUE | Element-wise clamp(grad, -max_val, max_val), single-pass |

### FFT & Spectral

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `fft` / `frequency_analysis` | real_input=True (default) | FFT_REAL | Real FFT exploiting Hermitian symmetry, ~2x speedup. N-point real FFT via N/2-point complex FFT |
| `fft` / `frequency_analysis` | complex, n is power-of-4 and n >= 256 | FFT_RADIX4 | Fewer stages than radix-2 |
| `fft` / `frequency_analysis` | complex, n is power-of-2 | FFT_RADIX2 | Standard radix-2 FFT |
| `fft` / `frequency_analysis` | complex, n is not power-of-2 | FFT_RADIX2 | Pad to next power-of-2, then radix-2 FFT |
| `ifft` | any | IFFT | Inverse FFT: conjugate twiddles + 1/N scaling |
| `spectral_density` / `power_spectrum` | any | FFT_REAL + SPECTRAL_POWER | FFT then \|X[k]\|² for power spectral density |
| `convolution_fft` | any | FFT_RADIX2 + MAP_MUL + IFFT | FFT-based convolution: FFT(x) * FFT(h) -> IFFT |

### Signal Processing

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `conv1d` | kernel_size > 64 | FFT_CONV1D | FFT-based convolution for large kernel |
| `conv1d` | kernel_size <= 64 | CONV1D | Direct 1D convolution, sliding window in shared memory |
| `autocorrelation` | data_size > 1024 | AUTOCORRELATION | FFT-based autocorrelation (Wiener-Khinchin theorem): IFFT(\|FFT(signal)\|²) |
| `autocorrelation` | data_size <= 1024 | AUTOCORRELATION | Direct autocorrelation for short signals |
| `cross_correlation` | any | CROSS_CORRELATION | Cross-correlation with optional zero-mean normalization |
| `window` / `windowing` | any | WINDOW_APPLY | Apply window (hann, hamming, etc.) with on-the-fly coefficient generation |
| `fir_filter` / `signal_filter` | kernel_size > 64 | FFT_CONV1D | FFT-based FIR for large kernel |
| `fir_filter` / `signal_filter` | kernel_size <= 64 | FIR_FILTER | Direct FIR with streaming optimization |

### Linear Algebra

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `spmv` / `sparse_matvec` | density < 5% | SPMV_CSR | SpMV CSR for sparse matrix. One thread per row, segmented reduction for long rows |
| `spmv` / `sparse_matvec` | density >= 5% | MATVEC | Dense matvec faster above 5% density |
| `batched_matmul` | batch_size > 1 and max_dim < 512 | MATMUL_BATCHED | Custom batched matmul beats MPS for small matrices. One threadgroup per batch element, shared memory tiling |
| `batched_matmul` | otherwise | MATMUL_BATCHED | Batched matmul (consider MPS for large individual matrices) |
| `matvec` | any | MATVEC | Each thread computes one output element via row dot product |
| `triangular_solve` / `linear_system` | any | TRIANGULAR_SOLVE | Level-set parallelism: identify independent rows, solve in waves |
| `outer_product` | any | OUTER_PRODUCT | Trivially parallel: one thread per output element |

### Simulation & Physics

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `heat_equation` / `diffusion` | any | SIM_HEAT_EQUATION | Explicit Euler with 5-point stencil. CFL stability: dt < dx²/(4*alpha). Warning emitted if dt may be unstable |
| `wave_equation` | any | SIM_WAVE_EQUATION | Leapfrog with two time levels, second-order central differences. CFL: c*dt/dx must be < 1. Warning if unstable |
| `stencil` | dimensions=2 | SIM_STENCIL_2D | General 2D stencil, configurable coefficients and offsets. Threadgroup memory tiling for large stencil radii |
| `stencil` | dimensions=3 | SIM_STENCIL_3D | General 3D stencil. Threadgroup memory tiling for large stencil radii |
| `ode_integrate` / `rk4` | any | INTEGRATE_RK4 | Fourth-order Runge-Kutta, 4 evaluations per step, O(dt⁴) error. Each thread = one equation or trajectory |
| `verlet` / `nbody_step` | any | INTEGRATE_VERLET | Velocity Verlet: symplectic, energy-conserving for Hamiltonian systems. Update order: position -> force eval -> velocity |
| `monte_carlo` | any | MONTE_CARLO_INTEGRATE | Monte Carlo integration. Error O(1/sqrt(N)), dimension-independent. Use PRNG_PHILOX for RNG |

### Random Number Generation

| Operation | Condition | Kernel(s) | Rationale |
|-----------|-----------|-----------|-----------|
| `random_uniform` / `rand` / `uniform_random` | any | PRNG_UNIFORM_F32 | Philox-based uniform PRNG, maps u32->[0,1) via fixed-point. Counter-based Philox4x32-10 |
| `random_normal` / `randn` / `gaussian_random` | any | PRNG_NORMAL_F32 | Box-Muller transform on Philox stream, 2 normals per 2 uniforms |
| `dropout_mask` / `bernoulli_mask` | any | PRNG_DROPOUT_MASK | Fused Philox + threshold for binary mask. Output u8 for memory efficiency |
| `random_init` / `prng_init` | any | PRNG_PHILOX | Raw Philox counter stream (4x u32 per thread). Statistically independent streams via counter offset |

---

## Fusion Patterns

Patterns where consecutive operations should be fused into single kernels for better performance.

### Two-Operation Fusions

| Pattern | Fused Kernel | Speedup | Memory Savings | Priority |
|---------|-------------|---------|----------------|----------|
| square -> sum | `map_reduce_square_sum` | 1.8-2.2x | 50% (no intermediate buffer) | normal |
| abs -> sum | `map_reduce_abs_sum` | 1.8-2.2x | 50% (no intermediate buffer) | normal |
| compare -> count | `map_reduce_threshold_count` | 1.5x | — | normal |
| activation (sigmoid/tanh/gelu/silu/relu) -> normalize/scale/layer_norm | _(custom fused)_ | 1.3x | — | normal |
| activation (gelu/silu/swish/relu) -> dropout | _(custom fused)_ | 1.4x | — | normal |
| loss/cross_entropy -> grad/backward | _(custom fused)_ | 1.5-2x | — | normal |
| grad clip -> adam/sgd | _(custom fused)_ | 1.3x | — | normal |
| fft -> magnitude/abs_squared/power | `spectral_power_density_f32` | 1.4x | 50% (skip complex buffer) | normal |

### Multi-Operation Fusions

| Pattern | Fused Kernel | Speedup | Priority |
|---------|-------------|---------|----------|
| max -> exp -> sum -> div **(SOFTMAX)** | `softmax_stable_f32` | 2-3x | **critical** |
| reduce_max -> map_exp -> reduce_sum -> map_div **(SOFTMAX)** | `softmax_stable_f32` | 2-3x | **critical** |
| mean -> var -> normalize -> scale **(LAYERNORM)** | `layer_norm_f32` | 2-3x | normal |
| reduce_mean -> reduce_var -> sub_div -> mul/affine **(LAYERNORM)** | `layer_norm_f32` | 2-3x | normal |
| matmul -> scale -> mask -> softmax -> matmul **(ATTENTION)** | `attention_flash_f16` | 2-4x (long sequences) | **critical** |

---

## Implementation Hints

Per-domain algorithm details, formulas, optimization notes, and Metal-specific guidance.

### Reductions

**REDUCE_SUM**
- Two-stage reduction: block-wise reduction in threadgroups, then final reduction across blocks
- Threadgroup size: 256
- Use shuffle operations for warp reduction

**MAP_SIGMOID**
- Formula: `1/(1+exp(-x))`
- Use fast_exp approximation for fp16
- Vectorization: process 4 elements per thread
- Max error 1e-6 for fp32

### Scans

**SCAN_EXCLUSIVE**
- Blelloch scan algorithm, O(n) work-efficient
- Stages: up-sweep, down-sweep
- Use cases: stream compaction, radix sort, histogram

### Dot Products

**DOT_BATCHED**
- Tiled dot products, tile size 64
- Coalesce memory access, use shared memory
- One thread per dot product

### Quantization

**QUANTIZE_F16_TO_I8**
- Per-channel scaling: `scale = max(abs(channel)) / 127`
- Compute scales in first pass, quantize in second
- Typically <1% accuracy loss for inference

### Statistics

**REDUCE_VAR**
- Welford's online algorithm with two-stage block reduction
- Per-thread Welford state (count, mean, M2) in registers, block-level reduction merges (n, mean, M2) tuples
- Merge formula: `M2_combined = M2_a + M2_b + delta² * n_a * n_b / (n_a + n_b)`
- Stability: avoids catastrophic cancellation, relative error ~1e-7 for fp32
- Threadgroup size: 256

**REDUCE_STDDEV**
- Same as REDUCE_VAR, single sqrt at end (~4 cycles on M-series)
- Edge case: clamp negative variance (from rounding) to 0 before sqrt

**REDUCE_HISTOGRAM**
- Privatized histograms with atomic shared memory
- Phase 1: each threadgroup maintains private 256-bin histogram in shared memory
- Phase 2: final merge of block histograms via global atomics
- Privatization reduces atomic conflicts by factor of num_blocks (2-3x speedup)
- Bin computation: `bin = (int)((value - min_val) * num_bins / (max_val - min_val))`

**REDUCE_ARGMAX**
- Reduction tree carrying (value, index) pairs
- Tie-breaking: keep pair with smaller index (first occurrence, deterministic)
- SIMD shuffle for warp-level reduction, then shared memory
- Process float4, track 4 indices, reduce within vector first

**REDUCE_ARGMIN**
- Same as argmax with reversed comparator
- Can template both from single kernel with comparison operator parameter

### Correlation

**REDUCE_CORRELATION**
- Single-pass Pearson: accumulate sum_x, sum_y, sum_xx, sum_yy, sum_xy
- Formula: `r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_xx - sum_x²)(n*sum_yy - sum_y²))`
- Use Welford's method for variance terms in final step

**REDUCE_COVARIANCE_MATRIX**
- Batched pairwise covariance, symmetric output
- Dispatch: 2D grid with n_features x n_features threads
- Only compute upper triangle, cache feature columns in threadgroup memory

**REDUCE_WEIGHTED_SUM**
- Fused weight*value multiplication in map phase before reduction
- Threadgroup reduction to minimize atomic contention

### Advanced Statistics

**TOPK_SELECT**
- Partial sort with per-thread min-heap, O(n log k)
- Each thread maintains heap of k elements, merge thread heaps in shared memory
- Switch to full sort when k/n > 0.1

**SORT_RADIX**
- 4-bit radix sort with prefix sum scatter
- 8 passes for 32-bit keys (4 bits per pass)
- Uses existing SCAN_EXCLUSIVE for scatter offsets
- Increase to 8-bit radix (256 bins) for large n

**SORT_BITONIC**
- Bitonic sorting network, O(n log²n), power-of-2 in-place
- Local threadgroup sort for blocks, then global merge passes
- Requirement: power-of-2 input size

### ML Activations

**MAP_RELU**
- `max(0.0f, x)`, branchless, fully memory-bound
- Fuse with preceding conv/matmul if possible

**MAP_LEAKY_RELU**
- `select(alpha*x, x, x > 0.0f)`, branchless
- Alpha typically 0.01 or 0.2
- Common in GAN discriminators

**MAP_ELU**
- `select(alpha*(exp(x)-1), x, x > 0.0f)`
- exp() is ~20-30 cycles on Metal, consider fast_exp for alpha=1.0

**MAP_GELU**
- Formula: `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x³)))`
- Constant: sqrt(2/pi) = 0.7978845608
- Tanh approx within 0.1% of exact erf version
- Models: GPT-2/3, BERT, ViT

**MAP_SILU**
- Formula: `x / (1 + exp(-x))`, fused to avoid storing sigmoid intermediate
- Stability: use `exp(-abs(x))` formulation for |x| > 10
- Models: LLaMA, Mistral, Gemma, EfficientNet

**MAP_MISH**
- Formula: `x * tanh(softplus(x))`
- Stability: branch at x > 20: tanh(softplus(x)) -> tanh(x) -> 1
- Models: YOLOv4

### Softmax

**SOFTMAX_STABLE**
- Three-pass fused: (1) find max, (2) exp(x-max)+sum, (3) normalize
- **CRITICAL WARNING: NEVER decompose into map_exp->reduce_sum->map_div (OVERFLOWS!)**
- Sequences <8192 fit in single threadgroup with shared memory
- Larger sequences: kernel 1 (max+exp+sum), kernel 2 (normalize)
- Threadgroup size: 256

**LOG_SOFTMAX**
- Formula: `log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))`
- More numerically stable than `log(softmax(x))`

**SOFTMAX_ONLINE**
- Single-pass with running max and rescaling (Flash Attention style)
- Maintain running_max and running_sum, rescale on max update
- Critical for Flash Attention's fused attention kernel

### Normalization

**LAYER_NORM**
- Single-pass Welford for mean+variance, then normalize+scale+shift
- Dispatch: 1D grid over batch*seq_len, one threadgroup per row
- Threadgroup size: 256
- Hidden dim <= 8192 fits in single threadgroup
- `rsqrt()` is native on Apple GPU, faster than `1/sqrt()`
- Gamma/beta are broadcast, cache well in L1

**BATCH_NORM**
- Training: batch-wide per-channel reduction. Inference: cached running stats
- Fuse BN+ReLU for Conv-BN-ReLU patterns in inference
- Channels-last (NHWC) gives better cache locality

**RMS_NORM**
- Formula: `y = x / RMS(x) * gamma`, where `RMS(x) = sqrt(mean(x²) + eps)`
- 30% fewer FLOPs than LayerNorm: no mean computation, no shift
- Models: LLaMA, LLaMA-2, Mistral, Gemma

**GROUP_NORM**
- Split channels into groups, normalize each independently
- Dispatch: 3D grid (spatial, num_groups, batch), one threadgroup per group
- Batch-size independent (useful for small batches)

### Attention

**ATTENTION_SDPA**
- Formula: `softmax(Q @ K^T / sqrt(d_k)) @ V`
- Memory complexity: O(N²) — full attention matrix materialized
- Fuse softmax for stability, use threadgroup memory for Q row

**ATTENTION_FLASH**
- Tile Q,K,V into blocks fitting in threadgroup memory (16-64KB)
- Memory complexity: O(N) — never materialize N x N attention matrix
- Key technique: online softmax with incremental max/sum rescaling
- Speedup: 2-4x for sequences > 2048 tokens
- Use fp16 for 2x memory bandwidth

**ROPE_APPLY**
- Formula: `[x*cos - y*sin, x*sin + y*cos]` on consecutive dimension pairs
- Precompute sin/cos once per sequence length, vectorize with float2
- Supports dynamic sequence length via frequency interpolation

**ROPE_PRECOMPUTE**
- Formula: `theta_i = base^(-2i/d)`, `freq[pos,i] = pos * theta_i` (base=10000)
- One-time setup, cache across batches/layers

**ALIBI_BIAS**
- Formula: `attention_scores[i,j] += m * (i - j)`, `m = 2^(-8*h/H)`
- No learned parameters, extrapolates to longer sequences

### Loss Functions

**LOSS_CROSS_ENTROPY**
- Fused log-softmax + NLL with log-sum-exp trick
- max(logits) subtracted before exp to prevent overflow
- Supports class weights, ignore_index

**LOSS_MSE**
- Fused (pred-target)² + reduction, no intermediate buffer
- Supports both mean and sum reduction

**LOSS_HUBER**
- Smooth L1: `0.5*x² if |x|<delta, else delta*(|x|-0.5*delta)`
- Branchless with `select(linear, quadratic, abs_diff < delta)`

**LOSS_KL_DIV**
- Formula: `target * (log(target) - log(pred))`
- Clamp target to epsilon (1e-10) for safe log, zero targets contribute 0

**LOSS_BCE**
- Formula: `-(y*log(p) + (1-y)*log(1-p))`
- Clamp predictions to [epsilon, 1-epsilon] for log stability

### Optimizers

**OPT_SGD**
- Formula: `p -= lr * g`
- Buffers: params (R/W), grads (R), lr (constant)

**OPT_SGD_MOMENTUM**
- Formula: `v = mu*v + g; p -= lr*v`
- Velocity buffer must persist across iterations (init to zeros)

**OPT_ADAM**
- Formula: `m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g²; m_hat = m/(1-b1^t); v_hat = v/(1-b2^t); p -= lr * m_hat / (sqrt(v_hat) + eps)`
- Buffers: params (R/W), grads (R), m (R/W), v (R/W), hyperparams (constant)
- m, v initialized to zeros; step counter for bias correction

**OPT_ADAMW**
- Formula: `p -= lr*wd*p` (decoupled weight decay), then Adam update
- Weight decay applied to parameters directly, not to gradients

**GRAD_CLIP_NORM**
- Two-pass: (1) REDUCE_L2 for global norm, (2) `scale = min(1, max_norm/global_norm)`
- Can fuse with optimizer step if norm pre-computed

**GRAD_CLIP_VALUE**
- Element-wise `clamp(grad, -max_val, max_val)`, single-pass

### FFT

**FFT_RADIX2**
- Cooley-Tukey radix-2 DIT
- Steps: 1. Bit-reversal permutation, 2. log2(N) stages of butterflies
- Butterfly: `X[i] = X[i] + W*X[i+m]; X[i+m] = X[i] - W*X[i+m]`
- Threadgroup memory for butterfly exchanges, precompute twiddle factors
- Complexity: O(N log N), 5N*log2(N) flops

**FFT_RADIX4**
- Radix-4 Cooley-Tukey, 4-point butterflies
- 25% fewer memory passes than radix-2 (log4(N) stages)
- Optimal for N = 4^k (256, 1024, 4096, ...)

**IFFT**
- Same as FFT with conjugate twiddle factors + 1/N scaling
- Can reuse FFT kernel with conjugate flag

**FFT_REAL**
- Exploit Hermitian symmetry: N-point real FFT via N/2-point complex FFT
- Steps: pack real->complex (N/2), FFT, Hermitian unpack + correction
- ~2x over naive complex FFT of real data

**SPECTRAL_POWER**
- Formula: `|X[k]|² = Re² + Im²`
- Can fuse with FFT final stage to avoid extra pass
- Apply window correction factor if windowed

### Signal Processing

**CONV1D**
- Sliding window in threadgroup shared memory with halo regions
- Supports zero, reflect, and circular padding via address computation
- SIMD vector loads (float4) when kernel_size % 4 == 0

**FFT_CONV1D**
- Algorithm: FFT(signal) * FFT(kernel) then IFFT
- Zero-pad to next power-of-2, faster than direct for kernel_size > 64

**AUTOCORRELATION**
- Short signals: direct `correlate[k] = sum(signal[i] * signal[i+k])`
- Long signals: FFT via Wiener-Khinchin: `IFFT(|FFT(signal)|²)`
- Autocorrelation is even function, exploit to halve work

**WINDOW_APPLY**
- Hann: `0.5 - 0.5*cos(2*pi*n/(N-1))`
- Hamming: `0.54 - 0.46*cos(2*pi*n/(N-1))`
- Blackman: `0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))`
- Compute coefficients on-the-fly with fast_math cos()

**FIR_FILTER**
- Formula: `output[n] = sum(input[n-k] * coeff[k])`, reversed coefficients
- Circular buffer in threadgroup memory for real-time processing

### Linear Algebra

**SPMV_CSR**
- One thread per row, coalesced access to values/col_indices
- Segmented reduction for rows with >128 non-zeros
- Dispatch: `dispatch(num_rows, 1, 1)`

**MATMUL_BATCHED**
- One threadgroup per batch element, shared memory tiling (16x16 or 32x32)
- Beats MPS for small matrices (<512) or heterogeneous batch dimensions

**MATVEC**
- Each thread computes one output element: `y[i] = dot(A[i,:], x)`
- Broadcast x into threadgroup memory for reuse across rows

**TRIANGULAR_SOLVE**
- Level-set parallelism: identify independent rows, solve in waves
- Wave 0: diagonal only; Wave k: rows depending on waves 0..k-1
- Pre-compute level-set structure on CPU

**OUTER_PRODUCT**
- Formula: `C[i,j] = u[i] * v[j]`, trivially parallel
- Dispatch: `dispatch((n+15)/16, (m+15)/16, 1)` with 16x16 threads

### Simulation

**SIM_HEAT_EQUATION**
- Explicit Euler with 5-point stencil: `u_next = u + r*(laplacian)`
- CFL stability: `dt < dx² / (4*alpha)`
- Double buffering (ping-pong), tile into threadgroup memory

**SIM_WAVE_EQUATION**
- Leapfrog: `u_next = 2*u - u_prev + (c*dt/dx)² * laplacian`
- CFL stability: `c*dt/dx < 1` (2D: `< 1/sqrt(2)`)
- Triple buffering: rotate u_prev <- u_curr <- u_next

**SIM_STENCIL_2D**
- General configurable stencil with coefficient array and neighbor offsets
- Threadgroup memory tiling: load `(blockDim+2*radius)²` into shared memory

**SIM_STENCIL_3D**
- General 3D stencil (7-point, 19-point, 27-point)
- Cache blocking essential, Z-curve (Morton order) improves locality

**INTEGRATE_RK4**
- Algorithm: `k1=f(t,y); k2=f(t+dt/2,y+dt/2*k1); k3=f(t+dt/2,y+dt/2*k2); k4=f(t+dt,y+dt*k3); y_next = y + dt/6*(k1+2k2+2k3+k4)`
- Store k1-k4 in registers, avoid intermediate global writes
- Each thread = one equation or one trajectory

**INTEGRATE_VERLET**
- Algorithm: 1. `x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²`; 2. Compute F(x(t+dt)); 3. `v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt`
- Symplectic: conserves phase-space volume, bounded energy drift
- Use cases: molecular dynamics, orbital mechanics, cloth simulation

**MONTE_CARLO_INTEGRATE**
- Formula: `Integral ~ Volume * (1/N) * sum(f(x_i))`, x_i ~ Uniform
- Error O(1/sqrt(N)), independent of dimension
- Use PRNG_PHILOX for GPU-friendly parallel random generation

### PRNG

**PRNG_PHILOX**
- Philox4x32-10 counter-based PRNG
- 10-round bijective function on 4x u32 state, separate 2x u32 key per round
- Each thread gets independent stream via unique counter
- SIMD u32x4 operations, unroll 10 rounds explicitly

**PRNG_UNIFORM_F32**
- Philox -> [0,1) float via `f32(u32 >> 9) * 0x1.0p-24f`
- Fuse generation + conversion, 4 outputs per Philox call

**PRNG_NORMAL_F32**
- Box-Muller: `z0 = sqrt(-2*ln(u1))*cos(2*pi*u2)`, `z1 = sqrt(-2*ln(u1))*sin(2*pi*u2)`
- 2 independent N(0,1) normals per pair of uniforms
- Use `metal::fast::sincos`, reject u1=0 edge case

**PRNG_DROPOUT_MASK**
- Philox uniform + threshold comparison -> binary mask
- `mask = select(0, 1, uniform < keep_prob)`, output u8
- Fuse Philox->uniform->compare in single kernel, 4 masks per Philox call

---

## Pipeline Recipes

Optimized multi-kernel pipelines for common workloads. Steps marked _(note)_ are CPU-side or use MPS rather than custom kernels.

### 1. Vector Search

1. **Quantize database vectors** — QUANTIZE_F16_TO_I8: reduce memory bandwidth for large searches
2. **Coarse filtering** — MAP_COMPARE: eliminate distant vectors with cheap L2 bound
3. **Compact candidates** — COMPACT_MASK: remove filtered vectors from computation
4. **Dequantize candidates** — DEQUANTIZE_I8_TO_F16: restore precision for final scoring
5. **Compute similarities** — DOT_BATCHED: batched dot products for all candidates
6. **Find top-k** — TOPK_SELECT: partial sort for top results without full sort

### 2. ETL

1. **Parse and validate** — MAP_COMPARE: check data constraints
2. **Filter invalid rows** — COMPACT_MASK: remove invalid data
3. **Transform values** — MAP_LOG (logarithmic) or MAP_CLAMP (other): apply transformations
4. **Compute aggregates** — REDUCE_MEAN: calculate statistics
5. **Normalize** — MAP_DIV: scale by statistics

### 3. ML Inference (Transformer)

1. **Embedding lookup** — _(CPU: irregular memory access)_
2. **RMS normalization** — RMS_NORM: pre-attention normalization (LLaMA-style)
3. **Attention** — ATTENTION_FLASH: flash attention for memory efficiency
4. **Residual add** — MAP_ADD: skip connection
5. **RMS normalization** — RMS_NORM: pre-MLP normalization
6. **MLP up-projection** — _(MPS matmul)_
7. **Activation** — MAP_SILU: SiLU activation (LLaMA/Mistral FFN)
8. **MLP down-projection** — _(MPS matmul)_
9. **Residual add** — MAP_ADD: skip connection

### 4. Random Initialization

**Uniform distribution:**
1. **Generate uniform random** — PRNG_UNIFORM_F32: Philox-based [0,1)
2. **Scale and shift** — MAP_FMA: map to [low, high)

**Normal distribution:**
1. **Generate normal random** — PRNG_NORMAL_F32: Box-Muller N(0,1)
2. **Scale and shift** — MAP_FMA: map to N(mean, std²)

**Bernoulli distribution:**
1. **Generate dropout mask** — PRNG_DROPOUT_MASK: Bernoulli(p) mask

### 5. Transformer Block

Parameters: hidden_dim, num_heads, seq_len. Uses Flash Attention when seq_len >= 2048, SDPA otherwise.

1. **Pre-attention norm** — RMS_NORM: RMS normalization
2. **Self-attention** — ATTENTION_FLASH or ATTENTION_SDPA: based on seq_len threshold
3. **Residual connection 1** — MAP_ADD: skip connection around attention
4. **Pre-MLP norm** — RMS_NORM: RMS normalization before feedforward
5. **MLP up-projection + gate** — _(MPS matmul: hidden_dim -> 4*hidden_dim)_
6. **MLP activation** — MAP_SILU: SiLU in gated FFN
7. **MLP down-projection** — _(MPS matmul: 4*hidden_dim -> hidden_dim)_
8. **Residual connection 2** — MAP_ADD: skip connection around MLP

### 6. Training Step

Parameters: optimizer type, num_params, optional gradient_clip config.

**With gradient clipping (norm):**
1. **Compute gradient global norm** — REDUCE_L2: L2 norm across all parameters
2. **Scale gradients** — GRAD_CLIP_NORM: clip to max_norm

**With gradient clipping (value):**
1. **Clamp gradients** — GRAD_CLIP_VALUE: clamp to [-max_val, max_val]

**Optimizer update (always):**
- OPT_SGD / OPT_SGD_MOMENTUM / OPT_ADAM / OPT_ADAMW: parameter update

### 7. Spectral Analysis

Parameters: signal_length, window_type.

1. **Apply window** — WINDOW_APPLY: reduce spectral leakage _(skipped for rectangular window)_
2. **Forward FFT** — FFT_RADIX4 (if n is power-of-4 and n >= 256), FFT_RADIX2 (if power-of-2), or FFT_RADIX2 with zero-padding (otherwise)
3. **Power spectral density** — SPECTRAL_POWER: |X[k]|² for frequency power

### 8. Signal Processing

1. **Apply window function** — WINDOW_APPLY: reduce spectral leakage
2. **Forward FFT** — FFT_RADIX2: transform to frequency domain
3. **Frequency-domain filter** — MAP_MUL: apply bandpass/lowpass/highpass filter
4. **Inverse FFT** — IFFT: reconstruct filtered time-domain signal
5. **Power spectrum analysis** — SPECTRAL_POWER: compute PSD and spectral metrics

### 9. N-Body Simulation

Parameters: num_particles, dt.

1. **Compute gravitational forces** — _(Custom N-body force kernel: all-pairs O(N²) or tree O(N log N))_
2. **Velocity Verlet position update** — INTEGRATE_VERLET: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
3. **Recompute forces at new positions** — _(Second force evaluation for Verlet velocity update)_
4. **Velocity Verlet velocity update** — INTEGRATE_VERLET: v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt

### 10. PDE Solver

Parameters: equation_type, grid_size, num_steps.

**Heat equation:**
1. **Heat equation step** (x num_steps) — SIM_HEAT_EQUATION: 5-point stencil, CFL limit dt < dx²/(4*alpha)
2. **Buffer swap** — _(host-managed ping-pong between read/write buffers)_

**Wave equation:**
1. **Wave equation step** (x num_steps) — SIM_WAVE_EQUATION: leapfrog, triple buffering
2. **Buffer swap** — _(host-managed ping-pong)_

**General stencil:**
1. **Stencil step** (x num_steps) — SIM_STENCIL_2D or SIM_STENCIL_3D: based on dimensions
2. **Buffer swap** — _(host-managed ping-pong)_

### 11. Data Analysis

1. **Validate and filter** — MAP_COMPARE + COMPACT_MASK: clean data, remove outliers
2. **Descriptive statistics** — REDUCE_MEAN + REDUCE_VAR + REDUCE_MIN + REDUCE_MAX: parallel column-wise stats
3. **Distribution histogram** — REDUCE_HISTOGRAM: build distribution histograms with atomic bins
4. **Percentiles** — PERCENTILE_SELECT: compute p25, p50, p75, p90, p99 via selection
5. **Correlation matrix** — REDUCE_PEARSON_PAIRWISE: pairwise Pearson correlation across features

### 12. Recommendation

Parameters: num_users, num_items, embedding_dim.

1. **Embedding lookup** — _(Gather user/item embeddings)_
2. **Compute similarity scores** — DOT_BATCHED: batched dot products (num_users x num_items scores)
3. **Select top-k recommendations** — TOPK_SELECT: top-k per user without full sort
4. **Normalize scores** — MAP_DIV: scale scores to [0,1] for serving

### 13. Diffusion Model

Parameters: num_steps.

1. **Precompute noise schedule** — _(Compute alpha/beta for diffusion steps)_
2. **Denoise** (x num_steps) — GROUP_NORM + ATTENTION_FLASH + MAP_SILU + MAP_ADD: per step GroupNorm -> Self-Attention -> SiLU MLP -> Residual

### 14. Scientific Computing

**Linear system (conjugate gradient):**
1. **SpMV (Ax computation)** — SPMV_CSR: sparse matrix-vector multiply
2. **Conjugate gradient step** — MAP_FMA + DOT_BATCHED + MAP_ADD: residual update, alpha/beta computation, search direction
3. **Convergence check** — REDUCE_L2: check ||r|| < tol for early exit

**Eigenvalue (Lanczos):**
1. **Lanczos iteration** — SPMV_CSR + DOT_BATCHED: build tridiagonal matrix from sparse matrix
2. **QR algorithm** — _(Implicit QR on tridiagonal for eigenvalue extraction)_
3. **Eigenvector reconstruction** — MATVEC: inverse iteration from Ritz values

**Optimization (gradient descent with line search):**
1. **Compute gradient** — _(Problem-specific gradient computation)_
2. **Line search** — DOT_BATCHED + MAP_FMA: backtracking Armijo line search for step size
3. **Parameter update** — MAP_FMA: x_new = x - alpha * grad
4. **Convergence check** — REDUCE_L2: check ||grad|| < tol

---

## Performance Estimation Reference

### Precision Multipliers

| Precision | Multiplier | Bytes per Element |
|-----------|-----------|-------------------|
| fp64 | 0.5x | 8 |
| fp32 | 1.0x | 4 |
| fp16 | 1.8x | 2 |
| int8 | 2.5x | 1 |

### Bandwidth Formula

```
effective_gflops = kernel_gflops * precision_multiplier
num_elements = (data_size_mb * 1024 * 1024) / bytes_per_element
total_flops = num_elements * 2.0  (flops per element)
runtime_ms = (total_flops / (effective_gflops * 1e9)) * 1000
bandwidth_gbps = (data_size_mb / 1024) / (runtime_ms / 1000)
bandwidth_efficiency = min(1.0, bandwidth_gbps / 400) * 100%
```

M-series unified memory bandwidth baseline: ~400 GB/s.

For memory-bound kernels: consider quantization to reduce bandwidth pressure.
For compute-bound kernels: already at compute optimum.
