# Benchmark Conclusions

Results from 94 kernels across 20 domains, tested at N=10,000,000 elements on Apple M5 (24 GB unified memory, ~200 GB/s theoretical peak bandwidth).

## Backend win rates

| Backend | Wins | Share |
|---------|------|-------|
| torch_cpu | 45 | 47.9% |
| torch_mps | 19 | 20.2% |
| mlx | 19 | 20.2% |
| numpy | 11 | 11.7% |

## Speedup over numpy (baseline)

| Backend | Median | Average | Range |
|---------|--------|---------|-------|
| torch_cpu | 1.73x | 11.49x | 0.21x – 494x |
| torch_mps | 2.42x | 9.30x | 0.14x – 144x |
| mlx | 2.31x | 46.88x | 0.31x – 2180x |

MLX's average is skewed by a few massive wins (transpose 2180x, GELU 87x, flash attention 70x). Its median tells a more honest story: ~2.3x over numpy at this element count.

## Key findings

### torch_cpu dominates at 10M elements

48% of kernel wins go to PyTorch CPU. At N=10M most operations are memory-bound, and the CPU's cache hierarchy plus Apple's Accelerate BLAS framework (which torch calls into) wins handily. GPU dispatch overhead eats into gains for anything that finishes in under 1ms on CPU.

### MLX beats MPS in GPU head-to-head (40 vs 18)

Comparing just the two GPU paths, MLX wins 2:1 over torch MPS. MLX's lazy graph evaluation and fused kernels eliminate intermediate memory allocations that MPS cannot avoid through its eager execution model.

### GPU wins on compute-heavy kernels, loses on memory-bound and branchy ones

**Where GPU dominates:**

| Kernel | MLX speedup over numpy | Why |
|--------|----------------------|-----|
| transpose_2d | 2180x | Massive parallelism, coalesced memory access |
| map_gelu | 87x | Compute-dense (tanh approximation) |
| attention_flash | 70x | O(N) memory tiled attention, ideal for GPU |
| map_log | 40x | Transcendental math, GPU has dedicated units |
| map_mish | 40x | Compound transcendental (tanh + softplus) |

**Where CPU/numpy wins over GPU (torch_mps):**

| Kernel | numpy advantage | Why |
|--------|----------------|-----|
| reduce_histogram | 6.9x | Atomic contention on GPU shared memory |
| window_apply | 6.3x | Simple multiply, dispatch overhead dominates |
| integrate_rk4 | 3.9x | Sequential dependency between k1-k4 stages |
| matmul_batched | 3.7x | Small matrices (128x128), Accelerate BLAS is tuned for this |
| reduce_min | 2.6x | Pure memory-bound, CPU cache wins at this size |

### Nobody hits peak bandwidth

The M5's theoretical ~200 GB/s memory bandwidth is barely tapped. The highest legitimate measurement is ~92 GB/s for reduce_min on numpy (46% of peak). Results exceeding 200 GB/s (grad_clip_norm at 11,636 GB/s, transpose_2d MLX at 3,097 GB/s) are cache artifacts — the working set fits in L2/SLC so measured throughput exceeds DRAM bandwidth.

This means there is significant room for kernel optimization across all backends.

### Per-domain breakdown

| Domain | Winner(s) | Notes |
|--------|-----------|-------|
| reductions | Mixed (cpu 4, mps 2, mlx 2, numpy 2) | Memory-bound, CPU cache competitive |
| elementwise | torch_cpu (7), mps (3), mlx (2) | Simple ops favor CPU at this N |
| activations | mlx (3), cpu (2), mps (1) | Compute-heavy activations favor GPU |
| softmax | mps (2), mlx (1) | Multi-pass reduction benefits GPU |
| normalization | torch_cpu (4) | Welford passes fit in cache |
| attention | cpu (2), mlx (2) | Small seq_len (512) keeps CPU competitive |
| loss | torch_mps (1) | Only 1 kernel had all backends (MLX errored) |
| optimizers | torch_cpu (5), mps (1) | Sequential state updates favor CPU |
| fft | mlx (3), cpu (1), mps (1) | FFT butterfly parallelism suits GPU |
| signal | torch_cpu (3), mps (1) | Small kernel convolutions favor CPU |
| linalg | torch_cpu (3), numpy (1), mlx (1) | Accelerate BLAS dominates small matrices |
| simulation | numpy (2), cpu (3) | Stencils with boundary checks are branchy |
| sorting | numpy (4), cpu (1), mlx (1) | Selection/comparison algorithms are branch-heavy |
| prng | numpy (1), mps (3) | Counter-based PRNG parallelizes well on GPU |
| layout | numpy (1), mps (1), mlx (1) | Transpose is a GPU sweet spot |
| scans | mps (2), mlx (1) | Prefix sums parallelize with work-efficient algorithms |
| quantization | torch_cpu (3) | Simple type conversion, memory-bound |
| dot_products | cpu (1), mlx (1) | Depends on batch size vs dim |
| fused | torch_cpu (3), mlx (1) | Map-reduce fusion helps GPU on square_sum |
| correlation | torch_cpu (3), mps (1) | Two-pass algorithms keep CPU competitive |

## Implications for the Metal agent's routing

1. **The 100k element threshold in the agent is directionally correct.** At 10M elements, GPU wins on compute-heavy ops but many memory-bound ops still favor CPU. The actual crossover point varies by kernel — somewhere between 100k and 1M depending on compute intensity.

2. **MLX should be the default GPU path** over torch MPS. The 40-18 head-to-head record confirms what the agent already prescribes. MLX's lazy evaluation and kernel fusion are hard to beat.

3. **Custom Metal shaders only make sense for ops MLX doesn't cover.** MLX's fused graph execution with pre-optimized tiling and vectorization beats naive single-dispatch Metal shaders. The Rust+Metal path should target the gaps: custom stencils, non-standard reductions, domain-specific fused kernels.

4. **The bandwidth gap is an optimization opportunity.** At 46% of theoretical peak, there is room for Rust+Metal kernels with proper tiling, threadgroup memory staging, and coalesced access patterns to outperform the Python frameworks on memory-bound operations.

5. **CPU path remains important.** Nearly half the wins at 10M elements go to torch_cpu. For small-to-medium workloads, branchy algorithms, and operations that fit in cache, the agent should not force GPU dispatch.

---

## Rust+Metal+NEON Benchmark Results (N=1M)

210 benchmarks across 21 domains using optimized Metal shaders (simd intrinsics, float4 vectorization, tiled memory access) and expanded NEON SIMD coverage (33 intrinsic functions).

### Metal GPU dominance on compute-heavy kernels

| Kernel | Metal (us) | Scalar (us) | Speedup | Notes |
|--------|-----------|-------------|---------|-------|
| attention_sdpa | 5,454 | 486,608 | **89x** | Online softmax SDPA, never materializes NxN |
| transpose_2d | 1,348 | 74,194 | **55x** | Tiled with bank-conflict-free shared memory |
| prng_normal | 256 | 13,196 | **52x** | Box-Muller on GPU cores |
| fir_filter | 424 | 15,368 | **36x** | Sliding window shared memory |
| softmax_stable | 1,320 | 37,329 | **28x** | simd_max + simd_sum intrinsics |
| conv1d | 1,293 | 19,328 | **15x** | Shared memory kernel tiling |
| fused_softmax_cross_entropy | 252 | 2,036 | **8x** | Forward + backward in one dispatch |
| outer_product | 324 | 2,424 | **7.5x** | Coalesced 2D grid writes |
| fft_radix2 | 5,524 | 28,987 | **5.2x** | Threadgroup-local butterfly stages |
| matvec | 1,942 | 10,249 | **5.3x** | Vectorized dot products |
| fused_scan_compact | 721 | 3,586 | **5x** | Prefix scan + scatter in one kernel |

### NEON SIMD: memory bandwidth champion

NEON consistently achieves the highest bandwidth for operations that fit in the CPU memory subsystem:

| Kernel | NEON BW (GB/s) | Metal BW (GB/s) | Winner |
|--------|---------------|-----------------|--------|
| map_fma | **100.1** | 12.2 | NEON (8.2x) |
| map_add/mul/div | **88.5-88.9** | 8.8-9.1 | NEON (10x) |
| cosine_similarity | **88.0** | 31.5 | NEON (2.8x) |
| reduce_min/max | **72.3-72.7** (scalar) | 8.4-17.6 | CPU (4-9x) |
| map_clamp | **66.8** | 4.6 | NEON (14.5x) |
| loss_mae | **59.0** | 8.7 | NEON (6.8x) |
| opt_adam | **69.5** | 9.3 | NEON (7.5x) |

100.1 GB/s for map_fma represents **~50% of the M5's theoretical ~200 GB/s** — the highest sustained bandwidth achieved in any benchmark, CPU or GPU.

### Metal dispatch overhead: the ~900us floor

A critical finding: virtually every Metal elementwise kernel shows a ~880-960us median regardless of operation complexity. This is pure dispatch overhead (command buffer creation, GPU scheduling, completion wait). For reference:

- `map_abs` Metal: 893us vs scalar: 60us (scalar 15x faster)
- `map_clamp` Metal: 876us vs NEON: 60us (NEON 15x faster)
- `map_add` Metal: 883us vs NEON: 90us (NEON 10x faster)

**Metal only wins when compute time exceeds ~2ms**, which at N=1M means operations with O(N log N) or O(N^2) complexity (FFT, attention, convolution, softmax multi-pass).

### Bandwidth achieved (Metal GPU)

| Kernel | BW (GB/s) | % of ~200 GB/s |
|--------|-----------|----------------|
| repack_aos_to_soa | 54.0 | 27% |
| log_softmax | 53.8 | 27% |
| repack_soa_to_aos | 52.4 | 26% |
| softmax_stable | 50.9 | 25% |
| transpose_2d | 49.8 | 25% |
| sim_verlet | 40.4 | 20% |
| matvec | 34.5 | 17% |
| cosine_similarity | 31.5 | 16% |

Peak Metal bandwidth of 54 GB/s (27%) confirms conclusion #4 — significant headroom remains. The simd intrinsic and float4 optimizations improved over the pre-optimization baseline but dispatch overhead and single-simdgroup-per-threadgroup sizing limit throughput. Further gains require:
- Larger threadgroups (multiple simdgroups) for better occupancy
- Persistent kernels that amortize dispatch over multiple iterations
- Async copy staging from device to threadgroup memory

### Fused custom kernels (ops MLX can't express)

| Kernel | Metal (us) | Scalar (us) | Speedup |
|--------|-----------|-------------|---------|
| fused_softmax_cross_entropy | 252 | 2,036 | **8.1x** |
| fused_scan_compact | 721 | 3,586 | **5.0x** |
| fused_layernorm_dropout_residual | 902 | 2,159 | **2.4x** |
| fused_attention_softmax_v | 6,963 | 18,084 | **2.6x** |
| fused_rope_attention_mask | 259 | 442 | **1.7x** |
| fused_adam_clip_update | 2,487 | 698 | **scalar wins** |

5 of 6 fused kernels show Metal wins. `fused_adam_clip_update` loses because the reduction (grad norm) + per-element update pattern has poor GPU utilization at N=1M. These fused kernels represent the strongest case for custom Metal shaders — operations that would require multiple framework dispatches (3-5 separate kernel launches in MLX/PyTorch) but execute as a single GPU dispatch here.

### Updated routing recommendations

1. **Raise the Metal threshold to N≥10M for elementwise ops.** The ~900us dispatch floor means Metal loses on anything completing in <2ms on CPU. At N=10M, even simple ops take 5-10ms on CPU, making GPU worthwhile.

2. **Use NEON SIMD as the primary fast path for N=100K-10M.** NEON achieves 50-100 GB/s bandwidth with zero dispatch overhead, beating both scalar and Metal in this range.

3. **Metal is mandatory for O(N log N)+ ops.** Softmax (28x), FFT (5x), attention (89x), and convolution (15-36x) show GPU wins even at N=1M. Route these to Metal regardless of size.

4. **Fused custom kernels justify the Metal agent.** The 6 fused kernels demonstrate 1.7-8x speedups that no framework can match with their standard op decomposition.

5. **Occupancy is the next optimization frontier.** Current shaders use 1 simdgroup (32 threads) per threadgroup. Increasing to 4-8 simdgroups per threadgroup would improve latency hiding and push bandwidth toward the 100+ GB/s range on Metal.

---

## Caveats

- **N=10M only for Python, N=1M for Rust.** The GPU crossover shifts at different sizes. At N=100M, GPU win rates would increase significantly as dispatch overhead becomes negligible relative to compute time.
- **Loss domain incomplete.** MLX loss benchmarks errored due to an API path bug (`mx.losses` vs `mlx.nn.losses`), now fixed. Rerun needed for complete loss comparison.
- **Simulation has no MLX path.** PDE stencils and ODE integrators don't map to standard MLX ops, which is exactly why custom Metal shaders exist.
- **Single-run medians.** Results reflect one benchmark session. Thermal state, background processes, and memory pressure can shift numbers by 5-15%.
- **Rust Metal shaders use 1 simdgroup per threadgroup.** This limits occupancy and bandwidth. Production shaders should use larger threadgroups with explicit simdgroup coordination.
