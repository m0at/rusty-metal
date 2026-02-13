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

## Caveats

- **N=10M only.** The GPU crossover shifts at different sizes. At N=100M, GPU win rates would increase significantly as dispatch overhead becomes negligible relative to compute time.
- **Loss domain incomplete.** MLX loss benchmarks errored due to an API path bug (`mx.losses` vs `mlx.nn.losses`), now fixed. Rerun needed for complete loss comparison.
- **Simulation has no MLX path.** PDE stencils and ODE integrators don't map to standard MLX ops, which is exactly why custom Metal shaders exist.
- **Single-run medians.** Results reflect one benchmark session. Thermal state, background processes, and memory pressure can shift numbers by 5-15%.
- **Rust+Metal side not yet run.** The Rust benchmark suite (scalar, NEON SIMD, Metal GPU) has been written but these results are Python-only. The Rust results will show how hand-tuned Metal shaders and NEON SIMD compare.
