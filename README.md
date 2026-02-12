# rusty-metal

A single command to give [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview) a Metal GPU compute specialist agent for Apple Silicon.

Installs a `.claude/agents/` directory with:

- **`metal.md`** — Agent prompt: architecture, dispatch conventions, performance rules, compute routing (local Metal vs cloud), build instructions
- **`metal-kernel-hints.md`** — 109 production Metal GPU kernels: catalog, operation routing tables, fusion patterns, implementation hints, 14 pipeline recipes, performance estimation

## Install

```bash
cargo install --path .
```

## Usage

```bash
cd your-project
rusty-metal init          # creates .claude/agents/ with Metal agent + kernel hints
rusty-metal init --force  # overwrite existing files
rusty-metal check         # show what exists
```

Then in Claude Code:

```bash
claude --agent metal
```

The agent knows how to write Metal shaders, Rust GPU dispatch code, pick optimal kernels for workloads, fuse operations, build multi-step pipelines, and route large jobs to cloud compute when local hardware isn't enough.

## What's inside

### Kernel catalog (109 kernels across 22 domains)

Reductions, correlation, elementwise, ML activations, softmax, normalization, attention (SDPA/Flash/MQA/GQA), loss functions, optimizers (SGD/Adam/AdamW), FFT, signal processing, linear algebra, simulation & physics, sorting, PRNG, layout transforms, fused ops.

### Operation routing

Decision tables that map operations + conditions to optimal kernel selections. Handles branching logic like:
- `sort`: bitonic if n <= 32768 and power-of-2, else radix
- `attention`: SDPA if seq < 2048, Flash otherwise
- `spmv`: CSR if density < 5%, dense matvec otherwise

### Fusion patterns

Identifies when consecutive operations should merge into single kernels (softmax 4-op, layernorm 4-op, attention 5-op, activation+dropout, loss+backward).

### Pipeline recipes

14 end-to-end GPU pipelines: vector search, ETL, transformer inference, training step, spectral analysis, signal processing, N-body simulation, PDE solver, data analysis, recommendation, diffusion model, scientific computing.

## Contributing

The kernel catalog and agent prompt live in `content/`. Edit those files and rebuild — they're embedded at compile time via `include_str!`.

Areas where contributions are welcome:

- New kernel domains (e.g., graph algorithms, image processing)
- Additional pipeline recipes
- Improved performance data from real benchmarks on specific Apple Silicon chips
- Compute routing refinements (new cloud providers, updated pricing)
- Metal shader best practices and Apple GPU architecture notes

## License

MIT
