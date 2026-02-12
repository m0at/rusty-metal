# rusty-metal

A single command to give [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview) a Metal GPU compute specialist agent for Apple Silicon.

Installs a `.claude/agents/` directory with:

- **`metal.md`** — Agent prompt: architecture, dispatch conventions, performance rules, compute routing (local Metal vs cloud), build instructions
- **`metal-kernel-hints.md`** — 109 production Metal GPU kernels: catalog, operation routing tables, fusion patterns, implementation hints, 14 pipeline recipes, performance estimation

## Install

```bash
# From GitHub
cargo install --git https://github.com/m0at/rusty-metal.git

# From a local clone
git clone https://github.com/m0at/rusty-metal.git
cargo install --path rusty-metal
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

The agent writes Metal shaders, Rust GPU dispatch code, picks optimal kernels for workloads, fuses operations, builds multi-step pipelines, and routes large jobs to cloud compute when local hardware isn't enough.

## What's inside

### Kernel catalog (109 kernels across 22 domains)

Reductions, correlation, elementwise, ML activations, softmax, normalization, attention (SDPA/Flash/MQA/GQA), loss functions, optimizers (SGD/Adam/AdamW), FFT, signal processing, linear algebra, simulation & physics, sorting, PRNG, layout transforms, fused ops.

### Operation routing

Decision tables that map operations + conditions to optimal kernel selections:

- `sort` — bitonic if n <= 32768 and power-of-2, else radix
- `attention` — SDPA if seq < 2048, Flash otherwise
- `spmv` — CSR if density < 5%, dense matvec otherwise

### Fusion patterns

Identifies when consecutive operations should merge into single kernels: softmax (4-op), layernorm (4-op), attention (5-op), activation+dropout, loss+backward.

### Pipeline recipes

14 end-to-end GPU pipelines: vector search, ETL, transformer inference, training step, spectral analysis, signal processing, N-body simulation, PDE solver, data analysis, recommendation, diffusion model, scientific computing.

### Compute routing

A state machine that evaluates workload size, parallelism, memory requirements, and hardware ceilings to recommend the right execution target — local Metal, local CPU with rayon, or cloud (Lambda H100, DigitalOcean fleet, AWS multi-node) — with cost and time estimates.

### System health monitor

A pre-flight gate and runtime watchdog that probes memory pressure, thermal throttling, CPU load, and power source before and during workloads — all via lightweight no-sudo macOS commands (< 50ms). Blocks launches when memory is critical, warns about thermal throttling with cooling/placement guidance, and monitors long-running workloads in a background thread.

## How it works

The agent prompt and kernel reference live in `content/` as markdown files. They're embedded into the binary at compile time via `include_str!`, so the CLI is a single zero-dependency binary with no runtime file access needed. Running `rusty-metal init` writes them into your project's `.claude/agents/` directory where Claude Code picks them up.

## Contributing

Edit files in `content/` and rebuild — changes are picked up automatically.

| File | What it contains |
|------|-----------------|
| `content/metal-agent.md` | Agent system prompt, dispatch conventions, performance rules, compute routing state machine |
| `content/metal-kernel-hints.md` | Kernel catalog, operation routing tables, fusion patterns, implementation hints, pipeline recipes |

Contributions welcome:

- New kernel domains (graph algorithms, image processing, cryptography)
- Additional pipeline recipes for common workloads
- Performance data from real benchmarks on specific Apple Silicon chips (M1–M5)
- Compute routing refinements (new cloud providers, updated pricing)
- Metal shader best practices and Apple GPU architecture notes

## License

[MIT](LICENSE)
