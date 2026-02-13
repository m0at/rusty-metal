#!/usr/bin/env bash
# Top-level benchmark runner for rusty-metal.
# Runs both Python (numpy/torch/MLX) and Rust (scalar/NEON/Metal) suites,
# then compares results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

N="${N:-10000000}"
DOMAIN="${DOMAIN:-}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          rusty-metal benchmark suite                    ║"
echo "║  N = $N elements                                        "
echo "╚══════════════════════════════════════════════════════════╝"

# ── Python benchmarks ──
echo ""
echo "━━━ Python benchmarks (numpy / PyTorch MPS / MLX) ━━━"
echo ""

PYTHON_DIR="$SCRIPT_DIR/python"
if [ ! -d "$PYTHON_DIR/.venv" ]; then
    echo "Setting up Python venv with uv..."
    cd "$PYTHON_DIR"
    uv venv
    uv pip install -e .
    cd "$SCRIPT_DIR"
fi

PYTHON_ARGS="--n $N --json $RESULTS_DIR/python_results.json"
if [ -n "$DOMAIN" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --domain $DOMAIN"
fi

"$PYTHON_DIR/.venv/bin/python" -m bench.run_all $PYTHON_ARGS

# ── Rust benchmarks ──
echo ""
echo "━━━ Rust benchmarks (scalar / NEON SIMD / Metal GPU) ━━━"
echo ""

RUST_DIR="$SCRIPT_DIR/rust"
echo "Building Rust benchmarks (release)..."
cargo build --release --manifest-path "$RUST_DIR/Cargo.toml" 2>&1 | tail -1

RUST_ARGS="--n $N --json $RESULTS_DIR/rust_results.json"
if [ -n "$DOMAIN" ]; then
    RUST_ARGS="$RUST_ARGS --domain $DOMAIN"
fi

"$RUST_DIR/target/release/rusty-metal-bench" $RUST_ARGS

# ── Comparison ──
echo ""
echo "━━━ Comparison ━━━"
echo ""

"$PYTHON_DIR/.venv/bin/python" "$SCRIPT_DIR/compare.py" \
    --python "$RESULTS_DIR/python_results.json" \
    --rust "$RESULTS_DIR/rust_results.json" \
    --output "$RESULTS_DIR/comparison.txt"

echo ""
echo "Done. Results in $RESULTS_DIR/"
echo "  python_results.json  — Python benchmark data"
echo "  rust_results.json    — Rust benchmark data"
echo "  comparison.txt       — Side-by-side comparison table"
