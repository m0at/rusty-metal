//! Rust benchmark suite for rusty-metal: Metal GPU, ARM NEON SIMD, and scalar Rust.

mod harness;
mod metal_ctx;
mod neon;
mod shaders;
mod kernels;

use harness::{BenchSuite, DEFAULT_N};
use metal_ctx::MetalCtx;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = args.iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);

    let filter: Option<&str> = args.iter()
        .position(|a| a == "--domain" || a == "-d")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let json_out: Option<PathBuf> = args.iter()
        .position(|a| a == "--json" || a == "-j")
        .and_then(|i| args.get(i + 1))
        .map(|s| PathBuf::from(s));

    let quiet = args.iter().any(|a| a == "--quiet" || a == "-q");

    println!("rusty-metal Rust benchmarks");
    println!("  N = {n}");
    println!("  Metal GPU: initializing...");

    let mut ctx = MetalCtx::new().expect("Metal GPU not available");
    println!("  Metal GPU: {} ({})",
        ctx.device.name(),
        if ctx.device.has_unified_memory() { "unified memory" } else { "discrete" }
    );

    let mut suite = BenchSuite::new();

    let domains: Vec<(&str, fn(&mut BenchSuite, &mut MetalCtx, usize))> = vec![
        ("reductions", kernels::reductions::run),
        ("correlation", kernels::correlation::run),
        ("elementwise", kernels::elementwise::run),
        ("activations", kernels::activations::run),
        ("softmax", kernels::softmax::run),
        ("normalization", kernels::normalization::run),
        ("attention", kernels::attention::run),
        ("loss", kernels::loss::run),
        ("optimizers", kernels::optimizers::run),
        ("fft", kernels::fft::run),
        ("signal", kernels::signal::run),
        ("linalg", kernels::linalg::run),
        ("simulation", kernels::simulation::run),
        ("sorting", kernels::sorting::run),
        ("prng", kernels::prng::run),
        ("layout", kernels::layout::run),
        ("scans", kernels::scans::run),
        ("quantization", kernels::quantization::run),
        ("dot_products", kernels::dot_products::run),
        ("fused", kernels::fused::run),
    ];

    for (name, run_fn) in &domains {
        if let Some(f) = filter {
            if *name != f { continue; }
        }

        println!("\n{}", "=".repeat(60));
        println!("  {}", name.to_uppercase());
        println!("{}", "=".repeat(60));

        run_fn(&mut suite, &mut ctx, n);

        if !quiet {
            suite.print_table(Some(name));
        }
    }

    // Summary
    if !quiet {
        println!("\n{}", "=".repeat(60));
        println!("  TOTAL: {} benchmarks", suite.results.len());
        println!("{}", "=".repeat(60));
    }

    // JSON output
    let out_path = json_out.unwrap_or_else(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../results");
        std::fs::create_dir_all(&dir).ok();
        dir.join("rust_results.json")
    });

    std::fs::write(&out_path, suite.to_json()).expect("Failed to write JSON results");
    println!("\nResults written to {}", out_path.display());
}
