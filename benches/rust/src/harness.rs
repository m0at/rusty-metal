//! Benchmark harness — timing, warmup, statistics, JSON output.

use serde::Serialize;
use std::time::Instant;

pub const WARMUP_RUNS: usize = 5;
pub const BENCH_RUNS: usize = 20;
pub const DEFAULT_N: usize = 10_000_000;

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub kernel: String,
    pub domain: String,
    pub backend: String,
    pub n: usize,
    pub median_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub throughput_meps: f64,
    pub bytes_per_sec_gb: f64,
}

pub struct BenchSuite {
    pub results: Vec<BenchResult>,
}

impl BenchSuite {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn add(&mut self, result: BenchResult) {
        self.results.push(result);
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.results).unwrap()
    }

    pub fn print_table(&self, domain: Option<&str>) {
        let filtered: Vec<&BenchResult> = self
            .results
            .iter()
            .filter(|r| domain.map_or(true, |d| r.domain == d))
            .collect();

        if filtered.is_empty() {
            return;
        }

        println!(
            "{:<35} {:<14} {:>12} {:>16} {:>10}",
            "Kernel", "Backend", "Median (µs)", "Throughput", "BW (GB/s)"
        );
        println!("{}", "─".repeat(90));

        let mut current_kernel = String::new();
        for r in &filtered {
            if r.kernel != current_kernel {
                if !current_kernel.is_empty() {
                    println!();
                }
                current_kernel = r.kernel.clone();
            }
            println!(
                "{:<35} {:<14} {:>12.1} {:>12.1} Melem/s {:>10.1}",
                r.kernel, r.backend, r.median_us, r.throughput_meps, r.bytes_per_sec_gb
            );
        }
        println!();
    }
}

/// Time a closure, returning durations in microseconds.
pub fn time_fn<F: FnMut()>(mut f: F) -> Vec<f64> {
    // Warmup
    for _ in 0..WARMUP_RUNS {
        f();
    }

    let mut times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1_000_000.0); // µs
    }
    times
}

/// Benchmark a function and return a BenchResult.
pub fn bench_fn<F: FnMut()>(
    kernel: &str,
    domain: &str,
    backend: &str,
    f: F,
    n: usize,
    elem_bytes: usize,
) -> BenchResult {
    let mut times = time_fn(f);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let throughput = n as f64 / median; // Melem/s (elements per µs)
    let bw = (n * elem_bytes) as f64 / (median * 1e-6) / 1e9; // GB/s

    BenchResult {
        kernel: kernel.to_string(),
        domain: domain.to_string(),
        backend: backend.to_string(),
        n,
        median_us: (median * 10.0).round() / 10.0,
        min_us: (min * 10.0).round() / 10.0,
        max_us: (max * 10.0).round() / 10.0,
        throughput_meps: (throughput * 10.0).round() / 10.0,
        bytes_per_sec_gb: (bw * 100.0).round() / 100.0,
    }
}
