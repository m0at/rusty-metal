#!/usr/bin/env python3
"""Compare Python and Rust benchmark results side-by-side."""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--python", "-p", required=True, help="Python results JSON")
    parser.add_argument("--rust", "-r", required=True, help="Rust results JSON")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    py_results = load_results(args.python)
    rs_results = load_results(args.rust)

    # Build lookup: kernel -> backend -> result
    all_results: dict[str, dict[str, dict]] = {}
    for r in py_results + rs_results:
        kernel = r["kernel"]
        backend = r["backend"]
        all_results.setdefault(kernel, {})[backend] = r

    # Backend display order
    backend_order = [
        "numpy", "torch_cpu", "torch_mps", "mlx",
        "rust_scalar", "neon_simd", "metal",
    ]

    # Build output
    lines = []
    lines.append("rusty-metal Benchmark Comparison")
    lines.append("=" * 120)
    lines.append("")

    # Group by domain
    domains: dict[str, list[str]] = {}
    for r in py_results + rs_results:
        domains.setdefault(r["domain"], set()).add(r["kernel"])
    # Convert sets to sorted lists
    domains = {d: sorted(kernels) for d, kernels in domains.items()}

    for domain, kernels in sorted(domains.items()):
        lines.append(f"  {domain.upper()}")
        lines.append("─" * 120)

        header = f"  {'Kernel':<32}"
        for b in backend_order:
            header += f" {b:>12}"
        lines.append(header)

        sub_header = f"  {'':<32}"
        for _ in backend_order:
            sub_header += f" {'(µs)':>12}"
        lines.append(sub_header)
        lines.append("  " + "─" * 116)

        for kernel in kernels:
            backends = all_results.get(kernel, {})
            line = f"  {kernel:<32}"
            best_time = float("inf")
            for b in backend_order:
                if b in backends:
                    best_time = min(best_time, backends[b]["median_us"])

            for b in backend_order:
                if b in backends:
                    t = backends[b]["median_us"]
                    # Mark fastest with *
                    marker = "*" if t == best_time else " "
                    line += f" {t:>11.1f}{marker}"
                else:
                    line += f" {'—':>12}"
            lines.append(line)

        lines.append("")

    # Summary statistics
    lines.append("=" * 120)
    lines.append("  SUMMARY")
    lines.append("─" * 120)

    # Count wins per backend
    wins: dict[str, int] = {b: 0 for b in backend_order}
    total_kernels = 0
    for kernel, backends in all_results.items():
        if len(backends) < 2:
            continue
        total_kernels += 1
        best_time = float("inf")
        best_backend = ""
        for b in backend_order:
            if b in backends and backends[b]["median_us"] < best_time:
                best_time = backends[b]["median_us"]
                best_backend = b
        if best_backend:
            wins[best_backend] += 1

    lines.append(f"  Total kernels compared: {total_kernels}")
    lines.append(f"  Fastest backend wins:")
    for b in backend_order:
        if wins[b] > 0:
            pct = 100.0 * wins[b] / max(total_kernels, 1)
            bar = "█" * int(pct / 2)
            lines.append(f"    {b:<14} {wins[b]:>4} wins ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("  * = fastest for that kernel")
    lines.append("")

    output = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output)
        print(output)
        print(f"\nSaved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
