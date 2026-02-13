"""Entry point: run all kernel benchmarks and emit results."""

import argparse
import json
import sys
from pathlib import Path

from bench.harness import SUITE
from bench.kernels import (
    reductions,
    correlation,
    elementwise,
    activations,
    softmax,
    normalization,
    attention,
    loss,
    optimizers,
    fft,
    signal,
    linalg,
    simulation,
    sorting,
    prng,
    layout,
    scans,
    quantization,
    dot_products,
    fused,
    fused_custom,
)

DOMAINS = {
    "reductions": reductions,
    "correlation": correlation,
    "elementwise": elementwise,
    "activations": activations,
    "softmax": softmax,
    "normalization": normalization,
    "attention": attention,
    "loss": loss,
    "optimizers": optimizers,
    "fft": fft,
    "signal": signal,
    "linalg": linalg,
    "simulation": simulation,
    "sorting": sorting,
    "prng": prng,
    "layout": layout,
    "scans": scans,
    "quantization": quantization,
    "dot_products": dot_products,
    "fused": fused,
    "fused_custom": fused_custom,
}


def main():
    parser = argparse.ArgumentParser(description="rusty-metal Python benchmarks")
    parser.add_argument(
        "--domain", "-d",
        nargs="*",
        choices=list(DOMAINS.keys()),
        help="Run only specific domains (default: all)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Override element count for benchmarks",
    )
    parser.add_argument(
        "--json", "-j", dest="json_out",
        type=str, default=None,
        help="Write JSON results to file",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress table output",
    )
    args = parser.parse_args()

    domains_to_run = args.domain if args.domain else list(DOMAINS.keys())
    kwargs = {}
    if args.n is not None:
        kwargs["n"] = args.n

    for name in domains_to_run:
        mod = DOMAINS[name]
        print(f"\n{'=' * 60}")
        print(f"  {name.upper()}")
        print(f"{'=' * 60}")
        try:
            mod.run(**kwargs)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        if not args.quiet:
            SUITE.print_table(domain=name)

    # Summary
    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"  TOTAL: {len(SUITE.results)} benchmarks")
        print(f"{'=' * 60}")

    # JSON output
    if args.json_out:
        Path(args.json_out).write_text(SUITE.to_json())
        print(f"\nResults written to {args.json_out}")
    else:
        # Also write to default location
        out_dir = Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "python_results.json"
        out_file.write_text(SUITE.to_json())
        print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    main()
