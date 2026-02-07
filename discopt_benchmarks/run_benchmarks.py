#!/usr/bin/env python3
"""
discopt Benchmark CLI

Usage:
    # Run smoke tests (quick sanity check)
    python run_benchmarks.py --suite smoke

    # Run phase gate check
    python run_benchmarks.py --gate phase1

    # Run comparison benchmark with report
    python run_benchmarks.py --suite comparison --solvers discopt,baron --report

    # Run nightly regression check
    python run_benchmarks.py --suite nightly --baseline reports/latest.json

    # List available suites
    python run_benchmarks.py --list-suites
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.metrics import BenchmarkResults, evaluate_phase_gate


def main():
    parser = argparse.ArgumentParser(
        description="discopt Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite", type=str, default="smoke",
        help="Benchmark suite to run (smoke, phase1, phase2, phase3, full, nightly, comparison)"
    )
    parser.add_argument(
        "--gate", type=str, default=None,
        help="Evaluate phase gate criteria (phase1, phase2, phase3, phase4)"
    )
    parser.add_argument(
        "--solvers", type=str, default="discopt",
        help="Comma-separated list of solvers to benchmark"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate markdown report after benchmarking"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline results JSON for regression detection"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--list-suites", action="store_true",
        help="List available benchmark suites"
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: compact JSON output for pipeline integration"
    )

    args = parser.parse_args()

    if args.list_suites:
        _list_suites()
        return

    if args.gate:
        _run_gate_check(args)
        return

    _run_benchmark(args)


def _list_suites():
    """List available benchmark suites."""
    suites = {
        "smoke":       "Quick sanity check (10 instances, 60s limit)",
        "phase1":      "Phase 1 validation (small instances, ≤10 vars)",
        "phase2":      "Phase 2 validation (medium instances, ≤50 vars)",
        "phase3":      "Phase 3 validation (large instances, ≤100 vars)",
        "full":        "Complete MINLPLib (~1700 instances)",
        "comparison":  "Head-to-head solver comparison (curated set)",
        "nightly":     "Nightly CI regression suite (100 instances)",
        "lp_netlib":   "Rust LP solver vs Netlib",
        "nlp_cutest":  "Hybrid NLP solver vs CUTEst",
        "pooling":     "GPU-amenable pooling problems",
        "gpu_scaling":  "GPU batching scalability measurement",
    }
    print("\nAvailable benchmark suites:\n")
    for name, desc in suites.items():
        print(f"  {name:15s}  {desc}")
    print()


def _run_gate_check(args):
    """Run phase gate evaluation."""
    print(f"\n{'='*60}")
    print(f"Phase Gate Check: {args.gate}")
    print(f"{'='*60}\n")

    # Load latest results
    results_path = Path(args.output) if args.output else _find_latest_results(args.gate)
    if not results_path or not results_path.exists():
        print(f"ERROR: No results found for gate '{args.gate}'.")
        print(f"Run benchmarks first: python run_benchmarks.py --suite {args.gate}")
        sys.exit(1)

    benchmark = BenchmarkResults.load(results_path)

    # Load gate config
    gate_config = _load_gate_config(args.gate)
    if not gate_config:
        print(f"ERROR: No gate configuration found for '{args.gate}'")
        sys.exit(1)

    all_passed, criteria = evaluate_phase_gate(
        args.gate, benchmark, gate_config
    )

    # Display results
    print(f"{'Criterion':<40s} {'Target':>12s} {'Actual':>12s} {'Status':>8s}")
    print("-" * 75)
    for c in criteria:
        target_str = f"{'≥' if c.direction == 'min' else '≤'} {c.target}"
        actual_str = f"{c.actual:.4f}" if isinstance(c.actual, float) else str(c.actual)
        status = "✅ PASS" if c.passed else "🔴 FAIL"
        print(f"{c.name:<40s} {target_str:>12s} {actual_str:>12s} {status:>8s}")

    print()
    if all_passed:
        print("✅ ALL CRITERIA PASSED — proceed to next phase")
        sys.exit(0)
    else:
        failed = [c for c in criteria if not c.passed]
        print(f"🔴 {len(failed)} CRITERIA FAILED — do not proceed")
        sys.exit(1)


def _run_benchmark(args):
    """Run a benchmark suite."""
    print(f"\ndiscopt Benchmark Runner")
    print(f"Suite: {args.suite}")
    print(f"Solvers: {args.solvers}")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Placeholder: actual benchmark execution
    print("NOTE: Benchmark execution requires discopt to be installed.")
    print("This framework is ready to use once the solver is available.")
    print()
    print("To run tests against the framework itself:")
    print("  pytest tests/ -v -m 'not slow'")
    print()
    print("To check CI readiness:")
    print("  pytest tests/ -v -m smoke")


def _find_latest_results(suite: str) -> Path | None:
    """Find the most recent results file for a suite."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob(f"{suite}_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _load_gate_config(gate_name: str) -> dict | None:
    """Load gate configuration from TOML config."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = Path("config/benchmarks.toml")
    if not config_path.exists():
        return None
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("gates", {}).get(gate_name)


if __name__ == "__main__":
    main()
