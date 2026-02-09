#!/usr/bin/env python3
"""
Phase 2 Gate Validation

Runs discopt solver on all available MINLPLib .nl instances, measures
gate criteria, and produces a report. Criteria that require external
infrastructure (GPU, BARON, Couenne) are reported as "not measurable."

Usage:
    python scripts/phase2_gate.py
    python scripts/phase2_gate.py --time-limit 60 --max-nodes 50000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


@dataclass
class InstanceResult:
    name: str
    n_vars: int
    n_cons: int
    status: str
    objective: float | None
    expected_obj: float | None
    correct: bool | None  # None if no reference
    wall_time: float
    node_count: int
    nodes_per_second: float
    rust_frac: float | None
    error: str | None = None


KNOWN_OPTIMA = {
    "ex1221": 7.66718007,
    "ex1225": 31.0,
    "ex1226": -17.0,
    "st_e13": 2.0,
    "st_e15": 7.66718007,
    "st_e27": 2.0,
    "st_e38": 7197.72714900,
    "st_e40": 30.41421350,
    "nvs01": 12.46966882,
    "nvs02": 5.96418452,
    "nvs03": 16.0,
    "nvs04": 0.72,
    "nvs05": 5.47093411,
    "nvs06": 1.77031250,
    "nvs07": 4.0,
    "nvs08": 23.44972735,
    "nvs10": -310.80,
    "nvs11": -431.0,
    "nvs12": -481.20,
    "nvs14": -40358.15477,
    "nvs15": 1.0,
    "nvs16": 0.70312500,
    "nvs21": -5.68478250,
    "prob03": 10.0,
    "prob06": 1.17712434,
    "prob10": 3.44550379,
    "gear": 0.0,
    "gear3": 0.0,
    "gear4": 1.64342847,
    "chance": 29.89437816,
    "dispatch": 3155.28792700,
    "meanvar": 5.24339907,
    "alan": 2.9250,
}

ABS_TOL = 1e-4
REL_TOL = 1e-3


def find_nl_instances() -> dict[str, Path]:
    """Find all unique .nl files across data directories."""
    project_root = Path(__file__).parent.parent
    nl_dirs = [
        project_root / "python" / "tests" / "data" / "minlplib",
        project_root / "python" / "tests" / "data" / "minlplib_nl",
    ]
    found = {}
    for d in nl_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.nl")):
            if f.stem not in found:
                found[f.stem] = f
    return dict(sorted(found.items()))


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Solver exceeded hard time limit")


def solve_instance(name: str, nl_path: Path, time_limit: float, max_nodes: int) -> InstanceResult:
    """Solve a single .nl instance and return metrics."""
    import signal

    import discopt.modeling as dm

    # Set hard timeout via signal (time_limit + 30s grace)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(time_limit) + 30)

    try:
        model = dm.from_nl(str(nl_path))
        n_vars = len(model._variables)
        n_cons = len(model._constraints)

        result = model.solve(
            time_limit=time_limit,
            gap_tolerance=1e-4,
            max_nodes=max_nodes,
        )

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        wt = result.wall_time if result.wall_time > 0 else 1e-10
        nc = result.node_count or 0
        nps = nc / wt if wt > 0 else 0.0
        rust_frac = result.rust_time / wt if result.rust_time else None

        expected = KNOWN_OPTIMA.get(name)
        correct = None
        if expected is not None and result.objective is not None:
            diff = abs(result.objective - expected)
            tol = ABS_TOL + REL_TOL * abs(expected)
            correct = diff <= tol

        return InstanceResult(
            name=name,
            n_vars=n_vars,
            n_cons=n_cons,
            status=result.status,
            objective=result.objective,
            expected_obj=expected,
            correct=correct,
            wall_time=result.wall_time,
            node_count=nc,
            nodes_per_second=nps,
            rust_frac=rust_frac,
        )
    except _Timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return InstanceResult(
            name=name,
            n_vars=0,
            n_cons=0,
            status="time_limit",
            objective=None,
            expected_obj=KNOWN_OPTIMA.get(name),
            correct=None,
            wall_time=time_limit,
            node_count=0,
            nodes_per_second=0.0,
            rust_frac=None,
            error="hard timeout",
        )
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return InstanceResult(
            name=name,
            n_vars=0,
            n_cons=0,
            status="error",
            objective=None,
            expected_obj=KNOWN_OPTIMA.get(name),
            correct=None,
            wall_time=0.0,
            node_count=0,
            nodes_per_second=0.0,
            rust_frac=None,
            error=str(e)[:200],
        )


def run_gate(time_limit: float, max_nodes: int) -> None:
    instances = find_nl_instances()
    print("\nPhase 2 Gate Validation")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Instances: {len(instances)}")
    print(f"Time limit: {time_limit}s, Max nodes: {max_nodes}")
    print(f"{'=' * 80}\n")

    results: list[InstanceResult] = []
    for i, (name, path) in enumerate(instances.items(), 1):
        print(f"  [{i:3d}/{len(instances)}] {name:30s}", end="", flush=True)
        r = solve_instance(name, path, time_limit, max_nodes)
        status_char = {
            "optimal": "+",
            "feasible": "~",
            "infeasible": "-",
            "time_limit": "T",
            "node_limit": "N",
            "error": "!",
        }.get(r.status, "?")
        correct_char = ""
        if r.correct is True:
            correct_char = " ok"
        elif r.correct is False:
            correct_char = " WRONG"
        print(
            f"  [{status_char}] {r.wall_time:7.2f}s  "
            f"nodes={r.node_count:6d}  n/s={r.nodes_per_second:8.1f}"
            f"{correct_char}"
        )
        results.append(r)

    # Compute gate metrics
    print(f"\n{'=' * 80}")
    print("PHASE 2 GATE RESULTS")
    print(f"{'=' * 80}\n")

    solved = [r for r in results if r.status in ("optimal", "feasible")]
    solved_30 = [r for r in solved if r.n_vars <= 30]
    solved_50 = [r for r in solved if r.n_vars <= 50]
    total_30 = [r for r in results if r.n_vars <= 30]
    total_50 = [r for r in results if r.n_vars <= 50]

    checked = [r for r in results if r.correct is not None]
    incorrect = [r for r in checked if r.correct is False]

    nps_values = [r.nodes_per_second for r in solved if r.node_count > 0]
    if nps_values:
        nps_values.sort()
        median_nps = nps_values[len(nps_values) // 2]
    else:
        median_nps = 0.0

    rust_fracs = [r.rust_frac for r in solved if r.rust_frac is not None]
    if rust_fracs:
        median_rust = sorted(rust_fracs)[len(rust_fracs) // 2]
    else:
        median_rust = None

    # Print gate criteria table
    criteria = [
        (
            "minlplib_30var_solved",
            ">=",
            55,
            len(solved_30),
            f"{len(solved_30)}/{len(total_30)} instances",
        ),
        (
            "minlplib_50var_solved",
            ">=",
            25,
            len(solved_50),
            f"{len(solved_50)}/{len(total_50)} instances",
        ),
        ("geomean_vs_couenne", "<=", 3.0, None, "NOT MEASURABLE (Couenne not installed)"),
        ("gpu_speedup", ">=", 15.0, None, "NOT MEASURABLE (JAX Metal broken on macOS)"),
        ("root_gap_vs_baron", "<=", 1.3, None, "NOT MEASURABLE (BARON not installed)"),
        ("node_throughput", ">=", 200, median_nps, f"median {median_nps:.1f} nodes/s"),
        (
            "rust_overhead",
            "<=",
            0.05,
            median_rust,
            f"median {median_rust:.4f}" if median_rust else "N/A",
        ),
        (
            "zero_incorrect",
            "<=",
            0,
            len(incorrect),
            f"{len(incorrect)} incorrect out of {len(checked)} checked",
        ),
    ]

    print(f"{'Criterion':<30s} {'Target':>12s} {'Actual':>12s} {'Status':>10s}  Notes")
    print("-" * 95)

    passed_count = 0
    failed_count = 0
    unmeasurable_count = 0
    for name, op, target, actual, notes in criteria:
        if actual is None:
            status = "SKIP"
            unmeasurable_count += 1
        elif op == ">=" and actual >= target:
            status = "PASS"
            passed_count += 1
        elif op == "<=" and actual <= target:
            status = "PASS"
            passed_count += 1
        else:
            status = "FAIL"
            failed_count += 1

        target_str = f"{op} {target}"
        actual_str = f"{actual:.4f}" if isinstance(actual, float) else str(actual)
        if actual is None:
            actual_str = "N/A"
        print(f"{name:<30s} {target_str:>12s} {actual_str:>12s} {status:>10s}  {notes}")

    print(f"\n{'=' * 80}")
    print(f"PASSED: {passed_count}  FAILED: {failed_count}  UNMEASURABLE: {unmeasurable_count}")
    if failed_count == 0:
        print("All measurable criteria PASSED.")
    else:
        print(f"{failed_count} criteria FAILED.")
    print(f"{'=' * 80}")

    # Print incorrect instances
    if incorrect:
        print("\nINCORRECT RESULTS:")
        for r in incorrect:
            print(f"  {r.name}: got {r.objective}, expected {r.expected_obj}")

    # Save results JSON
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = reports_dir / f"phase2_gate_{ts}.json"

    gate_report = {
        "timestamp": datetime.now().isoformat(),
        "gate": "phase2",
        "time_limit": time_limit,
        "max_nodes": max_nodes,
        "total_instances": len(results),
        "solved": len(solved),
        "solved_30var": len(solved_30),
        "solved_50var": len(solved_50),
        "incorrect_count": len(incorrect),
        "median_nodes_per_second": median_nps,
        "median_rust_overhead": median_rust,
        "criteria_passed": passed_count,
        "criteria_failed": failed_count,
        "criteria_unmeasurable": unmeasurable_count,
        "instances": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(gate_report, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Gate Validation")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="Time limit per instance (seconds)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200_000,
        help="Max B&B nodes per instance",
    )
    args = parser.parse_args()
    run_gate(args.time_limit, args.max_nodes)


if __name__ == "__main__":
    main()
