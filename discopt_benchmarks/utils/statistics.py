"""
Performance Profiles and Statistical Analysis

Implements Dolan-Moré performance profiles and related
statistical tools for solver comparison.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def dolan_more_profile(
    solver_times: dict[str, dict[str, float]],
    instances: list[str],
    tau_max: float = 1000.0,
    tau_points: int = 500,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute Dolan-Moré performance profiles.

    Args:
        solver_times: {solver_name: {instance_name: solve_time}}
                      Use inf for unsolved instances.
        instances: List of all instance names.
        tau_max: Maximum performance ratio to consider.
        tau_points: Number of points in the profile curve.

    Returns:
        {solver_name: (tau_array, fraction_array)}

    Reference:
        Dolan, E.D. and Moré, J.J. (2002). "Benchmarking optimization
        software with performance profiles." Mathematical Programming.
    """
    solvers = list(solver_times.keys())

    # Compute best time per instance across all solvers
    best_time = {}
    for inst in instances:
        times = [
            solver_times[s].get(inst, float("inf"))
            for s in solvers
        ]
        finite_times = [t for t in times if math.isfinite(t)]
        best_time[inst] = max(min(finite_times), 1e-6) if finite_times else float("inf")

    # Compute performance ratios
    ratios: dict[str, list[float]] = {s: [] for s in solvers}
    for inst in instances:
        bt = best_time[inst]
        for s in solvers:
            t = solver_times[s].get(inst, float("inf"))
            if math.isfinite(t) and math.isfinite(bt):
                ratios[s].append(t / bt)
            else:
                ratios[s].append(float("inf"))

    # Build profile curves
    tau_values = np.concatenate([
        np.array([1.0]),
        np.logspace(np.log10(1.001), np.log10(tau_max), tau_points - 1),
    ])

    profiles = {}
    n = len(instances)
    for s in solvers:
        sorted_ratios = np.sort(ratios[s])
        fractions = np.searchsorted(sorted_ratios, tau_values, side="right") / n
        profiles[s] = (tau_values, fractions)

    return profiles


def shifted_geomean(values: list[float], shift: float = 1.0) -> float:
    """
    Shifted geometric mean: exp(mean(log(x + shift))) - shift.

    Standard in solver benchmarking. shift=1.0 is conventional.
    """
    valid = [v for v in values if math.isfinite(v) and v >= 0]
    if not valid:
        return float("inf")
    log_sum = sum(math.log(v + shift) for v in valid)
    return math.exp(log_sum / len(valid)) - shift


def paired_comparison(
    times_a: dict[str, float],
    times_b: dict[str, float],
    shift: float = 1.0,
) -> dict:
    """
    Detailed paired comparison of two solvers.

    Returns comprehensive statistics over commonly-solved instances.
    """
    common = set(times_a.keys()) & set(times_b.keys())
    # Only consider instances solved by both
    common_solved = [
        i for i in common
        if math.isfinite(times_a[i]) and math.isfinite(times_b[i])
    ]

    if not common_solved:
        return {
            "common_count": 0,
            "geomean_ratio": float("nan"),
            "a_faster_count": 0,
            "b_faster_count": 0,
            "a_only_solved": len([i for i in times_a if math.isfinite(times_a[i]) and (i not in times_b or not math.isfinite(times_b[i]))]),
            "b_only_solved": len([i for i in times_b if math.isfinite(times_b[i]) and (i not in times_a or not math.isfinite(times_a[i]))]),
        }

    ta = [times_a[i] for i in common_solved]
    tb = [times_b[i] for i in common_solved]

    ratios = [times_a[i] / max(times_b[i], 1e-6) for i in common_solved]
    log_ratios = [math.log(r) for r in ratios if r > 0]

    return {
        "common_count": len(common_solved),
        "geomean_a": shifted_geomean(ta, shift),
        "geomean_b": shifted_geomean(tb, shift),
        "geomean_ratio": shifted_geomean(ta, shift) / max(shifted_geomean(tb, shift), 1e-10),
        "a_faster_count": sum(1 for r in ratios if r < 1.0),
        "b_faster_count": sum(1 for r in ratios if r > 1.0),
        "tie_count": sum(1 for r in ratios if abs(r - 1.0) < 0.01),
        "median_ratio": float(np.median(ratios)),
        "max_speedup_a": min(ratios),       # Smallest ratio = A's best showing
        "max_speedup_b": max(ratios),       # Largest ratio = B's best showing
        "a_only_solved": len([i for i in times_a if math.isfinite(times_a[i]) and (i not in times_b or not math.isfinite(times_b.get(i, float("inf"))))]),
        "b_only_solved": len([i for i in times_b if math.isfinite(times_b[i]) and (i not in times_a or not math.isfinite(times_a.get(i, float("inf"))))]),
    }


def wilcoxon_signed_rank(
    times_a: dict[str, float],
    times_b: dict[str, float],
) -> dict:
    """
    Wilcoxon signed-rank test on log-time ratios.

    Non-parametric test for whether one solver is statistically
    significantly faster than the other.
    """
    from scipy import stats

    common = [
        i for i in set(times_a.keys()) & set(times_b.keys())
        if math.isfinite(times_a[i]) and math.isfinite(times_b[i])
    ]
    if len(common) < 5:
        return {"p_value": float("nan"), "statistic": float("nan"), "n": len(common)}

    log_ratios = [
        math.log(max(times_a[i], 1e-6)) - math.log(max(times_b[i], 1e-6))
        for i in common
    ]

    stat, p_value = stats.wilcoxon(log_ratios, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": len(common),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "direction": "a_faster" if np.mean(log_ratios) < 0 else "b_faster",
    }


def instance_difficulty_classification(
    solver_times: dict[str, dict[str, float]],
    instances: list[str],
) -> dict[str, list[str]]:
    """
    Classify instances by difficulty based on how many solvers solve them.

    Categories:
    - easy: All solvers solve in <10s
    - medium: Most solvers solve, some take >10s
    - hard: Only some solvers solve
    - open: No solver finds global optimum
    """
    easy, medium, hard, open_instances = [], [], [], []

    for inst in instances:
        solved_by = []
        for solver, times in solver_times.items():
            t = times.get(inst, float("inf"))
            if math.isfinite(t):
                solved_by.append((solver, t))

        if not solved_by:
            open_instances.append(inst)
        elif len(solved_by) == len(solver_times) and all(t < 10 for _, t in solved_by):
            easy.append(inst)
        elif len(solved_by) > len(solver_times) / 2:
            medium.append(inst)
        else:
            hard.append(inst)

    return {
        "easy": easy,
        "medium": medium,
        "hard": hard,
        "open": open_instances,
    }
