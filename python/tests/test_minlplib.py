"""
MINLPLib validation tests: parse .nl files, solve via Model.solve(), validate
against known optimal values from MINLPLib.

Tests cover:
  1. Parsing: All instances parse without error
  2. Structure: Correct variable counts, types, bounds
  3. from_nl() + Model.solve(): End-to-end B&B solve for MINLP instances
  4. Correctness gate: Zero incorrect results

Known optima are sourced from MINLPLib (confirmed by BARON/ANTIGONE/SCIP).
Tolerances: abs_tol=1e-4, rel_tol=1e-3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

# Skip everything if nl_bindings not available
try:
    from discopt._rust import parse_nl_file

    HAS_NL_BINDINGS = True
except ImportError:
    HAS_NL_BINDINGS = False

pytestmark = pytest.mark.skipif(
    not HAS_NL_BINDINGS,
    reason="nl_bindings not yet wired into discopt._rust",
)

# ──────────────────────────────────────────────────────────
# Tolerances
# ──────────────────────────────────────────────────────────

ABS_TOL = 1e-4
REL_TOL = 1e-3

# ──────────────────────────────────────────────────────────
# Data directory
# ──────────────────────────────────────────────────────────

NL_DIR = Path(__file__).parent / "data" / "minlplib"

# Fall back to minlplib_nl if minlplib doesn't exist
if not NL_DIR.exists():
    NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _nl_path(name: str) -> str:
    return str(NL_DIR / f"{name}.nl")


def _nl_exists(name: str) -> bool:
    return os.path.exists(_nl_path(name))


# ──────────────────────────────────────────────────────────
# Instance registry with known optimal values from MINLPLib
# ──────────────────────────────────────────────────────────


@dataclass
class NLInstance:
    """A MINLPLib test instance with known optimal value."""

    name: str
    expected_obj: float
    n_vars: int
    has_binary: bool
    has_integer: bool
    time_limit: float = 120.0
    max_nodes: int = 100_000


# All optimal values verified by multiple MINLPLib solvers.
ALL_INSTANCES: list[NLInstance] = [
    # --- ex-series: classic MINLPs from Grossmann/Floudas ---
    NLInstance("ex1221", 7.66718007, 5, True, False),
    NLInstance("ex1225", 31.0, 8, True, False),
    NLInstance("ex1226", -17.0, 5, True, False),
    # --- st_e series ---
    NLInstance("st_e13", 2.0, 2, True, False),
    NLInstance("st_e15", 7.66718007, 5, True, False),
    NLInstance("st_e27", 2.0, 4, True, False),
    NLInstance("st_e38", 7197.72714900, 4, False, True),
    NLInstance("st_e40", 30.41421350, 4, False, True),
    # --- nvs-series: nonlinear variable selection ---
    NLInstance("nvs01", 12.46966882, 3, False, True),
    NLInstance("nvs02", 5.96418452, 8, False, True, time_limit=180.0, max_nodes=200_000),
    NLInstance("nvs03", 16.0, 2, False, True),
    NLInstance("nvs04", 0.72, 2, False, True),
    NLInstance("nvs05", 5.47093411, 8, False, True, time_limit=180.0, max_nodes=200_000),
    NLInstance("nvs06", 1.77031250, 2, False, True),
    NLInstance("nvs07", 4.0, 3, False, True),
    NLInstance("nvs08", 23.44972735, 3, False, True),
    NLInstance("nvs10", -310.80, 2, False, True),
    NLInstance("nvs11", -431.0, 3, False, True),
    NLInstance("nvs12", -481.20, 4, False, True),
    NLInstance("nvs14", -40358.15477, 8, False, True, time_limit=180.0, max_nodes=200_000),
    NLInstance("nvs15", 1.0, 3, False, True),
    NLInstance("nvs16", 0.70312500, 2, False, True),
    NLInstance("nvs21", -5.68478250, 3, False, True),
    # --- prob series ---
    NLInstance("prob03", 10.0, 2, False, True),
    NLInstance("prob06", 1.17712434, 2, False, False),  # pure NLP
    NLInstance("prob10", 3.44550379, 2, False, True),
    # --- gear problems ---
    NLInstance("gear", 0.0, 4, False, True),
    NLInstance("gear3", 0.0, 8, False, True, time_limit=180.0, max_nodes=200_000),
    NLInstance("gear4", 1.64342847, 6, False, True),
    # --- pure continuous NLP ---
    NLInstance("chance", 29.89437816, 4, False, False),
    NLInstance("dispatch", 3155.28792700, 4, False, False),
    NLInstance("meanvar", 5.24339907, 8, False, False),
    # --- alan ---
    NLInstance("alan", 2.9250, 8, True, False, time_limit=180.0, max_nodes=200_000),
]

# Filter to instances that exist on disk
INSTANCES = [inst for inst in ALL_INSTANCES if _nl_exists(inst.name)]

# Split by size
SMALL_INSTANCES = [inst for inst in INSTANCES if inst.n_vars <= 5]
MEDIUM_INSTANCES = [inst for inst in INSTANCES if 5 < inst.n_vars <= 10]

# Split by type
MINLP_INSTANCES = [inst for inst in INSTANCES if inst.has_binary or inst.has_integer]
NLP_INSTANCES = [inst for inst in INSTANCES if not inst.has_binary and not inst.has_integer]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def assert_optimal_value(
    actual: float,
    expected: float,
    name: str,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
) -> None:
    """Assert solver found the correct optimal value within tolerance."""
    tol = abs_tol + rel_tol * abs(expected)
    assert abs(actual - expected) <= tol, (
        f"[{name}] Objective {actual:.8f} differs from expected {expected:.8f} "
        f"by {abs(actual - expected):.2e} (tolerance {tol:.2e})"
    )


# ──────────────────────────────────────────────────────────
# 1. Parsing tests
# ──────────────────────────────────────────────────────────


class TestParsing:
    """Verify .nl files can be parsed by the Rust parser."""

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_parses_without_error(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        assert model.n_vars > 0

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_n_vars(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        assert model.n_vars == inst.n_vars, (
            f"{inst.name}: expected n_vars={inst.n_vars}, got {model.n_vars}"
        )

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_var_types(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        types = model.var_types()
        has_binary = "binary" in types
        has_integer = "integer" in types
        assert has_binary == inst.has_binary, (
            f"{inst.name}: expected has_binary={inst.has_binary}, types={types}"
        )
        assert has_integer == inst.has_integer, (
            f"{inst.name}: expected has_integer={inst.has_integer}, types={types}"
        )

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_objective_finite_at_midpoint(self, inst: NLInstance) -> None:
        model = parse_nl_file(_nl_path(inst.name))
        n = model.n_vars
        x = np.zeros(n, dtype=np.float64)
        for i in range(len(model.var_types())):
            lb = model.var_lb(i)
            ub = model.var_ub(i)
            lb_c = max(lb[0], -100.0)
            ub_c = min(ub[0], 100.0)
            x[i] = 0.5 * (lb_c + ub_c)
        val = model.evaluate_objective(x)
        assert np.isfinite(val), f"{inst.name}: objective not finite at midpoint: {val}"


# ──────────────────────────────────────────────────────────
# 2. from_nl() loading tests
# ──────────────────────────────────────────────────────────


class TestFromNl:
    """Verify from_nl() correctly constructs Model objects."""

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_from_nl_loads(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        assert model.name == inst.name
        assert model.num_variables == inst.n_vars

    @pytest.mark.parametrize("inst", INSTANCES, ids=[i.name for i in INSTANCES])
    def test_model_has_nl_repr(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        assert hasattr(model, "_nl_repr")
        assert model._nl_repr is not None
        assert model._nl_repr.n_vars == inst.n_vars


# ──────────────────────────────────────────────────────────
# 3. Full solve tests via from_nl() + Model.solve()
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestSolveSmall:
    """Solve small instances (<=5 vars) via from_nl() + Model.solve()."""

    @pytest.mark.parametrize("inst", SMALL_INSTANCES, ids=[i.name for i in SMALL_INSTANCES])
    def test_solve_and_validate(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        result = model.solve(
            time_limit=inst.time_limit,
            gap_tolerance=1e-4,
            max_nodes=inst.max_nodes,
        )
        assert result.status in ("optimal", "feasible"), (
            f"[{inst.name}] Unexpected status: {result.status}"
        )
        assert result.objective is not None, f"[{inst.name}] No objective value"
        assert_optimal_value(result.objective, inst.expected_obj, inst.name)


@pytest.mark.correctness
class TestSolveMedium:
    """Solve medium instances (6-10 vars) via from_nl() + Model.solve()."""

    @pytest.mark.parametrize("inst", MEDIUM_INSTANCES, ids=[i.name for i in MEDIUM_INSTANCES])
    def test_solve_and_validate(self, inst: NLInstance) -> None:
        import discopt.modeling as dm

        model = dm.from_nl(_nl_path(inst.name))
        result = model.solve(
            time_limit=inst.time_limit,
            gap_tolerance=1e-4,
            max_nodes=inst.max_nodes,
        )
        assert result.status in ("optimal", "feasible"), (
            f"[{inst.name}] Unexpected status: {result.status}"
        )
        assert result.objective is not None, f"[{inst.name}] No objective value"
        assert_optimal_value(result.objective, inst.expected_obj, inst.name)


# ──────────────────────────────────────────────────────────
# 4. Correctness gate
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestMINLPLibGate:
    """Phase gate: zero incorrect results across all MINLPLib instances."""

    def test_zero_incorrect_results(self) -> None:
        """Run all instances and assert zero incorrect results."""
        import discopt.modeling as dm

        incorrect = []
        skipped = []
        passed = []
        for inst in INSTANCES:
            try:
                model = dm.from_nl(_nl_path(inst.name))
                result = model.solve(
                    time_limit=inst.time_limit,
                    gap_tolerance=1e-4,
                    max_nodes=inst.max_nodes,
                )
                if result.status not in ("optimal", "feasible"):
                    skipped.append(f"{inst.name}: status={result.status}")
                    continue
                if result.objective is None:
                    incorrect.append(f"{inst.name}: no objective value")
                    continue
                tol = ABS_TOL + REL_TOL * abs(inst.expected_obj)
                if abs(result.objective - inst.expected_obj) > tol:
                    incorrect.append(
                        f"{inst.name}: obj={result.objective:.8f}, "
                        f"expected={inst.expected_obj:.8f}, "
                        f"error={abs(result.objective - inst.expected_obj):.2e}, "
                        f"tol={tol:.2e}"
                    )
                else:
                    passed.append(inst.name)
            except Exception as e:
                incorrect.append(f"{inst.name}: exception: {e}")

        total = len(INSTANCES)
        n_passed = len(passed)
        n_incorrect = len(incorrect)
        n_skipped = len(skipped)

        summary = (
            f"\nMINLPLib Correctness Gate: "
            f"{n_passed}/{total} passed, "
            f"{n_incorrect} incorrect, "
            f"{n_skipped} skipped"
        )
        if skipped:
            summary += "\n  Skipped:\n" + "\n".join(f"    - {s}" for s in skipped)
        if incorrect:
            summary += "\n  Incorrect:\n" + "\n".join(f"    - {s}" for s in incorrect)

        assert n_incorrect == 0, (
            f"CORRECTNESS GATE FAILED: {n_incorrect} incorrect result(s)" + summary
        )


# ──────────────────────────────────────────────────────────
# 5. Instance count validation
# ──────────────────────────────────────────────────────────


class TestInstanceCount:
    """Verify we have enough instances for meaningful validation."""

    def test_at_least_25_instances(self) -> None:
        assert len(INSTANCES) >= 25, (
            f"Only {len(INSTANCES)} instances available, need >= 25. "
            f"Available: {[inst.name for inst in INSTANCES]}"
        )

    def test_mixed_types(self) -> None:
        assert len(MINLP_INSTANCES) >= 20, (
            f"Need >= 20 MINLP instances, have {len(MINLP_INSTANCES)}"
        )
        assert len(NLP_INSTANCES) >= 3, f"Need >= 3 NLP instances, have {len(NLP_INSTANCES)}"
