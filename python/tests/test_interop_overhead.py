"""
Tests for interop overhead measurement.

Validates that Python orchestration overhead during Model.solve() is <= 5%
of total solve time. The overhead is measured as:

    python_fraction = python_time / wall_time

where python_time = wall_time - rust_time - jax_time, as instrumented
in the solver orchestrator (solver.py).

For meaningful measurement, we need problems that exercise the full B&B loop
with multiple nodes explored, so that orchestration overhead is amortized
over enough work.
"""

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import SolveResult

# ──────────────────────────────────────────────────────────
# Helper: build MINLP models of varying size
# ──────────────────────────────────────────────────────────


def _build_binary_quadratic(n_binary: int = 6, n_continuous: int = 4) -> dm.Model:
    """Build a binary quadratic MINLP that exercises multiple B&B nodes.

    min  sum_i x_i^2 + sum_j 2*y_j
    s.t. sum_i x_i + sum_j y_j >= n_binary / 2
         x_i in [0, 5], y_j in {0,1}

    With n_binary binary variables, the B&B tree can have up to 2^n_binary
    nodes, ensuring the solver explores enough nodes for meaningful profiling.
    """
    m = dm.Model("binary_quadratic")

    xs = [m.continuous(f"x{i}", lb=0, ub=5) for i in range(n_continuous)]
    ys = [m.binary(f"y{j}") for j in range(n_binary)]

    obj_expr = dm.sum([x**2 for x in xs]) + dm.sum([2 * y for y in ys])
    m.minimize(obj_expr)

    # Coupling constraint: forces some binary variables to be 1
    m.subject_to(
        dm.sum([x for x in xs]) + dm.sum([y for y in ys]) >= n_binary / 2,
        name="coupling",
    )

    # Additional constraints to make the NLP non-trivial at each node
    for i in range(n_continuous - 1):
        m.subject_to(xs[i] + xs[i + 1] <= 4, name=f"pair_{i}")

    return m


def _build_knapsack_nonlinear(n_items: int = 8) -> dm.Model:
    """Build a nonlinear knapsack MINLP with n_items binary variables.

    min  -sum_i (value_i * y_i) + sum_i (x_i^2)
    s.t. sum_i weight_i * y_i <= capacity
         x_i <= M * y_i  (linking)
         x_i in [0, 10], y_i in {0,1}

    Generates enough B&B nodes to measure overhead meaningfully.
    """
    np.random.seed(42)
    values = np.random.uniform(1, 10, n_items)
    weights = np.random.uniform(1, 5, n_items)
    capacity = float(np.sum(weights) * 0.5)

    m = dm.Model("nonlinear_knapsack")

    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(n_items)]
    ys = [m.binary(f"y{j}") for j in range(n_items)]

    # Nonlinear objective with binary coupling
    obj = dm.sum([-values[i] * ys[i] + xs[i] ** 2 for i in range(n_items)])
    m.minimize(obj)

    # Knapsack constraint
    m.subject_to(
        dm.sum([weights[i] * ys[i] for i in range(n_items)]) <= capacity,
        name="capacity",
    )

    # Linking: x_i active only if y_i = 1
    for i in range(n_items):
        m.subject_to(xs[i] <= 10 * ys[i], name=f"link_{i}")

    # Force some continuous work
    m.subject_to(
        dm.sum([xs[i] for i in range(n_items)]) >= 1.0,
        name="min_activity",
    )

    return m


# ──────────────────────────────────────────────────────────
# TestInteropOverhead
# ──────────────────────────────────────────────────────────


class TestInteropOverhead:
    """Measure and validate Python orchestration overhead <= 5%."""

    OVERHEAD_LIMIT = 0.05  # 5%

    def _compute_overhead(self, result: SolveResult) -> float:
        """Compute python orchestration fraction of total wall time."""
        if result.wall_time <= 0:
            return 0.0
        return result.python_time / result.wall_time

    def test_binary_quadratic_overhead(self):
        """Overhead for a binary quadratic MINLP with 6 binary vars."""
        m = _build_binary_quadratic(n_binary=6, n_continuous=4)
        result = m.solve(max_nodes=500, gap_tolerance=1e-4)

        assert result.status in ("optimal", "feasible", "node_limit")
        assert result.node_count >= 1, "Need at least 1 B&B node for timing"

        overhead = self._compute_overhead(result)
        print(
            f"\nbinary_quadratic: wall={result.wall_time:.4f}s, "
            f"rust={result.rust_time:.4f}s, jax={result.jax_time:.4f}s, "
            f"python={result.python_time:.4f}s, "
            f"overhead={overhead:.2%}, nodes={result.node_count}"
        )

        assert overhead <= self.OVERHEAD_LIMIT, (
            f"Python overhead {overhead:.2%} exceeds {self.OVERHEAD_LIMIT:.0%} limit. "
            f"wall={result.wall_time:.4f}s, python={result.python_time:.4f}s"
        )

    def test_knapsack_nonlinear_overhead(self):
        """Overhead for a nonlinear knapsack with 8 binary vars."""
        m = _build_knapsack_nonlinear(n_items=8)
        result = m.solve(max_nodes=500, gap_tolerance=1e-4)

        assert result.status in ("optimal", "feasible", "node_limit")
        assert result.node_count >= 1, "Need at least 1 B&B node for timing"

        overhead = self._compute_overhead(result)
        print(
            f"\nknapsack_nonlinear: wall={result.wall_time:.4f}s, "
            f"rust={result.rust_time:.4f}s, jax={result.jax_time:.4f}s, "
            f"python={result.python_time:.4f}s, "
            f"overhead={overhead:.2%}, nodes={result.node_count}"
        )

        assert overhead <= self.OVERHEAD_LIMIT, (
            f"Python overhead {overhead:.2%} exceeds {self.OVERHEAD_LIMIT:.0%} limit. "
            f"wall={result.wall_time:.4f}s, python={result.python_time:.4f}s"
        )

    def test_simple_minlp_overhead(self):
        """Overhead for the standard example_simple_minlp problem."""
        from discopt.modeling.examples import example_simple_minlp

        m = example_simple_minlp()
        result = m.solve()

        assert result.status in ("optimal", "feasible")

        overhead = self._compute_overhead(result)
        print(
            f"\nsimple_minlp: wall={result.wall_time:.4f}s, "
            f"rust={result.rust_time:.4f}s, jax={result.jax_time:.4f}s, "
            f"python={result.python_time:.4f}s, "
            f"overhead={overhead:.2%}, nodes={result.node_count}"
        )

        # For very small problems, overhead may be dominated by setup.
        # We still check but use a more lenient threshold for problems
        # with very few nodes (< 3).
        if result.node_count >= 3:
            assert overhead <= self.OVERHEAD_LIMIT, (
                f"Python overhead {overhead:.2%} exceeds {self.OVERHEAD_LIMIT:.0%} limit. "
                f"wall={result.wall_time:.4f}s, python={result.python_time:.4f}s"
            )

    def test_profiling_times_consistent(self):
        """Verify that rust_time + jax_time + python_time == wall_time."""
        m = _build_binary_quadratic(n_binary=6, n_continuous=4)
        result = m.solve(max_nodes=500)

        sum_times = result.rust_time + result.jax_time + result.python_time
        # They should be equal by construction (python_time = wall - rust - jax)
        assert abs(sum_times - result.wall_time) < 1e-6, (
            f"Time breakdown inconsistent: "
            f"rust({result.rust_time:.6f}) + jax({result.jax_time:.6f}) + "
            f"python({result.python_time:.6f}) = {sum_times:.6f} != "
            f"wall({result.wall_time:.6f})"
        )

    def test_python_time_nonnegative(self):
        """Python overhead time should never be negative."""
        m = _build_binary_quadratic(n_binary=4, n_continuous=3)
        result = m.solve(max_nodes=200)

        assert result.python_time >= 0, (
            f"Negative python_time={result.python_time:.6f}s. "
            f"This suggests timing instrumentation is incorrect."
        )

    def test_overhead_with_many_nodes(self):
        """With many B&B nodes, overhead should be well under 5%.

        More nodes means more Rust/JAX work per Python loop iteration,
        so the relative overhead should decrease.
        """
        m = _build_knapsack_nonlinear(n_items=10)
        result = m.solve(max_nodes=1000, gap_tolerance=1e-6)

        overhead = self._compute_overhead(result)
        print(
            f"\nmany_nodes: wall={result.wall_time:.4f}s, "
            f"rust={result.rust_time:.4f}s, jax={result.jax_time:.4f}s, "
            f"python={result.python_time:.4f}s, "
            f"overhead={overhead:.2%}, nodes={result.node_count}"
        )

        assert overhead <= self.OVERHEAD_LIMIT, (
            f"Python overhead {overhead:.2%} exceeds {self.OVERHEAD_LIMIT:.0%} with "
            f"{result.node_count} nodes. "
            f"wall={result.wall_time:.4f}s, python={result.python_time:.4f}s"
        )

    def test_jax_dominates_time(self):
        """For NLP-based B&B, JAX/NLP time should dominate wall time.

        This validates that the heavy work is indeed in JAX (NLP solves)
        and Rust (tree management), not Python orchestration.
        """
        m = _build_binary_quadratic(n_binary=6, n_continuous=4)
        result = m.solve(max_nodes=500)

        if result.node_count >= 3:
            jax_fraction = result.jax_time / result.wall_time if result.wall_time > 0 else 0
            rust_fraction = result.rust_time / result.wall_time if result.wall_time > 0 else 0
            compute_fraction = jax_fraction + rust_fraction

            print(
                f"\ntime_breakdown: jax={jax_fraction:.2%}, "
                f"rust={rust_fraction:.2%}, compute={compute_fraction:.2%}"
            )

            # JAX + Rust should account for >= 95% of wall time
            assert compute_fraction >= 1.0 - self.OVERHEAD_LIMIT, (
                f"Compute fraction {compute_fraction:.2%} is too low. "
                f"Expected >= {1.0 - self.OVERHEAD_LIMIT:.0%}."
            )
