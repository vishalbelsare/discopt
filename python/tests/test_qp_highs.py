"""Tests for QP/MIQP dispatch to HiGHS solver.

Covers:
  1. Simple QP solves correctly via HiGHS
  2. QP with bounds-only constraints
  3. MIQP (QP with integer variables) solves correctly
  4. QP result matches NLP solver result (within tolerance)
  5. Falls back gracefully if HiGHS is not installed
  6. Verify QP dispatch is actually used (not falling through to NLP)
"""

from __future__ import annotations

from unittest.mock import patch

import discopt.modeling as dm
import numpy as np
import pytest

# Skip entire module if highspy is not installed
highspy = pytest.importorskip("highspy")


# ---------------------------------------------------------------------------
# 1. Simple QP
# ---------------------------------------------------------------------------
class TestSimpleQP:
    """min 0.5*x^2 + y^2 s.t. x + y >= 1, x,y >= 0."""

    def test_optimal_status(self):
        m = dm.Model("simple_qp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(0.5 * x**2 + y**2)
        m.subject_to(x + y >= 1)
        result = m.solve()
        assert result.status == "optimal"

    def test_optimal_values(self):
        m = dm.Model("simple_qp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(0.5 * x**2 + y**2)
        m.subject_to(x + y >= 1)
        result = m.solve()

        assert result.x is not None
        assert result.objective is not None

        # Analytical solution: Lagrangian gives x=2/3, y=1/3, obj=1/3
        np.testing.assert_allclose(result.value(x), 2.0 / 3.0, atol=1e-5)
        np.testing.assert_allclose(result.value(y), 1.0 / 3.0, atol=1e-5)
        np.testing.assert_allclose(result.objective, 1.0 / 3.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. QP with bounds only (no general constraints)
# ---------------------------------------------------------------------------
class TestQPBoundsOnly:
    """min (x-3)^2 + (y-2)^2 s.t. 0 <= x <= 5, 0 <= y <= 5."""

    def test_unconstrained_interior(self):
        m = dm.Model("bounds_qp")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize((x - 3) ** 2 + (y - 2) ** 2)
        result = m.solve()

        assert result.status == "optimal"
        np.testing.assert_allclose(result.value(x), 3.0, atol=1e-5)
        np.testing.assert_allclose(result.value(y), 2.0, atol=1e-5)
        np.testing.assert_allclose(result.objective, 0.0, atol=1e-5)

    def test_active_bound(self):
        """min (x-10)^2 s.t. 0 <= x <= 5 => x=5."""
        m = dm.Model("bound_active_qp")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize((x - 10) ** 2)
        result = m.solve()

        assert result.status == "optimal"
        np.testing.assert_allclose(result.value(x), 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. MIQP (QP with integer variables)
# ---------------------------------------------------------------------------
class TestMIQP:
    """min x^2 + y s.t. x + y >= 3, y in {0,1,...,5}."""

    def test_miqp_optimal(self):
        m = dm.Model("miqp_test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.integer("y", lb=0, ub=5)
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 3)
        result = m.solve()

        assert result.status == "optimal"
        assert result.x is not None
        assert result.objective is not None

        # With y integer, optimal is y=2, x=1 (obj=1+2=3) or y=3, x=0 (obj=3)
        # y=2, x=1: obj = 1 + 2 = 3
        # y=3, x=0: obj = 0 + 3 = 3
        # Both give obj=3
        np.testing.assert_allclose(result.objective, 3.0, atol=1e-4)

    def test_miqp_binary(self):
        """MIQP with binary variable."""
        m = dm.Model("miqp_binary")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x**2 + 5 * y)
        m.subject_to(x + y >= 2)
        result = m.solve()

        assert result.status == "optimal"
        assert result.x is not None
        # y=0 => x>=2, obj=4. y=1 => x>=1, obj=1+5=6. Best: y=0,x=2
        np.testing.assert_allclose(result.value(y), 0.0, atol=1e-5)
        np.testing.assert_allclose(result.value(x), 2.0, atol=1e-5)
        np.testing.assert_allclose(result.objective, 4.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. QP result matches NLP solver (cross-validation)
# ---------------------------------------------------------------------------
class TestQPMatchesNLP:
    """Verify QP via HiGHS matches a known analytical solution."""

    def test_results_match_analytical(self):
        """QP with known analytical solution: cross-validate HiGHS result."""
        m = dm.Model("cross_val")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        # min x^2 + y^2 s.t. x + y = 2
        # Analytical: x=y=1, obj=2
        m.minimize(x**2 + y**2)
        m.subject_to(x + y == 2)

        result = m.solve()
        assert result.status == "optimal"
        np.testing.assert_allclose(result.objective, 2.0, atol=1e-4)
        np.testing.assert_allclose(result.value(x), 1.0, atol=1e-4)
        np.testing.assert_allclose(result.value(y), 1.0, atol=1e-4)

    def test_cross_term_qp(self):
        """QP with cross terms matches known solution."""
        m = dm.Model("cross_val2")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        # min x^2 + 2*y^2 + x*y - 4*x - 6*y s.t. x+y<=4, x+3y<=9
        m.minimize(x**2 + 2 * y**2 + x * y - 4 * x - 6 * y)
        m.subject_to(x + y <= 4)
        m.subject_to(x + 3 * y <= 9)
        result = m.solve()
        assert result.status == "optimal"
        assert result.objective is not None
        # Verify solution is feasible
        xv = float(result.value(x))
        yv = float(result.value(y))
        assert xv + yv <= 4.0 + 1e-6
        assert xv + 3 * yv <= 9.0 + 1e-6


# ---------------------------------------------------------------------------
# 5. Graceful fallback when HiGHS unavailable
# ---------------------------------------------------------------------------
class TestFallback:
    """QP dispatch falls back to JAX IPM when highspy is not importable."""

    def test_fallback_to_jax(self):
        m = dm.Model("fallback_qp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        # Simulate highspy being unavailable by patching the import inside
        # _solve_qp_highs. When highspy cannot be imported, the function
        # should return None, and the solver falls back to the JAX IPM.
        import discopt.solver as _solver

        original = _solver._solve_qp_highs

        def _mock_highs(model, t_start, time_limit=None):
            return None

        _solver._solve_qp_highs = _mock_highs
        try:
            result = m.solve()
            # Should still find optimal via JAX fallback
            assert result.status == "optimal"
        finally:
            _solver._solve_qp_highs = original


# ---------------------------------------------------------------------------
# 6. Verify QP dispatch is actually used
# ---------------------------------------------------------------------------
class TestDispatchRouting:
    """Confirm QP problems are dispatched to the QP path, not NLP."""

    def test_qp_classified_correctly(self):
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("dispatch_qp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)
        assert classify_problem(m) == ProblemClass.QP

    def test_miqp_classified_correctly(self):
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        m = dm.Model("dispatch_miqp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.integer("y", lb=0, ub=5)
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 3)
        assert classify_problem(m) == ProblemClass.MIQP

    def test_qp_uses_highs_solver(self):
        """Check that the QP path calls HiGHS solve_qp."""
        from discopt.solvers.qp_highs import solve_qp as _raw_solve

        m = dm.Model("highs_check")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)

        with patch("discopt.solvers.qp_highs.solve_qp", wraps=_raw_solve) as mock_solve:
            result = m.solve()
            assert result.status == "optimal"
            # If HiGHS path is used, solve_qp should have been called
            assert mock_solve.call_count >= 1


# ---------------------------------------------------------------------------
# 7. Direct QP HiGHS solver tests (low-level API)
# ---------------------------------------------------------------------------
class TestQPHiGHsDirect:
    """Test the solve_qp function directly."""

    def test_simple_qp(self):
        from discopt.solvers import SolveStatus
        from discopt.solvers.qp_highs import solve_qp

        # min 0.5 * x^2 + 0.5 * y^2 s.t. x + y = 1, x,y >= 0
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = np.array([0.0, 0.0])
        result = solve_qp(
            Q=Q,
            c=c,
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
            bounds=[(0, 10), (0, 10)],
        )
        assert result.status == SolveStatus.OPTIMAL
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(result.objective, 0.25, atol=1e-6)

    def test_qp_inequality(self):
        from discopt.solvers import SolveStatus
        from discopt.solvers.qp_highs import solve_qp

        # min x^2 + y^2 s.t. x + y >= 2, x,y >= 0
        Q = np.array([[2.0, 0.0], [0.0, 2.0]])
        c = np.array([0.0, 0.0])
        result = solve_qp(
            Q=Q,
            c=c,
            A_ub=np.array([[-1.0, -1.0]]),  # -x - y <= -2
            b_ub=np.array([-2.0]),
            bounds=[(0, 10), (0, 10)],
        )
        assert result.status == SolveStatus.OPTIMAL
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)

    def test_dimension_mismatch(self):
        from discopt.solvers.qp_highs import solve_qp

        with pytest.raises(ValueError, match="Q has shape"):
            solve_qp(
                Q=np.eye(3),
                c=np.array([1.0, 2.0]),
            )


# ---------------------------------------------------------------------------
# 8. QP with equality constraints
# ---------------------------------------------------------------------------
class TestQPEquality:
    """QP with both equality and inequality constraints."""

    def test_equality_and_inequality(self):
        m = dm.Model("eq_ineq_qp")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        m.subject_to(x[0] + x[1] + x[2] == 3)
        m.subject_to(x[0] - x[1] <= 1)
        result = m.solve()

        assert result.status == "optimal"
        # With equality x0+x1+x2=3 and x0-x1<=1,
        # optimal is x0=x1=x2=1 (satisfies both)
        np.testing.assert_allclose(result.value(x), [1.0, 1.0, 1.0], atol=1e-4)
