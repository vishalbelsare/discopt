"""Tests for the fast model construction API (add_linear_constraints, etc.)."""

import time

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────
# Basic linear constraints
# ─────────────────────────────────────────────────────────────


class TestLinearConstraints:
    """Tests for Model.add_linear_constraints()."""

    def test_basic(self):
        """100-row LP via fast API, correct n_constraints."""
        m = dm.Model("basic_lp")
        x = m.continuous("x", shape=(10,), lb=0, ub=100)

        A = sp.random(100, 10, density=0.3, format="csr", random_state=42)
        b = np.ones(100)
        m.add_linear_constraints(A, x, "<=", b, name="cap")
        m.add_linear_objective(np.ones(10), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == 100
        assert repr_.n_vars == 10

    def test_all_senses(self):
        """<= , ==, >= all produce correct ConstraintSense."""
        for sense_str in ("<=", "==", ">="):
            m = dm.Model(f"sense_{sense_str}")
            x = m.continuous("x", shape=(5,), lb=0, ub=10)
            A = sp.eye(5, format="csr")
            b = np.ones(5)
            m.add_linear_constraints(A, x, sense_str, b)
            m.add_linear_objective(np.ones(5), x)

            from discopt._rust import model_to_repr

            repr_ = model_to_repr(m, m._builder)
            for i in range(5):
                assert repr_.constraint_sense(i) == sense_str

    def test_dimension_mismatch(self):
        """Wrong A.shape[1] vs x.size raises ValueError."""
        m = dm.Model("mismatch")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        A = sp.random(10, 7, density=0.3, format="csr")
        b = np.ones(10)
        with pytest.raises(ValueError, match="columns"):
            m.add_linear_constraints(A, x, "<=", b)

    def test_sparse_format_tolerance(self):
        """COO, CSC, dense all auto-converted to CSR."""
        for fmt in ("coo", "csc"):
            m = dm.Model(f"fmt_{fmt}")
            x = m.continuous("x", shape=(5,), lb=0, ub=10)
            A = sp.random(3, 5, density=0.5, format=fmt, random_state=42)
            b = np.ones(3)
            m.add_linear_constraints(A, x, "<=", b)
            m.add_linear_objective(np.ones(5), x)

            from discopt._rust import model_to_repr

            repr_ = model_to_repr(m, m._builder)
            assert repr_.n_constraints == 3

        # Dense numpy array
        m = dm.Model("dense")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        A_dense = np.random.randn(3, 5)
        b = np.ones(3)
        m.add_linear_constraints(A_dense, x, "<=", b)
        m.add_linear_objective(np.ones(5), x)

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == 3

    def test_empty_row(self):
        """CSR row with zero nonzeros produces trivial constraint."""
        m = dm.Model("empty_row")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        # A with one empty row
        A = sp.csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))
        b = np.array([5.0, 0.0, 3.0])
        m.add_linear_constraints(A, x, "<=", b)
        m.add_linear_objective(np.ones(3), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == 3
        # Empty row evaluates to 0.0
        x_test = np.zeros(3)
        assert repr_.evaluate_constraint(1, x_test) == pytest.approx(0.0)

    def test_scalar_b_broadcast(self):
        """Scalar b is broadcast to all rows."""
        m = dm.Model("scalar_b")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        A = sp.eye(3, 5, format="csr")
        m.add_linear_constraints(A, x, "<=", 1.0)
        m.add_linear_objective(np.ones(5), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == 3
        for i in range(3):
            assert repr_.constraint_rhs(i) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────
# Objectives
# ─────────────────────────────────────────────────────────────


class TestObjectives:
    """Tests for add_linear_objective and add_quadratic_objective."""

    def test_linear_objective(self):
        """add_linear_objective evaluates correctly."""
        m = dm.Model("lin_obj")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        c = np.array([1.0, 2.0, 3.0])
        m.add_linear_objective(c, x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        x_test = np.array([1.0, 2.0, 3.0])
        # c'x = 1*1 + 2*2 + 3*3 = 14
        assert repr_.evaluate_objective(x_test) == pytest.approx(14.0)
        assert repr_.objective_sense == "minimize"

    def test_linear_objective_with_constant(self):
        """add_linear_objective with constant offset."""
        m = dm.Model("lin_obj_const")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        c = np.array([1.0, 1.0])
        m.add_linear_objective(c, x, constant=5.0)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        x_test = np.array([3.0, 4.0])
        assert repr_.evaluate_objective(x_test) == pytest.approx(12.0)

    def test_linear_objective_maximize(self):
        """add_linear_objective with maximize sense."""
        m = dm.Model("max_obj")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.add_linear_objective(np.array([1.0, 1.0]), x, sense="maximize")

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.objective_sense == "maximize"

    def test_quadratic_objective(self):
        """add_quadratic_objective evaluates correctly."""
        m = dm.Model("quad_obj")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        # Q = [[2, 0], [0, 4]], c = [1, 1]
        # obj = 0.5 * x' Q x + c'x = 0.5*(2*x0^2 + 4*x1^2) + x0 + x1
        Q = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))
        c = np.array([1.0, 1.0])
        m.add_quadratic_objective(Q, c, x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        x_test = np.array([1.0, 2.0])
        # 0.5*(2*1 + 4*4) + 1 + 2 = 0.5*18 + 3 = 12
        assert repr_.evaluate_objective(x_test) == pytest.approx(12.0)

    def test_quadratic_objective_offdiag(self):
        """add_quadratic_objective with off-diagonal terms."""
        m = dm.Model("quad_offdiag")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        # Q = [[2, 1], [1, 2]], c = [0, 0]
        # obj = 0.5 * (2*x0^2 + 2*x0*x1 + 2*x1^2)
        Q = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
        c = np.zeros(2)
        m.add_quadratic_objective(Q, c, x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        x_test = np.array([1.0, 2.0])
        # 0.5 * (2*1 + 2*1*2 + 2*4) = 0.5 * (2 + 4 + 8) = 7
        assert repr_.evaluate_objective(x_test) == pytest.approx(7.0)

    def test_objective_dimension_mismatch(self):
        """Wrong c size raises ValueError."""
        m = dm.Model("mismatch")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        with pytest.raises(ValueError, match="elements"):
            m.add_linear_objective(np.ones(7), x)


# ─────────────────────────────────────────────────────────────
# Hybrid mode
# ─────────────────────────────────────────────────────────────


class TestHybridMode:
    """Tests for mixing fast API with expression-based constraints."""

    def test_hybrid_linear_and_nonlinear(self):
        """1000 linear (fast) + 3 nonlinear (expressions), all in ModelRepr."""
        m = dm.Model("hybrid")
        x = m.continuous("x", shape=(100,), lb=0, ub=10)

        # 1000 linear constraints via fast API
        A = sp.random(1000, 100, density=0.01, format="csr", random_state=42)
        b = np.ones(1000)
        m.add_linear_constraints(A, x, "<=", b, name="linear")

        # 3 nonlinear constraints via expressions
        m.subject_to(dm.exp(x[0]) + x[1] <= 10.0, name="nl1")
        m.subject_to(x[2] * x[3] >= 1.0, name="nl2")
        m.subject_to(dm.sin(x[4]) == 0.5, name="nl3")

        m.add_linear_objective(np.ones(100), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        # 1000 from builder + 3 from expressions
        assert repr_.n_constraints == 1003

    def test_variable_added_after_builder(self):
        """Variable created after first fast-API call works."""
        m = dm.Model("late_var")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)

        # Trigger builder initialization
        A1 = sp.eye(3, 5, format="csr")
        m.add_linear_constraints(A1, x, "<=", np.ones(3))

        # Add a new variable AFTER builder init
        y = m.continuous("y", shape=(3,), lb=0, ub=5)

        # Use new variable with expressions
        m.subject_to(y[0] + y[1] <= 8.0)
        m.add_linear_objective(np.ones(5), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == 4  # 3 linear + 1 expression
        assert repr_.n_vars == 8  # 5 + 3


# ─────────────────────────────────────────────────────────────
# Constraint evaluation correctness
# ─────────────────────────────────────────────────────────────


class TestEvaluation:
    """Test that fast-API constraints evaluate correctly."""

    def test_constraint_evaluation(self):
        """Verify constraint body evaluates to A @ x."""
        m = dm.Model("eval")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        A = sp.csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = np.array([10.0, 20.0])
        m.add_linear_constraints(A, x, "<=", b)
        m.add_linear_objective(np.ones(3), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        x_test = np.array([1.0, 2.0, 3.0])
        # Row 0: 1*1 + 2*2 + 3*3 = 14
        assert repr_.evaluate_constraint(0, x_test) == pytest.approx(14.0)
        # Row 1: 4*1 + 5*2 + 6*3 = 32
        assert repr_.evaluate_constraint(1, x_test) == pytest.approx(32.0)

    def test_identity_matrix_constraints(self):
        """Identity matrix: each constraint is just one variable."""
        m = dm.Model("identity")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        m.add_linear_constraints(sp.eye(5, format="csr"), x, "<=", np.full(5, 5.0))
        m.add_linear_objective(np.ones(5), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        x_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for i in range(5):
            assert repr_.evaluate_constraint(i, x_test) == pytest.approx(x_test[i])


# ─────────────────────────────────────────────────────────────
# End-to-end solve
# ─────────────────────────────────────────────────────────────


class TestEndToEndSolve:
    """Test full solve pipeline with fast-API models."""

    def test_end_to_end_solve_lp(self):
        """Build LP via fast API, solve, verify optimal.

        min  -x0 - 2*x1
        s.t. x0 + x1 <= 4
             x0 <= 3
             x1 <= 3
             x >= 0
        Optimal: x0=1, x1=3, obj=-7
        """
        m = dm.Model("lp_solve")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)

        A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
        b = np.array([4.0, 3.0, 3.0])
        m.add_linear_constraints(A, x, "<=", b)
        m.add_linear_objective(np.array([-1.0, -2.0]), x)

        result = m.solve()
        assert result.status in ("optimal", "feasible")
        x_val = result.value(x)
        assert x_val[0] == pytest.approx(1.0, abs=1e-3)
        assert x_val[1] == pytest.approx(3.0, abs=1e-3)
        assert result.objective == pytest.approx(-7.0, abs=1e-3)

    def test_end_to_end_solve_qp(self):
        """Build QP via fast API, solve, verify optimal.

        min  0.5 * (x0^2 + x1^2) - x0 - x1
        s.t. x0 + x1 <= 1.5
             x >= 0
        Optimal: x0=0.75, x1=0.75
        """
        m = dm.Model("qp_solve")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)

        A = sp.csr_matrix(np.array([[1.0, 1.0]]))
        b = np.array([1.5])
        m.add_linear_constraints(A, x, "<=", b)

        Q = sp.eye(2, format="csr")
        c = np.array([-1.0, -1.0])
        m.add_quadratic_objective(Q, c, x)

        result = m.solve()
        assert result.status in ("optimal", "feasible")
        x_val = result.value(x)
        assert x_val[0] == pytest.approx(0.75, abs=1e-2)
        assert x_val[1] == pytest.approx(0.75, abs=1e-2)


# ─────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────


class TestPerformance:
    """Performance comparison tests."""

    def test_performance_vs_expression(self):
        """10k constraints via fast API should be fast (< 2s including import)."""
        m = dm.Model("perf")
        x = m.continuous("x", shape=(1000,), lb=0, ub=100)

        A = sp.random(10000, 1000, density=0.01, format="csr", random_state=42)
        b = np.ones(10000)

        t0 = time.perf_counter()
        m.add_linear_constraints(A, x, "<=", b)
        m.add_linear_objective(np.ones(1000), x)
        t1 = time.perf_counter()

        # Should be well under 2 seconds (usually milliseconds)
        assert t1 - t0 < 2.0, f"Fast API took {t1 - t0:.2f}s for 10k constraints"

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == 10000


# ─────────────────────────────────────────────────────────────
# Structure detection on fast-API models
# ─────────────────────────────────────────────────────────────


class TestStructureDetection:
    """Test that structure detection works on fast-API constraints."""

    def test_constraints_are_linear(self):
        """All fast-API constraints are detected as linear."""
        m = dm.Model("struct")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        A = sp.random(10, 5, density=0.5, format="csr", random_state=42)
        b = np.ones(10)
        m.add_linear_constraints(A, x, "<=", b)
        m.add_linear_objective(np.ones(5), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        for i in range(10):
            assert repr_.is_constraint_linear(i)

    def test_linear_objective_is_linear(self):
        """Linear objective detected as linear."""
        m = dm.Model("lin_det")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        m.add_linear_objective(np.ones(5), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.is_objective_linear()

    def test_quadratic_objective_is_quadratic(self):
        """Quadratic objective detected as quadratic."""
        m = dm.Model("quad_det")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        Q = sp.eye(3, format="csr")
        c = np.zeros(3)
        m.add_quadratic_objective(Q, c, x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.is_objective_quadratic()


# ─────────────────────────────────────────────────────────────
# Constraint naming
# ─────────────────────────────────────────────────────────────


class TestConstraintNaming:
    """Test constraint names from fast API."""

    def test_named_constraints(self):
        """Named constraints get prefix_i naming."""
        m = dm.Model("named")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        A = sp.eye(3, format="csr")
        m.add_linear_constraints(A, x, "<=", np.ones(3), name="bound")
        m.add_linear_objective(np.ones(3), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.constraint_name(0) == "bound_0"
        assert repr_.constraint_name(1) == "bound_1"
        assert repr_.constraint_name(2) == "bound_2"

    def test_unnamed_constraints(self):
        """Unnamed constraints have None names."""
        m = dm.Model("unnamed")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        A = sp.eye(3, format="csr")
        m.add_linear_constraints(A, x, "<=", np.ones(3))
        m.add_linear_objective(np.ones(3), x)

        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        for i in range(3):
            assert repr_.constraint_name(i) is None
