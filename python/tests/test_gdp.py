"""Tests for GDP (Generalized Disjunctive Programming) big-M reformulation.

Tests cover indicator constraints, disjunctive constraints, SOS1/SOS2
constraints, big-M computation, and end-to-end solve correctness.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.gdp_reformulate import (
    _bound_expression,
    _compute_big_m,
    _reformulate_indicator,
    reformulate_gdp,
)
from discopt.modeling.core import (
    Constraint,
    VarType,
    _IndicatorConstraint,
)

# ── Model with no GDP constraints passes through unchanged ──


class TestNoGDP:
    def test_no_gdp_returns_same_model(self):
        m = dm.Model("plain")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 1)

        result = reformulate_gdp(m)
        assert result is m  # identity check -- no copy

    def test_empty_model(self):
        m = dm.Model("empty")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x)
        result = reformulate_gdp(m)
        assert result is m


# ── Interval arithmetic / big-M computation ──


class TestBoundExpression:
    def test_variable_bounds(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-3, ub=7)
        lo, hi = _bound_expression(x, m)
        assert lo == -3.0
        assert hi == 7.0

    def test_constant(self):
        from discopt.modeling.core import Constant

        m = dm.Model("t")
        c = Constant(5.0)
        lo, hi = _bound_expression(c, m)
        assert lo == 5.0
        assert hi == 5.0

    def test_addition(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=1, ub=4)
        y = m.continuous("y", lb=2, ub=6)
        expr = x + y
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(10.0)

    def test_subtraction(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=1, ub=4)
        y = m.continuous("y", lb=2, ub=6)
        expr = x - y
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(-5.0)
        assert hi == pytest.approx(2.0)

    def test_multiplication(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-2, ub=3)
        y = m.continuous("y", lb=1, ub=4)
        expr = x * y
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(-8.0)
        assert hi == pytest.approx(12.0)

    def test_power_squared(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-3, ub=5)
        expr = x**2
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(25.0)

    def test_indexed_variable(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(3,), lb=np.array([1, 2, 3]), ub=np.array([4, 5, 6]))
        expr = x[1]
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(5.0)


class TestComputeBigM:
    def test_le_constraint(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        # body = x - 5 (from x <= 5 => x - 5 <= 0)
        con = Constraint(body=x - dm.core.Constant(5.0), sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Upper bound of body = 10 - 5 = 5, so M ~= 5 * 1.01
        assert M == pytest.approx(5.0 * 1.01)

    def test_ge_constraint(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        # body = x - 3 (from x >= 3 => x - 3 >= 0)
        con = Constraint(body=x - dm.core.Constant(3.0), sense=">=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Lower bound of body = 0 - 3 = -3, so M = -(-3) * 1.01 = 3.03
        assert M == pytest.approx(3.0 * 1.01)

    def test_infinite_bounds_use_default(self):
        m = dm.Model("t")
        x = m.continuous("x")  # lb=-1e20, ub=1e20
        con = Constraint(body=x, sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Should fallback to default 1e4
        assert M == pytest.approx(1e4 * 1.01)


# ── Indicator constraint reformulation ──


class TestIndicatorReformulation:
    def test_simple_le_indicator(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(5.0), sense="<=", rhs=0.0)
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=1)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 1
        assert new_cons[0].sense == "<="

    def test_simple_ge_indicator(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(3.0), sense=">=", rhs=0.0)
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=1)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 1
        assert new_cons[0].sense == ">="

    def test_eq_indicator_produces_two_constraints(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(5.0), sense="==", rhs=0.0, name="eq")
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=1)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 2
        assert new_cons[0].sense == "<="
        assert new_cons[1].sense == ">="

    def test_active_value_zero(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(5.0), sense="<=", rhs=0.0)
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=0)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 1
        # When y=0 constraint should be active (relaxed when y=1)


class TestReformulateGDP:
    def test_single_indicator(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        assert new_m is not m
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        assert new_m._objective is m._objective

    def test_multiple_indicators(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y", shape=(2,))
        m.minimize(x)
        m.if_then(y[0], [x <= 5])
        m.if_then(y[1], [x >= 2])

        new_m = reformulate_gdp(m)
        # Original has 2 _IndicatorConstraint objects
        # Each produces 1 standard Constraint
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        assert len(new_m._constraints) == 2

    def test_mixed_regular_and_gdp(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.subject_to(x >= 0)  # regular
        m.if_then(y, [x <= 5])  # GDP

        new_m = reformulate_gdp(m)
        # 1 regular + 1 from indicator = 2 total
        assert len(new_m._constraints) == 2
        assert all(isinstance(c, Constraint) for c in new_m._constraints)

    def test_variables_preserved(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        # Original vars should be in new model
        assert x in new_m._variables
        assert y in new_m._variables


# ── Disjunctive constraint reformulation ──


class TestDisjunctionReformulation:
    def test_two_disjuncts(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 15]],
            name="modes",
        )

        new_m = reformulate_gdp(m)
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        # Should have: 1 selector sum ==, 2 big-M constraints (one per disjunct)
        assert len(new_m._constraints) == 3
        # Two auxiliary binaries added
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 2

    def test_three_disjuncts(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=30)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 10, x <= 15], [x >= 25]],
            name="three_modes",
        )

        new_m = reformulate_gdp(m)
        # 1 selector sum + (1 + 2 + 1) = 5 total constraints
        assert len(new_m._constraints) == 5
        # 3 auxiliary binaries
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 3

    def test_disjunction_with_eq_constraint(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.either_or(
            [[x <= 3], [Constraint(body=x - dm.core.Constant(7.0), sense="==", rhs=0.0)]],
            name="eq_disj",
        )

        new_m = reformulate_gdp(m)
        # 1 selector + 1 LE + 2 (LE + GE for ==) = 4
        assert len(new_m._constraints) == 4


# ── SOS constraint reformulation ──


class TestSOSReformulation:
    def test_sos1_basic(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0] + x[1] + x[2])
        m.sos1([x[0], x[1], x[2]], name="set1")

        new_m = reformulate_gdp(m)
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        # 3 vars: 3 upper-bound linking + 1 sum constraint = 4
        # (No lower-bound linking since lb=0)
        assert len(new_m._constraints) == 4
        # 3 auxiliary binaries
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 3

    def test_sos1_with_negative_bounds(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)
        m.minimize(x[0] + x[1])
        m.sos1([x[0], x[1]], name="neg")

        new_m = reformulate_gdp(m)
        # 2 upper-bound + 2 lower-bound + 1 sum = 5
        assert len(new_m._constraints) == 5

    def test_sos2_basic(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(4,), lb=0, ub=10)
        m.minimize(x[0] + x[1] + x[2] + x[3])
        m.sos2([x[0], x[1], x[2], x[3]], name="adj")

        new_m = reformulate_gdp(m)
        # 4 upper-bound + 1 sum<=2 + non-adjacency pairs
        # Non-adjacent: (0,2), (0,3), (1,3) = 3 pairs
        # Total: 4 + 1 + 3 = 8
        assert len(new_m._constraints) == 8
        # 4 auxiliary binaries
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 4


# ── End-to-end solve tests ──


class TestEndToEndSolve:
    def test_indicator_solve_active(self):
        """If y=1 then x <= 5; minimize x+y => should set y=0, x at lower bound."""
        m = dm.Model("ind_active")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 10 * y)
        m.if_then(y, [x <= 5])
        m.subject_to(x >= 3)

        result = m.solve()
        assert result.status == "optimal"
        # y=0 is cheaper (saves 10), x=3 satisfies x>=3
        assert result.x["y"] == pytest.approx(0.0, abs=1e-3)
        assert result.x["x"] == pytest.approx(3.0, abs=1e-3)

    def test_indicator_solve_forced(self):
        """Force y=1 and check indicator constraint is respected."""
        m = dm.Model("ind_forced")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])
        m.subject_to(y >= 1)  # force y=1

        result = m.solve()
        assert result.status == "optimal"
        # y must be 1, so x <= 5 is active; minimize x => x = 0
        assert result.x["y"] == pytest.approx(1.0, abs=1e-3)
        assert result.x["x"] == pytest.approx(0.0, abs=1e-3)

    def test_disjunction_solve(self):
        """Either x <= 5 or x >= 15; minimize x => x should be at lower bound."""
        m = dm.Model("disj_solve")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 15]],
            name="modes",
        )

        result = m.solve()
        assert result.status == "optimal"
        # First disjunct allows x in [0, 5], which is cheaper => x = 0
        assert result.objective == pytest.approx(0.0, abs=1e-2)

    def test_disjunction_forced_second(self):
        """Either x <= 5 or x >= 15; maximize -x with x >= 16 => forces second."""
        m = dm.Model("disj_forced")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 15]],
            name="modes",
        )
        m.subject_to(x >= 16)

        result = m.solve()
        assert result.status == "optimal"
        # Must use second disjunct: x >= 16 forces x = 16
        assert result.x["x"] == pytest.approx(16.0, abs=1e-2)

    def test_sos1_solve(self):
        """SOS1: at most one of x1, x2, x3 nonzero. Minimize -x1 - 2*x2 - x3."""
        m = dm.Model("sos1_solve")
        x1 = m.continuous("x1", lb=0, ub=10)
        x2 = m.continuous("x2", lb=0, ub=10)
        x3 = m.continuous("x3", lb=0, ub=10)
        m.minimize(-x1 - 2 * x2 - x3)
        m.sos1([x1, x2, x3], name="s1")

        result = m.solve()
        assert result.status == "optimal"
        # Best: x2 = 10, rest = 0 => obj = -20
        assert result.objective == pytest.approx(-20.0, abs=0.5)
        assert result.x["x2"] == pytest.approx(10.0, abs=0.5)

    def test_no_gdp_solve_unchanged(self):
        """A plain MINLP should solve correctly without GDP overhead."""
        m = dm.Model("plain")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 5 * y)
        m.subject_to(x >= 1)

        result = m.solve()
        assert result.status == "optimal"
        assert result.x["x"] == pytest.approx(1.0, abs=1e-3)
        assert result.x["y"] == pytest.approx(0.0, abs=1e-3)


# ── Additional reformulation correctness tests ──


class TestReformulationCorrectness:
    def test_indicator_constraint_name_propagation(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5], name="test_ind")

        new_m = reformulate_gdp(m)
        # Named constraints should preserve names
        named = [c for c in new_m._constraints if c.name is not None]
        assert len(named) >= 1

    def test_disjunction_selector_sum(self):
        """Verify the selector sum == 1 constraint is present."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or([[x <= 5], [x >= 15]], name="sel")

        new_m = reformulate_gdp(m)
        eq_cons = [c for c in new_m._constraints if c.sense == "=="]
        assert len(eq_cons) == 1  # selector sum == 1

    def test_multiple_constraints_in_if_then(self):
        """if_then with multiple then-constraints."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5, x >= 2])

        new_m = reformulate_gdp(m)
        # 2 indicator constraints => 2 standard constraints
        assert len(new_m._constraints) == 2

    def test_parameters_preserved(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        p = m.parameter("p", value=5.0)
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        assert len(new_m._parameters) == 1
        assert new_m._parameters[0] is p

    def test_model_name_preserved(self):
        m = dm.Model("my_model")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        assert new_m.name == "my_model"

    def test_auxiliary_variables_are_binary(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or([[x <= 5], [x >= 15]], name="modes")

        new_m = reformulate_gdp(m)
        orig_vars = set(id(v) for v in m._variables)
        aux_vars = [v for v in new_m._variables if id(v) not in orig_vars]
        assert len(aux_vars) == 2
        for v in aux_vars:
            assert v.var_type == VarType.BINARY
            assert float(v.lb) == 0.0
            assert float(v.ub) == 1.0

    def test_big_m_finite(self):
        """All big-M values should be finite."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=100)
        con = Constraint(body=x - dm.core.Constant(50.0), sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        assert np.isfinite(M)
        assert M > 0

    def test_big_m_tightness(self):
        """Big-M from bounds should be tighter than default when bounds are finite."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=5)
        con = Constraint(body=x - dm.core.Constant(3.0), sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Upper bound of (x - 3) is 5 - 3 = 2, so M should be ~2.02
        assert M < 100  # much tighter than default 1e4

    def test_sos2_non_adjacency_count(self):
        """SOS2 with 5 vars should have correct number of non-adjacency constraints."""
        m = dm.Model("t")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        m.minimize(x[0])
        m.sos2([x[0], x[1], x[2], x[3], x[4]], name="s2")

        new_m = reformulate_gdp(m)
        # Non-adjacent pairs for n=5: (0,2), (0,3), (0,4), (1,3), (1,4), (2,4) = 6
        # Plus 5 upper-bound + 1 sum constraint = 12 total
        le_cons = [c for c in new_m._constraints if c.sense == "<="]
        # 5 ub + 1 sum + 6 nonadj = 12 LE constraints
        assert len(le_cons) == 12
