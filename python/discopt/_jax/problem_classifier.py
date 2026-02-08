"""
Problem classification and standard-form extraction for LP, QP, MILP, MIQP, NLP, MINLP.

Uses existing Rust structure detection (is_linear, is_quadratic) via PyO3 bindings
to classify problems, then extracts standard-form data using the JAX DAG compiler.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from discopt.modeling.core import (
    Constraint,
    Model,
    VarType,
)


class ProblemClass(Enum):
    """Classification of an optimization problem."""

    LP = "lp"  # linear obj + linear constraints + all continuous
    QP = "qp"  # ≤quadratic obj + linear constraints + all continuous
    MILP = "milp"  # linear obj + linear constraints + has integer/binary
    MIQP = "miqp"  # ≤quadratic obj + linear constraints + has integer/binary
    NLP = "nlp"  # general nonlinear + all continuous
    MINLP = "minlp"  # general nonlinear + has integer/binary


def classify_problem(model: Model) -> ProblemClass:
    """Classify a model into LP, QP, MILP, MIQP, NLP, or MINLP.

    Uses Rust structure detection for degree analysis of the objective
    and constraints. Falls back to NLP/MINLP if Rust bindings unavailable.

    Args:
        model: A discopt Model with objective and constraints.

    Returns:
        ProblemClass enum value.
    """
    has_integer = any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)

    try:
        from discopt._rust import model_to_repr

        repr = model_to_repr(model)
        obj_linear = repr.is_objective_linear()
        obj_quadratic = repr.is_objective_quadratic()
        all_constraints_linear = all(
            repr.is_constraint_linear(i) for i in range(repr.n_constraints)
        )
    except (ImportError, Exception):
        # Rust bindings unavailable — fall back to NLP/MINLP
        return ProblemClass.MINLP if has_integer else ProblemClass.NLP

    if all_constraints_linear:
        if obj_linear:
            return ProblemClass.MILP if has_integer else ProblemClass.LP
        if obj_quadratic:
            return ProblemClass.MIQP if has_integer else ProblemClass.QP

    return ProblemClass.MINLP if has_integer else ProblemClass.NLP


class LPData(NamedTuple):
    """Standard-form LP data: min c'x + d s.t. A_eq x = b_eq, x_l <= x <= x_u."""

    c: jnp.ndarray  # (n,) objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


class QPData(NamedTuple):
    """Standard-form QP: min 0.5 x'Qx + c'x + d s.t. A_eq x = b_eq, bounds."""

    Q: jnp.ndarray  # (n, n) quadratic objective matrix (symmetric)
    c: jnp.ndarray  # (n,) linear objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


def _get_variable_bounds(model: Model):
    """Extract flat lower and upper bounds from model variables."""
    lb_parts = []
    ub_parts = []
    for v in model._variables:
        lb_parts.append(v.lb.flatten())
        ub_parts.append(v.ub.flatten())
    n = sum(v.size for v in model._variables)
    if n == 0:
        return jnp.zeros(0), jnp.zeros(0)
    lb = jnp.array(np.concatenate(lb_parts), dtype=jnp.float64)
    ub = jnp.array(np.concatenate(ub_parts), dtype=jnp.float64)
    return lb, ub


def extract_lp_data(model: Model) -> LPData:
    """Extract LP standard form from a model classified as LP.

    Compiles the objective and constraints using the JAX DAG compiler,
    then extracts linear coefficients via autodiff (jax.grad for c,
    jax.jacobian for A).

    Inequality constraints are converted to equalities with slacks:
      - body <= 0 becomes body + s = 0, s >= 0
      - body >= 0 becomes body - s = 0, s >= 0

    Args:
        model: A Model classified as ProblemClass.LP.

    Returns:
        LPData with c, A_eq, b_eq, x_l, x_u.
    """
    from discopt._jax.dag_compiler import compile_constraint, compile_objective

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    # Extract c and constant: obj(x) = c'x + d, so grad(obj)(0) = c, obj(0) = d
    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)
    c = jax.grad(obj_fn)(x_zero)
    obj_const = float(obj_fn(x_zero))

    # Extract constraint coefficients
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    # Separate equality and inequality constraints
    eq_fns = []
    eq_rhs = []
    ineq_fns = []  # body <= 0 form
    ineq_senses = []  # "le" or "ge"

    for con in constraints:
        con_fn = compile_constraint(con, model)
        if con.sense == "==":
            eq_fns.append(con_fn)
            # body - rhs == 0 is compiled, so rhs = 0 in compiled form
            eq_rhs.append(0.0)
        elif con.sense == "<=":
            ineq_fns.append(con_fn)
            ineq_senses.append("le")
        elif con.sense == ">=":
            ineq_fns.append(con_fn)
            ineq_senses.append("ge")

    n_eq = len(eq_fns)
    n_ineq = len(ineq_fns)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    # Build equality constraint matrix for original equality constraints
    A_rows = []
    b_vals = []

    for i, fn in enumerate(eq_fns):
        # For linear constraint, A_row = grad(con)(0)
        a_row = jax.grad(fn)(x_zero)
        # Constant part: fn(0) = a'*0 + const = const, so b = -const
        const = fn(x_zero)
        A_rows.append(jnp.concatenate([a_row, jnp.zeros(n_slack)]))
        b_vals.append(-float(const))

    # Convert inequality constraints to equalities with slacks
    for i, fn in enumerate(ineq_fns):
        a_row = jax.grad(fn)(x_zero)
        const = fn(x_zero)
        slack_col = jnp.zeros(n_slack)
        if ineq_senses[i] == "le":
            # body <= 0 → body + s = 0, s >= 0
            slack_col = slack_col.at[i].set(1.0)
        else:
            # body >= 0 → body - s = 0, s >= 0
            slack_col = slack_col.at[i].set(-1.0)
        A_rows.append(jnp.concatenate([a_row, slack_col]))
        b_vals.append(-float(const))

    m_total = n_eq + n_ineq
    if m_total > 0:
        A_eq = jnp.stack(A_rows)
        b_eq = jnp.array(b_vals, dtype=jnp.float64)
    else:
        A_eq = jnp.zeros((0, n_total), dtype=jnp.float64)
        b_eq = jnp.zeros(0, dtype=jnp.float64)

    # Bounds: original vars keep their bounds, slack vars >= 0
    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = jnp.concatenate([c, jnp.zeros(n_slack)])
    x_l = jnp.concatenate([x_l_orig, jnp.zeros(n_slack)])
    x_u = jnp.concatenate([x_u_orig, jnp.full(n_slack, 1e20)])

    return LPData(
        c=c_full,
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data(model: Model) -> QPData:
    """Extract QP standard form from a model classified as QP.

    Uses jax.hessian to extract Q from the objective, and jax.grad/jacobian
    for constraint coefficients. The objective is:
      f(x) = 0.5 x'Qx + c'x + const

    So Q = hessian(f)(0) and c = grad(f)(0) (since hessian is constant for QP).

    Args:
        model: A Model classified as ProblemClass.QP.

    Returns:
        QPData with Q, c, A_eq, b_eq, x_l, x_u.
    """
    from discopt._jax.dag_compiler import compile_objective

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)

    # Q = hessian(obj) — constant for QP
    Q = jax.hessian(obj_fn)(x_zero)

    # c = grad(obj)(0) = Q*0 + c = c (linear part)
    c_vec = jax.grad(obj_fn)(x_zero)

    # Constant term: f(0) = 0.5*0'Q*0 + c'*0 + d = d
    obj_const = float(obj_fn(x_zero))

    # Extract LP data for constraints (they're all linear)
    lp_data = extract_lp_data(model)
    n_slack = lp_data.c.shape[0] - n_orig

    # Extend Q with zeros for slack variables
    if n_slack > 0:
        n_total = n_orig + n_slack
        Q_full = jnp.zeros((n_total, n_total), dtype=jnp.float64)
        Q_full = Q_full.at[:n_orig, :n_orig].set(Q)
        c_full = jnp.concatenate([c_vec, jnp.zeros(n_slack)])
    else:
        Q_full = Q
        c_full = c_vec

    return QPData(
        Q=Q_full,
        c=c_full,
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        x_l=lp_data.x_l,
        x_u=lp_data.x_u,
        obj_const=obj_const,
    )
