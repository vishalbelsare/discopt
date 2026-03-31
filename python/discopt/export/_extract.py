"""
Expression linearization and quadratic extraction helpers.

These walk the expression DAG to extract linear and quadratic coefficients,
raising ``ValueError`` for any nonlinear terms that cannot be represented
in MPS or LP format.
"""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    IndexExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)


def flatten_variables(
    model: Model,
) -> list[tuple[str, VarType, tuple[int, ...], float, float]]:
    """Flatten all model variables into a list of scalar entries.

    Each entry is ``(name, var_type, original_shape, lb, ub)`` where
    *name* is a unique scalar name like ``x`` for scalars or ``x_0``,
    ``x_1`` for arrays.

    Returns
    -------
    list of tuple
        One entry per scalar variable component.
    """
    flat: list[tuple[str, VarType, tuple[int, ...], float, float]] = []
    for var in model._variables:
        if var.shape == () or var.shape == (1,):
            flat.append(
                (var.name, var.var_type, var.shape, float(var.lb.flat[0]), float(var.ub.flat[0]))
            )
        else:
            for k in range(var.size):
                idx = np.unravel_index(k, var.shape)
                suffix = "_".join(str(i) for i in idx)
                name = f"{var.name}_{suffix}"
                flat.append(
                    (name, var.var_type, var.shape, float(var.lb.flat[k]), float(var.ub.flat[k]))
                )
    return flat


def _var_index(
    expr: Expression,
    flat_vars: list[tuple[str, VarType, tuple[int, ...], float, float]],
    model_vars: list[Variable],
) -> int:
    """Resolve an expression to a flat variable index.

    Handles both ``Variable`` (scalar) and ``IndexExpression`` (array element).

    Raises
    ------
    ValueError
        If the expression is not a simple variable reference.
    """
    if isinstance(expr, Variable):
        # Find offset of this variable in flat list
        offset = 0
        for v in model_vars:
            if v is expr:
                if expr.shape == () or expr.shape == (1,):
                    return offset
                raise ValueError(
                    f"Array variable '{expr.name}' used without indexing. "
                    "Only scalar variables or indexed elements can appear "
                    "in linear/quadratic expressions for export."
                )
            offset += max(v.size, 1) if v.shape != () else 1
        raise ValueError(f"Variable '{expr.name}' not found in model")

    if isinstance(expr, IndexExpression):
        base = expr.base
        if not isinstance(base, Variable):
            raise ValueError("Nonlinear expression: nested indexing is not supported for export.")
        offset = 0
        for v in model_vars:
            if v is base:
                idx = expr.index
                if isinstance(idx, (int, np.integer)):
                    return offset + int(idx)
                if isinstance(idx, tuple):
                    flat_idx = int(np.ravel_multi_index(idx, base.shape))
                    return offset + flat_idx
                raise ValueError(f"Unsupported index type {type(idx)} for variable '{base.name}'")
            offset += max(v.size, 1) if v.shape != () else 1
        raise ValueError(f"Variable '{base.name}' not found in model")

    raise ValueError("Expression is not a variable reference.")


def extract_linear_terms(
    expr: Expression,
    flat_vars: list[tuple[str, VarType, tuple[int, ...], float, float]],
    model_vars: list[Variable] | None = None,
) -> tuple[dict[int, float], float]:
    """Extract linear coefficients from an expression.

    Parameters
    ----------
    expr : Expression
        The expression to decompose.
    flat_vars : list
        Flattened variable list from :func:`flatten_variables`.
    model_vars : list of Variable, optional
        Model variable list. If ``None``, reconstructed from flat_vars names.

    Returns
    -------
    coefficients : dict[int, float]
        Maps flat variable index to coefficient.
    constant : float
        Constant term.

    Raises
    ------
    ValueError
        If the expression contains nonlinear terms.
    """
    if model_vars is None:
        # Must be provided; fall back to extracting from a model
        raise ValueError("model_vars must be provided")

    coeffs: dict[int, float] = {}
    const = _extract_linear_recursive(expr, coeffs, 1.0, flat_vars, model_vars)
    return coeffs, const


def _extract_linear_recursive(
    expr: Expression,
    coeffs: dict[int, float],
    multiplier: float,
    flat_vars: list,
    model_vars: list[Variable],
) -> float:
    """Recursively extract linear terms. Returns the constant contribution."""
    if isinstance(expr, Constant):
        val = float(expr.value) if expr.value.ndim == 0 else float(expr.value.flat[0])
        return multiplier * val

    if isinstance(expr, Variable) or isinstance(expr, IndexExpression):
        idx = _var_index(expr, flat_vars, model_vars)
        coeffs[idx] = coeffs.get(idx, 0.0) + multiplier
        return 0.0

    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return _extract_linear_recursive(
                expr.operand, coeffs, -multiplier, flat_vars, model_vars
            )
        raise ValueError(
            f"Nonlinear expression: unary operation '{expr.op}' "
            "cannot be exported to MPS/LP format."
        )

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            c1 = _extract_linear_recursive(expr.left, coeffs, multiplier, flat_vars, model_vars)
            c2 = _extract_linear_recursive(expr.right, coeffs, multiplier, flat_vars, model_vars)
            return c1 + c2

        if expr.op == "-":
            c1 = _extract_linear_recursive(expr.left, coeffs, multiplier, flat_vars, model_vars)
            c2 = _extract_linear_recursive(expr.right, coeffs, -multiplier, flat_vars, model_vars)
            return c1 + c2

        if expr.op == "*":
            # One side must be constant for linearity
            if _is_constant(expr.left):
                val = _get_constant_value(expr.left)
                return _extract_linear_recursive(
                    expr.right, coeffs, multiplier * val, flat_vars, model_vars
                )
            if _is_constant(expr.right):
                val = _get_constant_value(expr.right)
                return _extract_linear_recursive(
                    expr.left, coeffs, multiplier * val, flat_vars, model_vars
                )
            # Both sides have variables: nonlinear for linear extraction
            raise ValueError(
                "Nonlinear expression: product of two variable expressions "
                "cannot be exported as linear. Use extract_quadratic_terms "
                "for quadratic support."
            )

        if expr.op == "/":
            if _is_constant(expr.right):
                val = _get_constant_value(expr.right)
                if val == 0:
                    raise ValueError("Division by zero in expression.")
                return _extract_linear_recursive(
                    expr.left, coeffs, multiplier / val, flat_vars, model_vars
                )
            raise ValueError(
                "Nonlinear expression: division by a variable expression "
                "cannot be exported to MPS/LP format."
            )

        if expr.op == "**":
            raise ValueError(
                "Nonlinear expression: exponentiation cannot be exported as a linear expression."
            )

        raise ValueError(f"Unknown binary operation '{expr.op}'.")

    if isinstance(expr, SumOverExpression):
        total_const = 0.0
        for term in expr.terms:
            total_const += _extract_linear_recursive(
                term, coeffs, multiplier, flat_vars, model_vars
            )
        return total_const

    if isinstance(expr, SumExpression):
        return _extract_linear_recursive(expr.operand, coeffs, multiplier, flat_vars, model_vars)

    raise ValueError(
        f"Expression type '{type(expr).__name__}' cannot be exported "
        "to MPS/LP format. Only linear and quadratic expressions are supported."
    )


def extract_quadratic_terms(
    expr: Expression,
    flat_vars: list[tuple[str, VarType, tuple[int, ...], float, float]],
    model_vars: list[Variable] | None = None,
) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
    """Extract quadratic and linear coefficients from an expression.

    Parameters
    ----------
    expr : Expression
        The expression to decompose.
    flat_vars : list
        Flattened variable list from :func:`flatten_variables`.
    model_vars : list of Variable, optional
        If ``None``, extracted from the first variable found.

    Returns
    -------
    quad_terms : dict[tuple[int, int], float]
        Maps ``(var_i, var_j)`` with ``i <= j`` to coefficient.
    linear_terms : dict[int, float]
        Maps flat variable index to linear coefficient.
    constant : float
        Constant term.

    Raises
    ------
    ValueError
        If the expression contains terms beyond quadratic.
    """
    if model_vars is None:
        raise ValueError("model_vars must be provided")

    quad: dict[tuple[int, int], float] = {}
    linear: dict[int, float] = {}
    const = _extract_quad_recursive(expr, quad, linear, 1.0, flat_vars, model_vars)
    return quad, linear, const


def _extract_quad_recursive(
    expr: Expression,
    quad: dict[tuple[int, int], float],
    linear: dict[int, float],
    multiplier: float,
    flat_vars: list,
    model_vars: list[Variable],
) -> float:
    """Recursively extract quadratic and linear terms."""
    if isinstance(expr, Constant):
        val = float(expr.value) if expr.value.ndim == 0 else float(expr.value.flat[0])
        return multiplier * val

    if isinstance(expr, (Variable, IndexExpression)):
        idx = _var_index(expr, flat_vars, model_vars)
        linear[idx] = linear.get(idx, 0.0) + multiplier
        return 0.0

    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return _extract_quad_recursive(
                expr.operand, quad, linear, -multiplier, flat_vars, model_vars
            )
        raise ValueError(f"Nonlinear expression: unary '{expr.op}' cannot be exported.")

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            c1 = _extract_quad_recursive(expr.left, quad, linear, multiplier, flat_vars, model_vars)
            c2 = _extract_quad_recursive(
                expr.right, quad, linear, multiplier, flat_vars, model_vars
            )
            return c1 + c2

        if expr.op == "-":
            c1 = _extract_quad_recursive(expr.left, quad, linear, multiplier, flat_vars, model_vars)
            c2 = _extract_quad_recursive(
                expr.right, quad, linear, -multiplier, flat_vars, model_vars
            )
            return c1 + c2

        if expr.op == "*":
            if _is_constant(expr.left):
                val = _get_constant_value(expr.left)
                return _extract_quad_recursive(
                    expr.right,
                    quad,
                    linear,
                    multiplier * val,
                    flat_vars,
                    model_vars,
                )
            if _is_constant(expr.right):
                val = _get_constant_value(expr.right)
                return _extract_quad_recursive(
                    expr.left,
                    quad,
                    linear,
                    multiplier * val,
                    flat_vars,
                    model_vars,
                )

            # Product of two variable expressions: extract linear from each
            # and combine. Both must be purely linear (no quad * quad).
            left_lin: dict[int, float] = {}
            left_const = _extract_linear_from_quad(expr.left, left_lin, 1.0, flat_vars, model_vars)
            right_lin: dict[int, float] = {}
            right_const = _extract_linear_from_quad(
                expr.right, right_lin, 1.0, flat_vars, model_vars
            )

            # var_i * var_j terms
            for i, ci in left_lin.items():
                for j, cj in right_lin.items():
                    key = (min(i, j), max(i, j))
                    quad[key] = quad.get(key, 0.0) + multiplier * ci * cj

            # var * const terms
            for i, ci in left_lin.items():
                if right_const != 0.0:
                    linear[i] = linear.get(i, 0.0) + multiplier * ci * right_const
            for j, cj in right_lin.items():
                if left_const != 0.0:
                    linear[j] = linear.get(j, 0.0) + multiplier * cj * left_const

            # const * const
            return multiplier * left_const * right_const

        if expr.op == "/":
            if _is_constant(expr.right):
                val = _get_constant_value(expr.right)
                if val == 0:
                    raise ValueError("Division by zero in expression.")
                return _extract_quad_recursive(
                    expr.left,
                    quad,
                    linear,
                    multiplier / val,
                    flat_vars,
                    model_vars,
                )
            raise ValueError(
                "Nonlinear expression: division by variable expression cannot be exported."
            )

        if expr.op == "**":
            # x**2 is quadratic
            if _is_constant(expr.right):
                power = _get_constant_value(expr.right)
                if power == 2.0:
                    # base must be linear
                    base_lin: dict[int, float] = {}
                    base_const = _extract_linear_from_quad(
                        expr.left, base_lin, 1.0, flat_vars, model_vars
                    )
                    # (sum ci*xi + k)^2 = sum ci*cj*xi*xj + 2*k*sum ci*xi + k^2
                    items = list(base_lin.items())
                    for a, (i, ci) in enumerate(items):
                        for b, (j, cj) in enumerate(items):
                            if a <= b:
                                key = (min(i, j), max(i, j))
                                coeff = ci * cj
                                if a != b:
                                    coeff *= 2.0  # off-diagonal counted once
                                quad[key] = quad.get(key, 0.0) + multiplier * coeff
                    for i, ci in items:
                        if base_const != 0.0:
                            linear[i] = linear.get(i, 0.0) + multiplier * 2.0 * ci * base_const
                    return multiplier * base_const * base_const
                if power == 1.0:
                    return _extract_quad_recursive(
                        expr.left,
                        quad,
                        linear,
                        multiplier,
                        flat_vars,
                        model_vars,
                    )
                if power == 0.0:
                    return multiplier
            raise ValueError(
                "Nonlinear expression: exponentiation with power != 0, 1, 2 "
                "cannot be exported to MPS/LP format."
            )

        raise ValueError(f"Unknown binary operation '{expr.op}'.")

    if isinstance(expr, SumOverExpression):
        total = 0.0
        for term in expr.terms:
            total += _extract_quad_recursive(term, quad, linear, multiplier, flat_vars, model_vars)
        return total

    if isinstance(expr, SumExpression):
        return _extract_quad_recursive(
            expr.operand, quad, linear, multiplier, flat_vars, model_vars
        )

    raise ValueError(
        f"Expression type '{type(expr).__name__}' cannot be exported. "
        "Only linear and quadratic expressions are supported."
    )


def _extract_linear_from_quad(
    expr: Expression,
    coeffs: dict[int, float],
    multiplier: float,
    flat_vars: list,
    model_vars: list[Variable],
) -> float:
    """Extract only linear terms for use inside quadratic product expansion.

    Same as _extract_linear_recursive but raises if quadratic terms appear.
    """
    return _extract_linear_recursive(expr, coeffs, multiplier, flat_vars, model_vars)


def _is_constant(expr: Expression) -> bool:
    """Check whether an expression contains no variables."""
    if isinstance(expr, Constant):
        return True
    if isinstance(expr, (Variable, IndexExpression)):
        return False
    if isinstance(expr, UnaryOp):
        return _is_constant(expr.operand)
    if isinstance(expr, BinaryOp):
        return _is_constant(expr.left) and _is_constant(expr.right)
    if isinstance(expr, SumOverExpression):
        return all(_is_constant(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _is_constant(expr.operand)
    return False


def _get_constant_value(expr: Expression) -> float:
    """Evaluate a constant expression (no variables) to a float."""
    if isinstance(expr, Constant):
        return float(expr.value) if expr.value.ndim == 0 else float(expr.value.flat[0])
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return -_get_constant_value(expr.operand)
        raise ValueError(f"Cannot evaluate unary '{expr.op}' as constant.")
    if isinstance(expr, BinaryOp):
        left = _get_constant_value(expr.left)
        right = _get_constant_value(expr.right)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "**":
            return float(left**right)
        raise ValueError(f"Cannot evaluate binary '{expr.op}' as constant.")
    raise ValueError(f"Expression type '{type(expr).__name__}' is not constant.")
