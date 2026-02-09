"""GDP (Generalized Disjunctive Programming) reformulation pass.

Converts indicator constraints, disjunctive constraints, and SOS constraints
into standard MINLP constraints via big-M reformulation.

The reformulation is applied as a preprocessing step before the model is
passed to the NLP evaluator and solver. If no GDP constraints exist, the
original model is returned unchanged (zero overhead).
"""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    UnaryOp,
    Variable,
    VarType,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _SOSConstraint,
    _wrap,
)

_DEFAULT_BIG_M = 1e4


def reformulate_gdp(model: Model) -> Model:
    """Replace GDP constraints with standard MINLP constraints via big-M.

    Parameters
    ----------
    model : Model
        Input model potentially containing indicator, disjunctive, or SOS
        constraints.

    Returns
    -------
    Model
        A new model with GDP constraints replaced by equivalent standard
        constraints. If no GDP constraints exist, returns the original
        model unchanged.
    """
    has_gdp = any(
        isinstance(c, (_IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint))
        for c in model._constraints
    )
    if not has_gdp:
        return model

    new_model = Model(model.name)
    # Copy variables (share the same Variable objects so expressions still work)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._objective = model._objective

    # Track auxiliary binaries added by the reformulation
    _aux_counter = [0]

    def _add_aux_binary(prefix: str, shape=()) -> Variable:
        name = f"_gdp_aux_{prefix}_{_aux_counter[0]}"
        _aux_counter[0] += 1
        var = Variable(
            name,
            VarType.BINARY,
            shape if isinstance(shape, tuple) else (shape,),
            0.0,
            1.0,
            new_model,
        )
        new_model._variables.append(var)
        return var

    for c in model._constraints:
        if isinstance(c, _IndicatorConstraint):
            new_cons = _reformulate_indicator(c, new_model)
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _DisjunctiveConstraint):
            new_vars, new_cons = _reformulate_disjunction(c, new_model, _add_aux_binary)
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _SOSConstraint):
            new_cons = _reformulate_sos(c, new_model, _add_aux_binary)
            new_model._constraints.extend(new_cons)
        else:
            # Regular Constraint -- keep as-is
            new_model._constraints.append(c)

    return new_model


def _compute_big_m(
    constraint: Constraint,
    model: Model,
    default: float = _DEFAULT_BIG_M,
) -> float:
    """Compute a big-M bound for a constraint from variable bounds.

    For a constraint ``body sense 0``, we need an upper bound on ``body``
    (for ``<=``), a lower bound (for ``>=``), or both (for ``==``).

    Uses interval arithmetic over the expression tree to get tight M.
    Falls back to *default* if any bound is infinite.

    Parameters
    ----------
    constraint : Constraint
        The constraint whose body expression needs a big-M.
    model : Model
        Model containing variable bound information.
    default : float
        Fallback M value when bounds are infinite.

    Returns
    -------
    float
        A finite big-M value.
    """
    lo, hi = _bound_expression(constraint.body, model)

    # Treat bounds >= 1e15 as effectively infinite (default variable bounds are 1e20)
    _INF_THRESH = 1e15

    if constraint.sense == "<=":
        # body <= 0 is the active constraint; when deactivated we need body <= M
        # so M = upper bound of body
        M = hi if np.isfinite(hi) and abs(hi) < _INF_THRESH else default
    elif constraint.sense == ">=":
        # body >= 0 is active; when deactivated body >= -M
        # so M = -lower_bound of body = abs(lo)
        M = -lo if np.isfinite(lo) and abs(lo) < _INF_THRESH else default
    elif constraint.sense == "==":
        # Need both directions
        M_hi = hi if np.isfinite(hi) and abs(hi) < _INF_THRESH else default
        M_lo = -lo if np.isfinite(lo) and abs(lo) < _INF_THRESH else default
        M = max(M_hi, M_lo)
    else:
        M = default

    # Ensure M is positive and add small safety margin
    return max(abs(M), 1e-8) * 1.01


def _bound_expression(
    expr: Expression,
    model: Model,
) -> tuple[float, float]:
    """Compute interval bounds [lo, hi] for an expression via interval arithmetic.

    Traverses the expression DAG and propagates bounds from variable
    bounds through operations.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of the expression.
    """
    if isinstance(expr, Variable):
        lo = float(np.min(expr.lb))
        hi = float(np.max(expr.ub))
        return lo, hi

    if isinstance(expr, Constant):
        val = float(np.min(expr.value))
        val_max = float(np.max(expr.value))
        return val, val_max

    if isinstance(expr, IndexExpression):
        base_lo, base_hi = _bound_expression(expr.base, model)
        # For indexed expressions on variables, get tighter bounds
        if isinstance(expr.base, Variable):
            v = expr.base
            idx = expr.index
            lb_slice = v.lb[idx] if v.shape != () else v.lb
            ub_slice = v.ub[idx] if v.shape != () else v.ub
            lo = float(np.min(lb_slice))
            hi = float(np.max(ub_slice))
            return lo, hi
        return base_lo, base_hi

    if isinstance(expr, BinaryOp):
        left_lo, left_hi = _bound_expression(expr.left, model)
        right_lo, right_hi = _bound_expression(expr.right, model)

        if expr.op == "+":
            return left_lo + right_lo, left_hi + right_hi
        elif expr.op == "-":
            return left_lo - right_hi, left_hi - right_lo
        elif expr.op == "*":
            products = [
                left_lo * right_lo,
                left_lo * right_hi,
                left_hi * right_lo,
                left_hi * right_hi,
            ]
            return min(products), max(products)
        elif expr.op == "/":
            if right_lo > 0 or right_hi < 0:
                # Divisor doesn't cross zero
                quotients = [
                    left_lo / right_lo,
                    left_lo / right_hi,
                    left_hi / right_lo,
                    left_hi / right_hi,
                ]
                return min(quotients), max(quotients)
            return -np.inf, np.inf
        elif expr.op == "**":
            # Conservative: could be tightened for integer exponents
            if isinstance(expr.right, Constant):
                p = float(expr.right.value)
                if p == 2.0:
                    # x^2 is always >= 0
                    vals = [left_lo**2, left_hi**2]
                    if left_lo <= 0 <= left_hi:
                        return 0.0, max(vals)
                    return min(vals), max(vals)
                if p == int(p) and p > 0:
                    vals = [left_lo**p, left_hi**p]
                    return min(vals), max(vals)
            return -np.inf, np.inf

    if isinstance(expr, UnaryOp):
        arg_lo, arg_hi = _bound_expression(expr.operand, model)
        if expr.op == "neg":
            return -arg_hi, -arg_lo
        elif expr.op == "abs":
            vals = [abs(arg_lo), abs(arg_hi)]
            if arg_lo <= 0 <= arg_hi:
                return 0.0, max(vals)
            return min(vals), max(vals)
        return -np.inf, np.inf

    if isinstance(expr, FunctionCall):
        arg_lo, arg_hi = _bound_expression(expr.args[0], model)
        if expr.func_name == "exp":
            lo = np.exp(arg_lo) if np.isfinite(arg_lo) else 0.0
            hi = np.exp(arg_hi) if np.isfinite(arg_hi) else np.inf
            return lo, hi
        elif expr.func_name == "log":
            lo = np.log(max(arg_lo, 1e-300)) if arg_lo > 0 else -np.inf
            hi = np.log(max(arg_hi, 1e-300)) if arg_hi > 0 else -np.inf
            return lo, hi
        elif expr.func_name == "abs":
            vals = [abs(arg_lo), abs(arg_hi)]
            if arg_lo <= 0 <= arg_hi:
                return 0.0, max(vals)
            return min(vals), max(vals)
        elif expr.func_name == "sqrt":
            lo = np.sqrt(max(arg_lo, 0.0))
            hi = np.sqrt(max(arg_hi, 0.0)) if np.isfinite(arg_hi) else np.inf
            return lo, hi
        elif expr.func_name in ("sin", "cos"):
            # Conservative for trig
            return -1.0, 1.0
        elif expr.func_name == "neg":
            return -arg_hi, -arg_lo

    # Fallback: unknown expression type
    return -np.inf, np.inf


# ── Indicator constraint reformulation ──


def _reformulate_indicator(
    ic: _IndicatorConstraint,
    model: Model,
) -> list[Constraint]:
    """Reformulate an indicator constraint to big-M constraints.

    ``if indicator == active_value then constraint`` becomes:
    - For ``body <= 0``: ``body <= M * (1 - indicator)``
      i.e. ``body - M*(1-indicator) <= 0``
    - For ``body >= 0``: ``body >= -M * (1 - indicator)``
      i.e. ``body + M*(1-indicator) >= 0``
    - For ``body == 0``: both ``<=`` and ``>=`` reformulations

    When ``active_value == 0``, the logic flips: we use ``M * indicator``
    instead of ``M * (1 - indicator)``.
    """
    con = ic.constraint
    y = ic.indicator
    M = _compute_big_m(con, model)

    if ic.active_value == 1:
        # When y=1 constraint is active; when y=0, relaxed by M
        deactivation_expr = _wrap(M) * (_wrap(1.0) - y)
    else:
        # When y=0 constraint is active; when y=1, relaxed by M
        deactivation_expr = _wrap(M) * y

    result = []

    if con.sense == "<=":
        # body <= M*(1-y) => body - M*(1-y) <= 0
        new_body = con.body - deactivation_expr
        result.append(Constraint(body=new_body, sense="<=", rhs=0.0, name=con.name))

    elif con.sense == ">=":
        # body >= -M*(1-y) => body + M*(1-y) >= 0
        new_body = con.body + deactivation_expr
        result.append(Constraint(body=new_body, sense=">=", rhs=0.0, name=con.name))

    elif con.sense == "==":
        # body == 0 when active => -M*(1-y) <= body <= M*(1-y)
        name_le = f"{con.name}_le" if con.name else None
        name_ge = f"{con.name}_ge" if con.name else None
        result.append(
            Constraint(
                body=con.body - deactivation_expr,
                sense="<=",
                rhs=0.0,
                name=name_le,
            )
        )
        result.append(
            Constraint(
                body=con.body + deactivation_expr,
                sense=">=",
                rhs=0.0,
                name=name_ge,
            )
        )

    return result


# ── Disjunctive constraint reformulation ──


def _reformulate_disjunction(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a disjunction via big-M.

    For ``either_or([[g1<=0, g2<=0], [h1<=0, h2<=0]])``:
    1. Introduce binary selectors y_0, y_1 with y_0 + y_1 == 1
    2. For each disjunct k and constraint j in disjunct k:
       g_j(x) <= M_j * (1 - y_k)

    Returns new variables and constraints.
    """
    n_disjuncts = len(dc.disjuncts)
    new_vars = []
    new_cons = []

    # Create selector binaries
    selectors = []
    for k in range(n_disjuncts):
        y_k = add_aux_binary(f"disj_{dc.name or 'anon'}_{k}")
        selectors.append(y_k)
        new_vars.append(y_k)

    # Sum of selectors == 1 (exactly one disjunct active)
    if n_disjuncts == 2:
        sum_expr = selectors[0] + selectors[1]
    else:
        sum_expr = selectors[0]
        for k in range(1, n_disjuncts):
            sum_expr = sum_expr + selectors[k]

    new_cons.append(
        Constraint(
            body=sum_expr - _wrap(1.0),
            sense="==",
            rhs=0.0,
            name=f"_gdp_select_{dc.name}" if dc.name else None,
        )
    )

    # Big-M reformulation for each constraint in each disjunct
    for k, disjunct in enumerate(dc.disjuncts):
        y_k = selectors[k]
        for j, con in enumerate(disjunct):
            M = _compute_big_m(con, model)
            deactivation = _wrap(M) * (_wrap(1.0) - y_k)

            if con.sense == "<=":
                new_body = con.body - deactivation
                new_cons.append(
                    Constraint(
                        body=new_body,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == ">=":
                new_body = con.body + deactivation
                new_cons.append(
                    Constraint(
                        body=new_body,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == "==":
                new_cons.append(
                    Constraint(
                        body=con.body - deactivation,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}_le" if dc.name else None,
                    )
                )
                new_cons.append(
                    Constraint(
                        body=con.body + deactivation,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}_ge" if dc.name else None,
                    )
                )

    return new_vars, new_cons


# ── SOS constraint reformulation ──


def _reformulate_sos(
    sc: _SOSConstraint,
    model: Model,
    add_aux_binary,
) -> list[Constraint]:
    """Reformulate SOS Type 1 or Type 2 constraint via binary indicators.

    SOS1: At most one variable x_i can be nonzero.
      - Introduce binary z_i for each variable
      - x_i <= ub_i * z_i (linking upper bound)
      - x_i >= lb_i * z_i (linking lower bound, handles negative vars)
      - sum(z_i) <= 1

    SOS2: At most two *adjacent* variables can be nonzero.
      - Introduce binary z_i for each variable
      - x_i <= ub_i * z_i
      - x_i >= lb_i * z_i
      - sum(z_i) <= 2
      - z_i + z_j <= 1 for all non-adjacent pairs |i-j| > 1
    """
    new_cons = []
    n = len(sc.variables)

    # Create indicator binaries
    indicators = []
    for i in range(n):
        z_i = add_aux_binary(f"sos{sc.sos_type}_{sc.name or 'anon'}_{i}")
        indicators.append(z_i)

    # Linking constraints: x_i <= ub_i * z_i and x_i >= lb_i * z_i
    for i in range(n):
        v = sc.variables[i]
        # Extract bounds — v may be a Variable or IndexExpression
        lo_v, hi_v = _bound_expression(v, model)
        ub_val = hi_v
        lb_val = lo_v

        # Clamp to finite values for big-M
        if not np.isfinite(ub_val):
            ub_val = _DEFAULT_BIG_M
        if not np.isfinite(lb_val):
            lb_val = -_DEFAULT_BIG_M

        # x_i - ub_i * z_i <= 0
        new_cons.append(
            Constraint(
                body=v - _wrap(ub_val) * indicators[i],
                sense="<=",
                rhs=0.0,
                name=f"_sos{sc.sos_type}_{sc.name}_ub_{i}" if sc.name else None,
            )
        )

        # x_i - lb_i * z_i >= 0  =>  x_i >= lb_i * z_i
        if lb_val < 0:
            new_cons.append(
                Constraint(
                    body=v - _wrap(lb_val) * indicators[i],
                    sense=">=",
                    rhs=0.0,
                    name=f"_sos{sc.sos_type}_{sc.name}_lb_{i}" if sc.name else None,
                )
            )

    if sc.sos_type == 1:
        # sum(z_i) <= 1
        if n == 1:
            sum_z = indicators[0]
        else:
            sum_z = indicators[0]
            for i in range(1, n):
                sum_z = sum_z + indicators[i]
        new_cons.append(
            Constraint(
                body=sum_z - _wrap(1.0),
                sense="<=",
                rhs=0.0,
                name=f"_sos1_{sc.name}_sum" if sc.name else None,
            )
        )

    elif sc.sos_type == 2:
        # sum(z_i) <= 2
        if n >= 2:
            sum_z = indicators[0]
            for i in range(1, n):
                sum_z = sum_z + indicators[i]
            new_cons.append(
                Constraint(
                    body=sum_z - _wrap(2.0),
                    sense="<=",
                    rhs=0.0,
                    name=f"_sos2_{sc.name}_sum" if sc.name else None,
                )
            )

            # Non-adjacency: z_i + z_j <= 1 for |i - j| > 1
            for i in range(n):
                for j in range(i + 2, n):
                    new_cons.append(
                        Constraint(
                            body=indicators[i] + indicators[j] - _wrap(1.0),
                            sense="<=",
                            rhs=0.0,
                            name=(f"_sos2_{sc.name}_nonadj_{i}_{j}" if sc.name else None),
                        )
                    )

    return new_cons
