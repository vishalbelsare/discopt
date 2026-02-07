"""
JaxMINLP — An LLM-native global optimization solver.

import jaxminlp as jm

m = jm.Model("my_problem")
x = m.continuous("x", shape=(3,), lb=0, ub=10)
y = m.binary("y", shape=(2,))

m.minimize(cost @ x + fixed_cost @ y)
m.subject_to(A @ x <= b, name="capacity")

result = m.solve()
"""

from jaxminlp_api.core import (
    # Model
    Model,

    # Variable types (for isinstance checks, rarely needed)
    Variable,
    VarType,
    Parameter,

    # Expressions (for isinstance checks, rarely needed)
    Expression,
    Constraint,

    # Mathematical functions
    exp,
    log,
    log2,
    log10,
    sqrt,
    sin,
    cos,
    tan,
    abs_ as abs,
    sign,
    minimum,
    maximum,

    # Aggregation
    sum,
    prod,
    norm,

    # Results
    SolveResult,
    SolveUpdate,

    # Import functions
    from_pyomo,
    from_nl,
    from_gams,
    from_description,
)

__version__ = "0.1.0"
__all__ = [
    "Model",
    "Variable",
    "VarType",
    "Parameter",
    "Expression",
    "Constraint",
    "exp", "log", "log2", "log10", "sqrt",
    "sin", "cos", "tan",
    "abs", "sign", "minimum", "maximum",
    "sum", "prod", "norm",
    "SolveResult", "SolveUpdate",
    "from_pyomo", "from_nl", "from_gams", "from_description",
]
