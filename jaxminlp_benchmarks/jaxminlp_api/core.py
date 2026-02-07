"""
JaxMINLP Modeling API

A clean, expressive Python API for formulating Mixed-Integer Nonlinear Programs.
Designed for:
  - Readability: models look like the math
  - JAX compatibility: expressions are traceable and JIT-compilable
  - Rust interop: expression graphs map to the Rust DAG for structure detection
  - LLM integration: the API doubles as the tool-calling schema for the formulation agent

Example:
    import jaxminlp as jm

    m = jm.Model("blending")
    x = m.continuous("flow", shape=(3,), lb=0, ub=100)
    y = m.binary("active", shape=(2,))

    m.minimize(cost @ x + fixed_cost @ y)
    m.subject_to(A @ x <= b, name="mass_balance")
    m.subject_to(x[0] * x[1] <= 50 * y[0], name="bilinear_coupling")

    result = m.solve()
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

import numpy as np


# ─────────────────────────────────────────────────────────────
# Variable Types
# ─────────────────────────────────────────────────────────────

class VarType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"


# ─────────────────────────────────────────────────────────────
# Expression System
#
# All operations on Variables produce Expression objects that
# build a DAG. This DAG is later compiled to:
#   (1) A Rust-side expression graph for structure detection
#   (2) A JAX-traceable function for evaluation and autodiff
# ─────────────────────────────────────────────────────────────

class Expression:
    """
    Base class for all mathematical expressions in a JaxMINLP model.

    Supports standard arithmetic (+, -, *, /, **), comparison operators
    (<=, >=, ==) that produce Constraint objects, and mathematical
    functions via the jm namespace (jm.exp, jm.log, jm.sin, etc.).

    Expressions are lazy — they build a DAG, not compute values.
    """

    def __add__(self, other):
        return BinaryOp("+", self, _wrap(other))

    def __radd__(self, other):
        return BinaryOp("+", _wrap(other), self)

    def __sub__(self, other):
        return BinaryOp("-", self, _wrap(other))

    def __rsub__(self, other):
        return BinaryOp("-", _wrap(other), self)

    def __mul__(self, other):
        return BinaryOp("*", self, _wrap(other))

    def __rmul__(self, other):
        return BinaryOp("*", _wrap(other), self)

    def __truediv__(self, other):
        return BinaryOp("/", self, _wrap(other))

    def __rtruediv__(self, other):
        return BinaryOp("/", _wrap(other), self)

    def __pow__(self, other):
        return BinaryOp("**", self, _wrap(other))

    def __rpow__(self, other):
        return BinaryOp("**", _wrap(other), self)

    def __neg__(self):
        return UnaryOp("neg", self)

    def __abs__(self):
        return UnaryOp("abs", self)

    # ── Comparison operators produce Constraints, not booleans ──

    def __le__(self, other):
        return Constraint(self - _wrap(other), sense="<=", rhs=0.0)

    def __ge__(self, other):
        return Constraint(_wrap(other) - self, sense="<=", rhs=0.0)

    def __eq__(self, other):
        return Constraint(self - _wrap(other), sense="==", rhs=0.0)

    # ── Indexing for array variables ──

    def __getitem__(self, idx):
        return IndexExpression(self, idx)

    # ── Matrix operations ──

    def __matmul__(self, other):
        return MatMulExpression(self, _wrap(other))

    def __rmatmul__(self, other):
        return MatMulExpression(_wrap(other), self)

    def _repr_latex_(self):
        """Jupyter/IPython LaTeX rendering."""
        return f"${self}$"


class Constant(Expression):
    """A numeric constant in the expression DAG."""

    def __init__(self, value: Union[float, int, np.ndarray]):
        if isinstance(value, np.ndarray):
            self.value = value.astype(np.float64)
        else:
            self.value = np.asarray(value, dtype=np.float64)

    def __repr__(self):
        if self.value.ndim == 0:
            return f"{float(self.value):.6g}"
        return f"Constant({self.value.shape})"


class Variable(Expression):
    """
    A decision variable in the optimization problem.

    Variables are created through Model.continuous(), Model.binary(),
    or Model.integer() — not directly.
    """

    def __init__(
        self,
        name: str,
        var_type: VarType,
        shape: tuple[int, ...],
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        model: "Model",
    ):
        self.name = name
        self.var_type = var_type
        self.shape = shape
        self.lb = np.broadcast_to(np.asarray(lb, dtype=np.float64), shape)
        self.ub = np.broadcast_to(np.asarray(ub, dtype=np.float64), shape)
        self.model = model
        self._index = len(model._variables)  # Position in flat variable vector

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __repr__(self):
        if self.shape == () or self.shape == (1,):
            return self.name
        return f"{self.name}{list(self.shape)}"


class IndexExpression(Expression):
    """Result of indexing into an array variable: x[i] or x[0, 1]."""

    def __init__(self, base: Expression, index):
        self.base = base
        self.index = index

    def __repr__(self):
        return f"{self.base}[{self.index}]"


class BinaryOp(Expression):
    """Binary operation: a op b."""

    def __init__(self, op: str, left: Expression, right: Expression):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class UnaryOp(Expression):
    """Unary operation: op(a)."""

    def __init__(self, op: str, operand: Expression):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"{self.op}({self.operand})"


class FunctionCall(Expression):
    """Named function call: exp(x), log(x), sin(x), etc."""

    def __init__(self, func_name: str, *args: Expression):
        self.func_name = func_name
        self.args = args

    def __repr__(self):
        arg_str = ", ".join(str(a) for a in self.args)
        return f"{self.func_name}({arg_str})"


class MatMulExpression(Expression):
    """Matrix multiplication: A @ x."""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} @ {self.right})"


class SumExpression(Expression):
    """Summation over expressions."""

    def __init__(self, operand: Expression, axis: Optional[int] = None):
        self.operand = operand
        self.axis = axis

    def __repr__(self):
        if self.axis is not None:
            return f"sum({self.operand}, axis={self.axis})"
        return f"sum({self.operand})"


class SumOverExpression(Expression):
    """Sum of expr(i) for i in index_set — the indexed summation pattern."""

    def __init__(self, terms: list[Expression]):
        self.terms = terms

    def __repr__(self):
        return f"Σ[{len(self.terms)} terms]"


def _wrap(x) -> Expression:
    """Convert a Python scalar or numpy array to a Constant expression."""
    if isinstance(x, Expression):
        return x
    return Constant(x)


# ─────────────────────────────────────────────────────────────
# Mathematical Functions (jm.exp, jm.log, jm.sin, etc.)
# ─────────────────────────────────────────────────────────────

def exp(x: Union[Expression, float]) -> Expression:
    """Exponential function."""
    return FunctionCall("exp", _wrap(x))

def log(x: Union[Expression, float]) -> Expression:
    """Natural logarithm."""
    return FunctionCall("log", _wrap(x))

def log2(x: Union[Expression, float]) -> Expression:
    """Base-2 logarithm."""
    return FunctionCall("log2", _wrap(x))

def log10(x: Union[Expression, float]) -> Expression:
    """Base-10 logarithm."""
    return FunctionCall("log10", _wrap(x))

def sqrt(x: Union[Expression, float]) -> Expression:
    """Square root."""
    return FunctionCall("sqrt", _wrap(x))

def sin(x: Union[Expression, float]) -> Expression:
    """Sine."""
    return FunctionCall("sin", _wrap(x))

def cos(x: Union[Expression, float]) -> Expression:
    """Cosine."""
    return FunctionCall("cos", _wrap(x))

def tan(x: Union[Expression, float]) -> Expression:
    """Tangent."""
    return FunctionCall("tan", _wrap(x))

def abs_(x: Union[Expression, float]) -> Expression:
    """Absolute value (use jm.abs_ to avoid shadowing builtins)."""
    return FunctionCall("abs", _wrap(x))

def sign(x: Union[Expression, float]) -> Expression:
    """Sign function."""
    return FunctionCall("sign", _wrap(x))

def minimum(x: Union[Expression, float], y: Union[Expression, float]) -> Expression:
    """Element-wise minimum."""
    return FunctionCall("min", _wrap(x), _wrap(y))

def maximum(x: Union[Expression, float], y: Union[Expression, float]) -> Expression:
    """Element-wise maximum."""
    return FunctionCall("max", _wrap(x), _wrap(y))


# ─────────────────────────────────────────────────────────────
# Aggregation Functions
# ─────────────────────────────────────────────────────────────

def sum(x: Union[Expression, list, Callable], *,
        over: Optional[Sequence] = None,
        axis: Optional[int] = None) -> Expression:
    """
    Summation.

    Three calling patterns:
        jm.sum(x)                      # sum all elements of array variable x
        jm.sum(x, axis=0)              # sum along axis
        jm.sum(lambda i: cost[i]*x[i], over=range(n))  # indexed sum
    """
    if over is not None and callable(x):
        # Indexed summation: jm.sum(lambda i: expr(i), over=index_set)
        terms = [_wrap(x(i)) for i in over]
        return SumOverExpression(terms)
    if isinstance(x, list):
        terms = [_wrap(t) for t in x]
        return SumOverExpression(terms)
    return SumExpression(_wrap(x), axis=axis)

def prod(x: Union[Expression, list, Callable], *,
         over: Optional[Sequence] = None) -> Expression:
    """Product — analogous to sum."""
    if over is not None and callable(x):
        terms = [_wrap(x(i)) for i in over]
        result = terms[0]
        for t in terms[1:]:
            result = result * t
        return result
    return FunctionCall("prod", _wrap(x))

def norm(x: Expression, ord: int = 2) -> Expression:
    """Vector norm."""
    return FunctionCall(f"norm{ord}", _wrap(x))


# ─────────────────────────────────────────────────────────────
# Constraints
# ─────────────────────────────────────────────────────────────

class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


@dataclass
class Constraint:
    """
    A single constraint in the model.

    Internally stored as: body sense rhs
    where body is an Expression and rhs is 0.0 (normalized form).

    Created via comparison operators on Expressions:
        x[0] + x[1] <= 10
        jm.exp(x[2]) == 1.0
        A @ x >= b
    """
    body: Expression
    sense: str
    rhs: float = 0.0
    name: Optional[str] = None

    def __repr__(self):
        return f"{self.body} {self.sense} {self.rhs}"


@dataclass
class ConstraintList:
    """A collection of constraints created from vectorized expressions."""
    constraints: list[Constraint]
    name: Optional[str] = None

    def __len__(self):
        return len(self.constraints)


# ─────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────

class ObjectiveSense(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """Objective function with sense (minimize/maximize)."""
    expression: Expression
    sense: ObjectiveSense


# ─────────────────────────────────────────────────────────────
# Parameter (for parametric optimization / sensitivity)
# ─────────────────────────────────────────────────────────────

class Parameter(Expression):
    """
    A parameter — a value that is fixed during a single solve
    but can change between solves (for parametric studies, sensitivity
    analysis, or differentiating through the solve).

    Unlike constants, parameters are tracked in the expression DAG
    so that JAX can differentiate the optimal objective w.r.t. them.

    Example:
        price = m.parameter("price", value=50.0)
        m.minimize(price * x[0] + cost * x[1])

        result = m.solve()
        sensitivity = result.gradient(price)  # d(obj*)/d(price) via implicit diff
    """

    def __init__(self, name: str, value: Union[float, np.ndarray], model: "Model"):
        self.name = name
        self.value = np.asarray(value, dtype=np.float64)
        self.shape = self.value.shape
        self.model = model

    def __repr__(self):
        return f"param({self.name})"


# ─────────────────────────────────────────────────────────────
# Solve Result
# ─────────────────────────────────────────────────────────────

@dataclass
class SolveResult:
    """
    Result of solving a JaxMINLP model.

    Provides solution values, solve statistics, explanation,
    and sensitivity analysis (for models with Parameters).
    """
    status: str                             # optimal, feasible, infeasible, ...
    objective: Optional[float] = None       # Optimal objective value
    bound: Optional[float] = None           # Best dual bound
    gap: Optional[float] = None             # Relative optimality gap
    x: Optional[dict[str, np.ndarray]] = None  # Variable values by name
    wall_time: float = 0.0
    node_count: int = 0

    # Layer profiling
    rust_time: float = 0.0
    jax_time: float = 0.0
    python_time: float = 0.0

    # LLM explanation (populated if llm=True)
    _explanation: Optional[str] = None
    _model: Optional["Model"] = None

    def value(self, var: Variable) -> np.ndarray:
        """Get the optimal value of a variable."""
        if self.x is None:
            raise ValueError("No solution available")
        return self.x[var.name]

    def explain(self) -> str:
        """Get LLM-generated explanation of the solve."""
        if self._explanation:
            return self._explanation
        return (
            f"Solved to {self.status} in {self.wall_time:.1f}s. "
            f"Objective: {self.objective}, Gap: {self.gap}, "
            f"Nodes: {self.node_count}"
        )

    def gradient(self, param: Parameter) -> np.ndarray:
        """
        Sensitivity of optimal objective w.r.t. a parameter.

        Uses implicit differentiation through KKT conditions.
        Requires the model to have been solved with sensitivity=True.
        """
        raise NotImplementedError(
            "Sensitivity analysis requires JAX backend (Phase 3 feature)"
        )

    def __repr__(self):
        return (
            f"SolveResult(status={self.status!r}, obj={self.objective}, "
            f"gap={self.gap}, time={self.wall_time:.1f}s, nodes={self.node_count})"
        )


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class Model:
    """
    A Mixed-Integer Nonlinear Program.

    Usage:
        m = jm.Model("my_problem")

        # Variables
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        y = m.binary("y", shape=(2,))
        n = m.integer("n", lb=0, ub=100)

        # Parameters (for sensitivity analysis)
        price = m.parameter("price", value=50.0)

        # Objective
        m.minimize(price * jm.sum(x) + jm.sum(fixed_cost * y))

        # Constraints
        m.subject_to(A @ x <= b, name="capacity")
        m.subject_to(x[0] * x[1] <= 50 * y[0], name="bilinear")
        m.subject_to(jm.exp(x[2]) + x[0] == 5.0, name="nonlinear_eq")

        # Solve
        result = m.solve()
        print(result.value(x))
    """

    def __init__(self, name: str = "model"):
        self.name = name
        self._variables: list[Variable] = []
        self._parameters: list[Parameter] = []
        self._constraints: list[Constraint] = []
        self._objective: Optional[Objective] = None

    # ── Variable constructors ──

    def continuous(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = -1e20,
        ub: Union[float, np.ndarray] = 1e20,
    ) -> Variable:
        """
        Create continuous decision variable(s).

        Args:
            name: Variable name (must be unique in model)
            shape: Scalar () or tuple for array variables
            lb: Lower bound (scalar or array matching shape)
            ub: Upper bound (scalar or array matching shape)

        Returns:
            Variable expression that can be used in objectives and constraints

        Examples:
            x = m.continuous("x")                    # scalar, unbounded
            flow = m.continuous("flow", shape=(5,), lb=0)  # 5-vector, non-negative
            X = m.continuous("X", shape=(3, 4), lb=0, ub=1)  # 3x4 matrix
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.CONTINUOUS, shape, lb, ub, self)
        self._variables.append(var)
        return var

    def binary(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
    ) -> Variable:
        """
        Create binary (0/1) decision variable(s).

        Examples:
            use = m.binary("use")                   # single binary
            active = m.binary("active", shape=(5,)) # 5 binary indicators
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.BINARY, shape, 0.0, 1.0, self)
        self._variables.append(var)
        return var

    def integer(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]] = (),
        lb: Union[float, np.ndarray] = 0,
        ub: Union[float, np.ndarray] = 1e6,
    ) -> Variable:
        """
        Create general integer decision variable(s).

        Examples:
            n = m.integer("n_units", lb=0, ub=10)
            batch = m.integer("batch", shape=(3,), lb=1, ub=100)
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._check_name(name)
        var = Variable(name, VarType.INTEGER, shape, lb, ub, self)
        self._variables.append(var)
        return var

    def parameter(
        self,
        name: str,
        value: Union[float, np.ndarray],
    ) -> Parameter:
        """
        Create a parameter for parametric optimization / sensitivity.

        Parameters are fixed during a solve but tracked in the expression
        DAG for differentiation. Use for: prices, demands, specifications
        that you want to do sensitivity analysis on.

        Examples:
            price = m.parameter("price", value=50.0)
            demand = m.parameter("demand", value=np.array([100, 200, 150]))
        """
        self._check_name(name)
        param = Parameter(name, value, self)
        self._parameters.append(param)
        return param

    # ── Objective ──

    def minimize(self, expr: Expression):
        """
        Set the objective to minimize.

        Args:
            expr: Expression to minimize

        Examples:
            m.minimize(cost @ x)
            m.minimize(jm.sum(lambda i: c[i] * x[i], over=range(n)))
        """
        self._objective = Objective(_wrap(expr), ObjectiveSense.MINIMIZE)

    def maximize(self, expr: Expression):
        """
        Set the objective to maximize.

        Examples:
            m.maximize(profit @ x - jm.sum(penalty * y))
        """
        self._objective = Objective(_wrap(expr), ObjectiveSense.MAXIMIZE)

    # ── Constraints ──

    def subject_to(
        self,
        constraint: Union[Constraint, list[Constraint], bool],
        name: Optional[str] = None,
    ):
        """
        Add constraint(s) to the model.

        Constraints are created by comparison operators on expressions:
            m.subject_to(x[0] + x[1] <= 10)           # inequality
            m.subject_to(jm.exp(x[0]) == 1.0)          # equality
            m.subject_to(A @ x <= b)                    # vectorized
            m.subject_to(A @ x <= b, name="capacity")   # named

        Indexed constraints:
            m.subject_to([
                x[i] + x[i+1] <= limits[i]
                for i in range(n-1)
            ], name="adjacent_limits")

        Named constraints enable better LLM explanations and debugging.
        """
        if isinstance(constraint, list):
            for k, c in enumerate(constraint):
                if isinstance(c, Constraint):
                    c.name = f"{name}_{k}" if name else None
                    self._constraints.append(c)
        elif isinstance(constraint, Constraint):
            constraint.name = name
            self._constraints.append(constraint)
        else:
            raise TypeError(
                f"Expected Constraint (from <=, >=, == on expressions), "
                f"got {type(constraint)}. Did you mean to compare expressions?"
            )

    # ── Logical constraints (GDP) ──

    def if_then(
        self,
        indicator: Variable,
        then_constraints: list[Constraint],
        name: Optional[str] = None,
    ):
        """
        Indicator (if-then) constraint.

        If indicator == 1, then all then_constraints must hold.
        If indicator == 0, then_constraints are relaxed.

        This avoids manual big-M formulation.

        Examples:
            m.if_then(y[0], [
                x[0] >= 10,
                x[1] <= 50,
            ], name="unit0_active")
        """
        for k, c in enumerate(then_constraints):
            c.name = f"{name}_then_{k}" if name else None
            # Store as indicator constraint; Rust presolve will handle
            # reformulation to big-M or GDP branching
            self._constraints.append(_IndicatorConstraint(
                indicator=indicator,
                constraint=c,
                active_value=1,
            ))

    def either_or(
        self,
        disjuncts: list[list[Constraint]],
        name: Optional[str] = None,
    ):
        """
        Disjunctive constraint (GDP).

        Exactly one group of constraints must hold.

        Examples:
            m.either_or([
                [x[0] <= 5, x[1] >= 10],   # mode A
                [x[0] >= 15, x[1] <= 3],   # mode B
            ], name="operating_mode")
        """
        self._constraints.append(_DisjunctiveConstraint(
            disjuncts=disjuncts,
            name=name,
        ))

    # ── Special ordered sets ──

    def sos1(self, variables: list[Variable], name: Optional[str] = None):
        """SOS Type 1: at most one variable in the set can be nonzero."""
        self._constraints.append(_SOSConstraint(1, variables, name))

    def sos2(self, variables: list[Variable], name: Optional[str] = None):
        """SOS Type 2: at most two adjacent variables can be nonzero."""
        self._constraints.append(_SOSConstraint(2, variables, name))

    # ── Solve ──

    def solve(
        self,
        time_limit: float = 3600,
        gap_tolerance: float = 1e-4,
        threads: int = 1,
        gpu: bool = True,
        llm: bool = False,
        sensitivity: bool = False,
        stream: bool = False,
        deterministic: bool = True,
        **kwargs,
    ) -> Union[SolveResult, Iterator["SolveUpdate"]]:
        """
        Solve the model.

        Args:
            time_limit: Wall-clock time limit in seconds
            gap_tolerance: Relative optimality gap tolerance
            threads: Number of CPU threads for Rust components
            gpu: Enable GPU acceleration for JAX components
            llm: Enable LLM explanation of results
            sensitivity: Compute sensitivities w.r.t. Parameters
            stream: Return iterator of SolveUpdate instead of final result
            deterministic: Ensure reproducible results across runs

        Returns:
            SolveResult (or Iterator[SolveUpdate] if stream=True)
        """
        self.validate()

        if stream:
            return self._solve_streaming(
                time_limit=time_limit, gap_tolerance=gap_tolerance, **kwargs
            )

        from discopt.solver import solve_model
        return solve_model(
            self,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            threads=threads,
            gpu=gpu,
            deterministic=deterministic,
            **kwargs,
        )

    def _solve_streaming(self, **kwargs) -> Iterator["SolveUpdate"]:
        """Streaming solve that yields updates during B&B."""
        raise NotImplementedError("Streaming solve requires solver backend")

    # ── Validation ──

    def validate(self):
        """
        Validate model consistency.

        Checks:
        - Objective is set
        - All variable names are unique
        - Variable bounds are consistent (lb <= ub)
        - Binary variables have lb=0, ub=1
        - No circular references in expression DAG
        """
        if self._objective is None:
            raise ValueError("No objective set. Call m.minimize() or m.maximize().")

        names = set()
        for var in self._variables:
            if var.name in names:
                raise ValueError(f"Duplicate variable name: '{var.name}'")
            names.add(var.name)
            if np.any(var.lb > var.ub):
                raise ValueError(
                    f"Variable '{var.name}' has lb > ub at some index"
                )

    # ── Model statistics ──

    @property
    def num_variables(self) -> int:
        return builtins_sum(v.size for v in self._variables)

    @property
    def num_continuous(self) -> int:
        return builtins_sum(
            v.size for v in self._variables if v.var_type == VarType.CONTINUOUS
        )

    @property
    def num_integer(self) -> int:
        return builtins_sum(
            v.size for v in self._variables
            if v.var_type in (VarType.INTEGER, VarType.BINARY)
        )

    @property
    def num_constraints(self) -> int:
        return len(self._constraints)

    def summary(self) -> str:
        """Human-readable model summary."""
        lines = [
            f"Model: {self.name}",
            f"  Variables: {self.num_variables} "
            f"({self.num_continuous} continuous, {self.num_integer} integer/binary)",
            f"  Constraints: {self.num_constraints}",
            f"  Objective: {self._objective.sense.value} {self._objective.expression}",
            f"  Parameters: {len(self._parameters)}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    def _check_name(self, name: str):
        """Ensure variable/parameter name is unique."""
        existing = {v.name for v in self._variables} | {p.name for p in self._parameters}
        if name in existing:
            raise ValueError(f"Name '{name}' already used in model")


# Internal constraint types (not part of public API)

@dataclass
class _IndicatorConstraint:
    indicator: Variable
    constraint: Constraint
    active_value: int = 1
    name: Optional[str] = None

@dataclass
class _DisjunctiveConstraint:
    disjuncts: list[list[Constraint]]
    name: Optional[str] = None

@dataclass
class _SOSConstraint:
    sos_type: int
    variables: list[Variable]
    name: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Streaming updates
# ─────────────────────────────────────────────────────────────

@dataclass
class SolveUpdate:
    """Intermediate update from streaming solve."""
    elapsed: float
    incumbent: Optional[float]
    lower_bound: float
    gap: Optional[float]
    node_count: int
    open_nodes: int
    message: Optional[str] = None  # LLM commentary if llm=True


# ─────────────────────────────────────────────────────────────
# Import functions
# ─────────────────────────────────────────────────────────────

def from_pyomo(pyomo_model) -> Model:
    """
    Import a Pyomo ConcreteModel as a JaxMINLP model.

    Supports: Var, Constraint, Objective, Param, Set.
    GDP (Disjunct/Disjunction) is mapped to jm.either_or.

    Example:
        import pyomo.environ as pyo
        import jaxminlp as jm

        pyo_model = pyo.ConcreteModel()
        # ... build Pyomo model ...

        jm_model = jm.from_pyomo(pyo_model)
        result = jm_model.solve(gpu=True)
    """
    raise NotImplementedError("Pyomo import requires pyomo bridge (Phase 4)")


def from_nl(path: str) -> Model:
    """
    Import a model from AMPL .nl format.

    Uses the Rust .nl parser for speed.

    Example:
        model = jm.from_nl("problem.nl")
        result = model.solve()
    """
    raise NotImplementedError("NL import requires Rust parser (Phase 1)")


def from_gams(path: str) -> Model:
    """
    Import a model from GAMS .gms format.

    Example:
        model = jm.from_gams("problem.gms")
    """
    raise NotImplementedError("GAMS import requires Rust parser (Phase 1)")


def from_description(
    description: str,
    data: Optional[dict] = None,
    llm_model: str = "claude-sonnet-4-20250514",
    validate: bool = True,
    explain: bool = True,
) -> Model:
    """
    Create a model from a natural language description using LLM.

    The LLM generates a JaxMINLP model via function calling
    (not free-form code generation), ensuring type safety.

    Args:
        description: Natural language problem description
        data: Dict of named data arrays (DataFrames, numpy arrays, dicts)
        llm_model: LLM model to use for formulation
        validate: Validate the generated model before returning
        explain: Print LLM's explanation of the formulation

    Example:
        model = jm.from_description(
            "Minimize total shipping cost from 3 warehouses to 5 customers. "
            "Each warehouse has limited supply. Each customer's demand must be met. "
            "We must decide which warehouses to open (fixed cost) and how much "
            "to ship on each route.",
            data={
                "supply": [100, 150, 200],
                "demand": [80, 60, 70, 40, 50],
                "shipping_cost": cost_matrix,    # 3x5 array
                "fixed_cost": [500, 600, 450],
            },
        )
        result = model.solve()
    """
    raise NotImplementedError("LLM formulation requires LLM integration (Phase 2)")


# Keep Python's built-in sum accessible
import builtins as _builtins
builtins_sum = _builtins.sum
