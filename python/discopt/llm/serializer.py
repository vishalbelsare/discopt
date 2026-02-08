"""
Serialize Model and SolveResult to LLM-friendly text representations.

The serialized output is equivalent to what ``model.summary()`` provides —
no secrets or internal data beyond what the user can already see.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model, SolveResult


def serialize_model(model: Model) -> str:
    """Serialize a Model to a structured text description for LLM consumption.

    Parameters
    ----------
    model : Model
        The optimization model to serialize.

    Returns
    -------
    str
        Human-readable text describing the model structure.
    """
    lines = [f"# Model: {model.name}", ""]

    # Variables
    lines.append("## Variables")
    for var in model._variables:
        vtype = var.var_type.value
        shape_str = f"shape={var.shape}" if var.shape else "scalar"
        lb_str = _format_bound(var.lb, "lb")
        ub_str = _format_bound(var.ub, "ub")
        lines.append(f"- {var.name} ({vtype}, {shape_str}, {lb_str}, {ub_str})")
    lines.append("")

    # Parameters
    if model._parameters:
        lines.append("## Parameters")
        for param in model._parameters:
            val_str = _format_value(param.value)
            lines.append(f"- {param.name} = {val_str}")
        lines.append("")

    # Objective
    if model._objective is not None:
        lines.append("## Objective")
        lines.append(f"{model._objective.sense.value} {model._objective.expression}")
        lines.append("")

    # Constraints
    if model._constraints:
        lines.append(f"## Constraints ({len(model._constraints)} total)")
        for i, con in enumerate(model._constraints):
            name = getattr(con, "name", None) or f"c{i}"
            lines.append(f"- [{name}] {con}")
        lines.append("")

    # Statistics
    lines.append("## Statistics")
    lines.append(f"- Total variables: {model.num_variables}")
    lines.append(f"- Continuous: {model.num_continuous}")
    lines.append(f"- Integer/Binary: {model.num_integer}")
    lines.append(f"- Constraints: {model.num_constraints}")

    return "\n".join(lines)


def serialize_solve_result(result: SolveResult, model: Model | None = None) -> str:
    """Serialize a SolveResult to structured text for LLM consumption.

    Parameters
    ----------
    result : SolveResult
        The solve result to serialize.
    model : Model, optional
        The model that was solved (for additional context).

    Returns
    -------
    str
        Human-readable text describing the solve result.
    """
    lines = ["# Solve Result", ""]

    # Status and objective
    lines.append(f"Status: {result.status}")
    if result.objective is not None:
        lines.append(f"Objective value: {result.objective:.8g}")
    if result.bound is not None:
        lines.append(f"Best bound: {result.bound:.8g}")
    if result.gap is not None:
        lines.append(f"Relative gap: {result.gap:.4%}")
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append(f"- Wall time: {result.wall_time:.3f}s")
    rust_pct = _pct(result.rust_time, result.wall_time)
    jax_pct = _pct(result.jax_time, result.wall_time)
    py_pct = _pct(result.python_time, result.wall_time)
    lines.append(f"- Rust (B&B): {result.rust_time:.3f}s ({rust_pct})")
    lines.append(f"- JAX (NLP): {result.jax_time:.3f}s ({jax_pct})")
    lines.append(f"- Python: {result.python_time:.3f}s ({py_pct})")
    lines.append(f"- Nodes explored: {result.node_count}")
    lines.append("")

    # Solution values
    if result.x is not None:
        lines.append("## Solution")
        for name, val in result.x.items():
            lines.append(f"- {name} = {_format_value(val)}")
        lines.append("")

    # Model context
    if model is not None:
        lines.append("## Model Context")
        lines.append(model.summary())

    return "\n".join(lines)


def serialize_data_schema(data: dict) -> str:
    """Serialize a data dictionary's schema for LLM consumption.

    Describes the shape, dtype, and sample values of each data item,
    without sending all the data to the LLM.

    Parameters
    ----------
    data : dict
        Named data arrays (DataFrames, numpy arrays, dicts, lists).

    Returns
    -------
    str
        Text description of the data schema.
    """
    lines = ["# Available Data", ""]

    for name, value in data.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            # numpy array or similar
            lines.append(f"## {name}")
            lines.append("- Type: numpy array")
            lines.append(f"- Shape: {value.shape}")
            lines.append(f"- Dtype: {value.dtype}")
            flat = np.asarray(value).ravel()
            sample = flat[:5]
            lines.append(f"- Sample values: {sample.tolist()}")
            if flat.size > 0:
                lines.append(f"- Range: [{float(flat.min()):.4g}, {float(flat.max()):.4g}]")
        elif hasattr(value, "columns"):
            # DataFrame-like
            lines.append(f"## {name}")
            lines.append("- Type: DataFrame")
            lines.append(f"- Shape: {value.shape}")
            lines.append(f"- Columns: {list(value.columns)}")
            lines.append(f"- Dtypes: {dict(value.dtypes)}")
            lines.append(f"- Head:\n{value.head(3).to_string()}")
        elif isinstance(value, dict):
            lines.append(f"## {name}")
            lines.append(f"- Type: dict with {len(value)} keys")
            keys = list(value.keys())[:5]
            lines.append(f"- Sample keys: {keys}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"## {name}")
            lines.append(f"- Type: {type(value).__name__} with {len(value)} elements")
            lines.append(f"- Sample: {value[:5]}")
        else:
            lines.append(f"## {name}")
            lines.append(f"- Type: {type(value).__name__}")
            lines.append(f"- Value: {value}")
        lines.append("")

    return "\n".join(lines)


def _format_bound(bound, label: str) -> str:
    """Format a variable bound for display."""
    if isinstance(bound, np.ndarray):
        if bound.size == 1:
            val = float(bound.ravel()[0])
        else:
            return f"{label}=array({bound.shape})"
    else:
        val = float(bound)
    if val <= -1e19:
        return f"{label}=-inf"
    if val >= 1e19:
        return f"{label}=+inf"
    return f"{label}={val:g}"


def _format_value(val) -> str:
    """Format a numpy value for display."""
    if isinstance(val, np.ndarray):
        if val.size <= 10:
            return str(val.tolist())
        return f"array(shape={val.shape}, mean={val.mean():.4g})"
    return str(val)


def _pct(part: float, total: float) -> str:
    """Format a percentage."""
    if total <= 0:
        return "N/A"
    return f"{100 * part / total:.1f}%"
