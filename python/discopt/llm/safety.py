"""
Safety and validation for LLM outputs.

Ensures LLM-generated content never compromises solver correctness.
All validation is deterministic — LLM outputs are advisory only.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discopt.modeling.core import Model

logger = logging.getLogger(__name__)


def validate_explanation(text: str) -> str:
    """Validate and sanitize an LLM-generated explanation.

    Parameters
    ----------
    text : str
        Raw LLM output text.

    Returns
    -------
    str
        Sanitized explanation text.
    """
    if not text or not text.strip():
        return "No explanation available."

    # Truncate overly long explanations
    max_chars = 5000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Explanation truncated]"

    return text.strip()


def validate_model(model: Model) -> list[str]:
    """Validate an LLM-generated model for correctness issues.

    This runs the standard ``model.validate()`` plus additional checks
    specific to LLM-generated models.

    Parameters
    ----------
    model : Model
        The model to validate.

    Returns
    -------
    list of str
        Warning messages (empty list if no issues found).
    """
    warnings = []

    # Run standard validation
    try:
        model.validate()
    except ValueError as e:
        warnings.append(f"Validation error: {e}")

    # Check for suspiciously large bounds
    for var in model._variables:
        import numpy as np

        if np.any(np.abs(var.lb) > 1e15) or np.any(np.abs(var.ub) > 1e15):
            warnings.append(
                f"Variable '{var.name}' has very large bounds — "
                f"consider tightening for numerical stability"
            )

    # Check for missing constraints (model with variables but no constraints)
    if model._variables and not model._constraints:
        warnings.append("Model has variables but no constraints — this may be a formulation error")

    # Check for unused variables (not in objective or constraints)
    # This is a heuristic — can't fully detect without expression analysis
    if model.num_variables > 50:
        warnings.append(
            f"Model has {model.num_variables} variables — verify the LLM didn't over-generate"
        )

    return warnings


def sanitize_tool_args(tool_name: str, args: dict) -> dict:
    """Sanitize arguments from LLM tool calls.

    Ensures tool arguments are within expected ranges and types.

    Parameters
    ----------
    tool_name : str
        Name of the tool being called.
    args : dict
        Raw arguments from the LLM.

    Returns
    -------
    dict
        Sanitized arguments.

    Raises
    ------
    ValueError
        If arguments are clearly invalid.
    """
    sanitized = dict(args)

    if tool_name == "add_continuous":
        # Clamp bounds to reasonable range
        if "lb" in sanitized:
            sanitized["lb"] = max(float(sanitized["lb"]), -1e20)
        if "ub" in sanitized:
            sanitized["ub"] = min(float(sanitized["ub"]), 1e20)
        # Validate shape
        if "shape" in sanitized:
            shape = sanitized["shape"]
            if isinstance(shape, (list, tuple)):
                for dim in shape:
                    if not isinstance(dim, int) or dim < 1 or dim > 10000:
                        raise ValueError(f"Invalid dimension {dim} in shape {shape}")

    elif tool_name == "add_integer":
        if "lb" in sanitized:
            sanitized["lb"] = int(sanitized["lb"])
        if "ub" in sanitized:
            sanitized["ub"] = int(sanitized["ub"])

    elif tool_name == "set_objective":
        if sanitized.get("sense") not in ("minimize", "maximize"):
            raise ValueError(f"Invalid objective sense: {sanitized.get('sense')}")

    # Sanitize all string fields (prevent injection of code-like content)
    for key, value in sanitized.items():
        if isinstance(value, str) and key == "name":
            # Variable/constraint names: alphanumeric + underscore only
            sanitized[key] = re.sub(r"[^a-zA-Z0-9_]", "_", value)

    return sanitized
