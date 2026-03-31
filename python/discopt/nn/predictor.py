"""Convenience dispatcher for embedding trained ML models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from discopt.nn.formulations.base import NNFormulation, TreeFormulation
from discopt.nn.network import Activation, NetworkDefinition
from discopt.nn.tree import TreeEnsembleDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable


def add_predictor(
    model: Model,
    inputs: Variable,
    predictor: object,
    *,
    method: str = "auto",
    prefix: str = "pred",
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Variable, Union[NNFormulation, TreeFormulation]]:
    """Embed a trained ML model as constraints in a discopt Model.

    Auto-detects the predictor type and converts it to the appropriate
    internal representation, then formulates it as optimization constraints.

    Parameters
    ----------
    model : discopt.Model
        The optimization model.
    inputs : Variable
        Existing input variables to connect the predictor to.
    predictor : object
        A trained ML model. Supported types:

        - :class:`NetworkDefinition` or :class:`TreeEnsembleDefinition`
        - ``sklearn.neural_network.MLPRegressor`` / ``MLPClassifier``
        - ``sklearn.tree.DecisionTreeRegressor`` / ``DecisionTreeClassifier``
        - ``sklearn.ensemble.GradientBoostingRegressor`` / ``RandomForestRegressor``
        - ``torch.nn.Sequential``
        - A file path (str or Path) to an ONNX model
    method : str
        Formulation method. ``"auto"`` selects based on the model type.
        For neural networks: ``"relu_bigm"``, ``"full_space"``, or
        ``"reduced_space"``. Tree ensembles always use MILP encoding.
    prefix : str
        Name prefix for created variables and constraints.
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on features. Required for big-M and
        tree formulations. Overrides bounds already set on the predictor.

    Returns
    -------
    outputs : Variable
        Output variable(s) of the embedded predictor.
    formulation : NNFormulation or TreeFormulation
        The formulation object (for inspection or further use).
    """
    definition = _convert(predictor, input_bounds)

    if isinstance(definition, NetworkDefinition):
        if input_bounds is not None:
            definition = NetworkDefinition(
                definition.layers,
                input_bounds=input_bounds,
            )
        if method == "auto":
            has_relu = any(layer.activation == Activation.RELU for layer in definition.layers)
            method = "relu_bigm" if has_relu else "full_space"

        form = NNFormulation(model, definition, strategy=method, prefix=prefix)
        form.formulate()
        _link_inputs(model, inputs, form.inputs, prefix)
        return form.outputs, form

    if isinstance(definition, TreeEnsembleDefinition):
        if input_bounds is not None:
            definition = TreeEnsembleDefinition(
                trees=definition.trees,
                n_features=definition.n_features,
                base_score=definition.base_score,
                input_bounds=input_bounds,
            )
        form_t = TreeFormulation(model, definition, prefix=prefix)
        form_t.formulate()
        _link_inputs(model, inputs, form_t.inputs, prefix)
        return form_t.outputs, form_t

    raise TypeError(f"Cannot embed predictor of type {type(predictor).__name__}")


def _link_inputs(model: Model, user_inputs: Variable, form_inputs: Variable, prefix: str) -> None:
    """Add equality constraints linking user's input variables to formulation inputs."""
    n = form_inputs.shape[0] if form_inputs.shape else 1
    constraints = []
    for j in range(n):
        constraints.append(user_inputs[j] == form_inputs[j])
    model.subject_to(constraints, name=f"{prefix}_link_inputs")


def _convert(
    predictor: object,
    input_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> NetworkDefinition | TreeEnsembleDefinition:
    """Convert a predictor object to a discopt definition."""
    if isinstance(predictor, (NetworkDefinition, TreeEnsembleDefinition)):
        return predictor

    if isinstance(predictor, (str, Path)):
        p = Path(predictor)
        if p.suffix in (".onnx",) and p.exists():
            from discopt.nn.readers.onnx_reader import load_onnx

            return load_onnx(str(predictor), input_bounds=input_bounds)
        raise TypeError(f"File not found or unsupported format: {predictor}")

    mod = type(predictor).__module__ or ""

    # sklearn MLP
    if hasattr(predictor, "coefs_") and hasattr(predictor, "intercepts_"):
        from discopt.nn.readers.sklearn_reader import load_sklearn_mlp

        return load_sklearn_mlp(predictor, input_bounds=input_bounds)

    # sklearn single tree
    if hasattr(predictor, "tree_") and not hasattr(predictor, "estimators_"):
        from discopt.nn.readers.sklearn_reader import load_sklearn_tree

        return load_sklearn_tree(predictor, input_bounds=input_bounds)

    # sklearn ensemble (GBR, RF)
    if hasattr(predictor, "estimators_") and hasattr(predictor, "n_features_in_"):
        from discopt.nn.readers.sklearn_reader import load_sklearn_ensemble

        return load_sklearn_ensemble(predictor, input_bounds=input_bounds)

    # PyTorch Sequential
    if mod.startswith("torch"):
        from discopt.nn.readers.torch_reader import load_torch_sequential

        return load_torch_sequential(predictor, input_bounds=input_bounds)

    raise TypeError(
        f"Cannot auto-detect predictor type for {type(predictor).__name__}. "
        "Pass a NetworkDefinition, TreeEnsembleDefinition, sklearn model, "
        "torch.nn.Sequential, or path to an ONNX file."
    )
