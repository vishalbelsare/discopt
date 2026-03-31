"""Import trained scikit-learn models into discopt data structures."""

from __future__ import annotations

import numpy as np

from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.tree import DecisionTree, TreeEnsembleDefinition

_SKLEARN_ACTIVATION_MAP = {
    "relu": Activation.RELU,
    "logistic": Activation.SIGMOID,
    "tanh": Activation.TANH,
    "identity": Activation.LINEAR,
}


def load_sklearn_mlp(
    model,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> NetworkDefinition:
    """Convert a trained sklearn MLP to a NetworkDefinition.

    Parameters
    ----------
    model : sklearn.neural_network.MLPRegressor or MLPClassifier
        Trained sklearn MLP model.
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on input features.

    Returns
    -------
    NetworkDefinition
    """
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise TypeError("Expected a fitted sklearn MLP with coefs_ and intercepts_")

    activation = _SKLEARN_ACTIVATION_MAP.get(model.activation)
    if activation is None:
        raise ValueError(f"Unsupported sklearn activation: {model.activation!r}")

    layers = []
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        is_last = i == len(model.coefs_) - 1
        act = Activation.LINEAR if is_last else activation
        layers.append(DenseLayer(W.astype(np.float64), b.astype(np.float64), act))

    return NetworkDefinition(layers, input_bounds=input_bounds)


def _sklearn_tree_to_decision_tree(tree, n_features: int) -> DecisionTree:
    """Convert a single sklearn tree_ object to a DecisionTree."""
    t = tree
    feature = np.array(t.feature, dtype=int)
    threshold = np.array(t.threshold, dtype=np.float64)
    left_child = np.array(t.children_left, dtype=int)
    right_child = np.array(t.children_right, dtype=int)

    # sklearn uses -2 for leaf marker; we use -1
    feature = np.where(feature == -2, -1, feature)
    left_child = np.where(left_child == -1, -1, left_child)
    right_child = np.where(right_child == -1, -1, right_child)

    # value shape: (n_nodes, n_outputs, max_n_classes) for classifiers,
    # (n_nodes, 1, 1) for regressors
    raw_value = t.value.squeeze()
    if raw_value.ndim > 1:
        raise ValueError("Multi-output trees are not supported; use single-output models")
    value = raw_value.astype(np.float64)

    return DecisionTree(
        n_features=n_features,
        feature=feature,
        threshold=threshold,
        left_child=left_child,
        right_child=right_child,
        value=value,
    )


def load_sklearn_tree(
    model,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> TreeEnsembleDefinition:
    """Convert a trained sklearn DecisionTree to a TreeEnsembleDefinition.

    Parameters
    ----------
    model : sklearn.tree.DecisionTreeRegressor or DecisionTreeClassifier
        Trained sklearn decision tree.
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on input features.

    Returns
    -------
    TreeEnsembleDefinition
    """
    if not hasattr(model, "tree_"):
        raise TypeError("Expected a fitted sklearn DecisionTree with tree_ attribute")

    tree = _sklearn_tree_to_decision_tree(model.tree_, model.n_features_in_)
    return TreeEnsembleDefinition(
        trees=[tree],
        n_features=model.n_features_in_,
        input_bounds=input_bounds,
    )


def load_sklearn_ensemble(
    model,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> TreeEnsembleDefinition:
    """Convert a trained sklearn ensemble to a TreeEnsembleDefinition.

    Supports GradientBoostingRegressor, RandomForestRegressor, and their
    classifier counterparts.

    Parameters
    ----------
    model : sklearn ensemble estimator
        Trained sklearn ensemble with ``estimators_`` attribute.
    input_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds on input features.

    Returns
    -------
    TreeEnsembleDefinition
    """
    if not hasattr(model, "estimators_"):
        raise TypeError("Expected a fitted sklearn ensemble with estimators_ attribute")

    n_features = model.n_features_in_
    trees = []
    base_score = 0.0

    # GradientBoosting: estimators_ is (n_estimators, n_outputs) array of trees
    # RandomForest: estimators_ is a flat list of trees
    estimators = model.estimators_
    if hasattr(estimators, "ravel"):
        estimators = estimators.ravel()

    learning_rate = getattr(model, "learning_rate", 1.0)
    if hasattr(model, "init_"):
        # GradientBoosting has an initial estimator (usually mean)
        init = model.init_
        if hasattr(init, "constant_"):
            c = init.constant_
            base_score = float(c.ravel()[0]) if hasattr(c, "ravel") else float(c)

    for estimator in estimators:
        dt = _sklearn_tree_to_decision_tree(estimator.tree_, n_features)
        if learning_rate != 1.0:
            dt.value = dt.value * learning_rate
        trees.append(dt)

    # RandomForest averages; GradientBoosting sums
    is_rf = type(model).__name__.startswith("RandomForest")
    if is_rf and len(trees) > 0:
        for dt in trees:
            dt.value = dt.value / len(trees)

    return TreeEnsembleDefinition(
        trees=trees,
        n_features=n_features,
        base_score=base_score,
        input_bounds=input_bounds,
    )
