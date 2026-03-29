"""Interval arithmetic bound propagation through neural networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.nn.network import Activation, NetworkDefinition


@dataclass
class LayerBounds:
    """Pre- and post-activation bounds for a single layer."""

    pre_lb: np.ndarray
    pre_ub: np.ndarray
    post_lb: np.ndarray
    post_ub: np.ndarray


def _apply_activation_bounds(
    activation: Activation, pre_lb: np.ndarray, pre_ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply activation function to pre-activation bounds (monotone functions)."""
    if activation == Activation.LINEAR:
        return pre_lb.copy(), pre_ub.copy()
    elif activation == Activation.RELU:
        return np.maximum(pre_lb, 0.0), np.maximum(pre_ub, 0.0)
    elif activation == Activation.SIGMOID:
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return sig(pre_lb), sig(pre_ub)
    elif activation == Activation.TANH:
        return np.tanh(pre_lb), np.tanh(pre_ub)
    elif activation == Activation.SOFTPLUS:
        sp = lambda z: np.logaddexp(z, 0)  # noqa: E731
        return sp(pre_lb), sp(pre_ub)
    else:
        raise ValueError(f"Unknown activation: {activation}")


def propagate_bounds(network: NetworkDefinition) -> list[LayerBounds]:
    """Propagate input bounds through the network via interval arithmetic.

    Uses natural interval extension: split weights into positive and negative
    parts to compute tight affine bounds, then apply monotone activation bounds.

    Parameters
    ----------
    network : NetworkDefinition
        Network with ``input_bounds`` set.

    Returns
    -------
    list of LayerBounds
        Bounds for each layer (pre- and post-activation).

    Raises
    ------
    ValueError
        If ``network.input_bounds`` is None.
    """
    if network.input_bounds is None:
        raise ValueError("input_bounds must be set for bound propagation")

    lb, ub = network.input_bounds
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)

    result: list[LayerBounds] = []

    for layer in network.layers:
        W = layer.weights
        b = layer.biases

        # Natural interval extension for affine: zhat = W^T @ x + b
        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        pre_lb = W_pos.T @ lb + W_neg.T @ ub + b
        pre_ub = W_pos.T @ ub + W_neg.T @ lb + b

        post_lb, post_ub = _apply_activation_bounds(layer.activation, pre_lb, pre_ub)

        result.append(LayerBounds(pre_lb, pre_ub, post_lb, post_ub))

        # Post-activation bounds become input bounds for the next layer
        lb, ub = post_lb, post_ub

    return result
