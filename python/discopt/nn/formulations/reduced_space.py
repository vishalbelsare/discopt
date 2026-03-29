"""Reduced-space formulation with nested expressions (no intermediate variables)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.nn.network import Activation, NetworkDefinition
from discopt.nn.scaling import OffsetScaling

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

_ACTIVATION_FN = {
    Activation.LINEAR: lambda x: x,
    Activation.SIGMOID: dm.sigmoid,
    Activation.TANH: dm.tanh,
    Activation.SOFTPLUS: dm.softplus,
}


class ReducedSpaceFormulation:
    """Reduced-space formulation with no intermediate variables.

    Builds a single nested expression representing the entire network.
    The output variable is constrained to equal this expression.

    Works well for small networks. For large networks, the deeply nested
    expressions may cause slow compilation.

    Supports all smooth activations plus ReLU (via ``dm.maximum``).
    """

    def __init__(
        self,
        model: Model,
        network: NetworkDefinition,
        prefix: str,
        scaling: OffsetScaling | None,
    ) -> None:
        self._model = model
        self._network = network
        self._prefix = prefix
        self._scaling = scaling

    def build(self) -> tuple[Variable, Variable]:
        """Add variables and constraints to the model.

        Returns (inputs, outputs) variable handles.
        """
        m = self._model
        net = self._network
        pfx = self._prefix

        # Create input variables
        if net.input_bounds is not None:
            lb, ub = net.input_bounds
            inputs = m.continuous(f"{pfx}_input", shape=(net.input_size,), lb=lb, ub=ub)
        else:
            inputs = m.continuous(f"{pfx}_input", shape=(net.input_size,))

        # Handle input scaling
        if self._scaling is not None:
            sc = self._scaling
            if net.input_bounds is not None:
                s_lb = (net.input_bounds[0] - sc.x_offset) / sc.x_factor
                s_ub = (net.input_bounds[1] - sc.x_offset) / sc.x_factor
                s_lo = np.minimum(s_lb, s_ub)
                s_hi = np.maximum(s_lb, s_ub)
                scaled_in = m.continuous(
                    f"{pfx}_scaled_input", shape=(net.input_size,), lb=s_lo, ub=s_hi
                )
            else:
                scaled_in = m.continuous(f"{pfx}_scaled_input", shape=(net.input_size,))
            for j in range(net.input_size):
                m.subject_to(
                    scaled_in[j] == (inputs[j] - sc.x_offset[j]) / sc.x_factor[j],
                    name=f"{pfx}_scale_in_{j}",
                )
            prev = scaled_in
        else:
            prev = inputs

        # Build nested expression layer by layer
        for k, layer in enumerate(net.layers):
            W = np.asarray(layer.weights, dtype=np.float64)
            b = np.asarray(layer.biases, dtype=np.float64)
            n_out = layer.n_outputs

            # Compute affine + activation as expressions (no new variables)
            new_exprs = []
            for j in range(n_out):
                # zhat_j = sum(W[i,j] * prev[i]) + b[j]
                zhat_j = (
                    dm.sum(
                        lambda i, _j=j, _W=W: _W[i, _j] * prev[i],
                        over=range(layer.n_inputs),
                    )
                    + b[j]
                )

                # Apply activation
                if layer.activation == Activation.RELU:
                    new_exprs.append(dm.maximum(zhat_j, 0))
                elif layer.activation in _ACTIVATION_FN:
                    new_exprs.append(_ACTIVATION_FN[layer.activation](zhat_j))
                else:
                    raise ValueError(f"Unsupported activation: {layer.activation}")

            # Store expressions in an output variable for this layer
            z = m.continuous(
                f"{pfx}_z_{k}",
                shape=(n_out,),
            )
            for j in range(n_out):
                m.subject_to(z[j] == new_exprs[j], name=f"{pfx}_layer_{k}_{j}")

            prev = z

        # Handle output scaling
        if self._scaling is not None:
            outputs = m.continuous(f"{pfx}_output", shape=(net.output_size,))
            for j in range(net.output_size):
                m.subject_to(
                    outputs[j] == prev[j] * self._scaling.y_factor[j] + self._scaling.y_offset[j],
                    name=f"{pfx}_scale_out_{j}",
                )
        else:
            outputs = prev

        return inputs, outputs
