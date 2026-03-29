"""Load ONNX models into NetworkDefinition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from discopt.nn.network import Activation, DenseLayer, NetworkDefinition

_ONNX_ACTIVATION_MAP = {
    "Relu": Activation.RELU,
    "Sigmoid": Activation.SIGMOID,
    "Tanh": Activation.TANH,
    "Softplus": Activation.SOFTPLUS,
}


def load_onnx(
    path: str | Path,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> NetworkDefinition:
    """Load an ONNX model into a :class:`NetworkDefinition`.

    Supports sequential feedforward networks with dense layers (Gemm/MatMul)
    and standard activations (ReLU, Sigmoid, Tanh, Softplus).

    Parameters
    ----------
    path : str or Path
        Path to the ``.onnx`` file.
    input_bounds : tuple of (np.ndarray, np.ndarray) or None
        ``(lb, ub)`` bounds on each input feature.

    Returns
    -------
    NetworkDefinition

    Raises
    ------
    ImportError
        If ``onnx`` is not installed.
    ValueError
        If the model contains unsupported operations or is not sequential.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as e:
        raise ImportError(
            "onnx is required to load ONNX models. Install with: pip install discopt[nn]"
        ) from e

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph

    # Build initializer lookup: name -> numpy array
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Parse graph nodes into layers
    layers: list[DenseLayer] = []
    nodes = list(graph.node)
    i = 0

    while i < len(nodes):
        node = nodes[i]

        if node.op_type in ("Gemm", "MatMul"):
            weights, biases = _extract_gemm(node, initializers, nodes, i)
            activation, i = _extract_activation(nodes, i + 1)

            # Handle bias from a separate Add node
            if biases is None:
                if i < len(nodes) and nodes[i].op_type == "Add":
                    biases = _extract_add_bias(nodes[i], initializers)
                    i += 1
                    # Check for activation after the Add
                    activation, i = _extract_activation(nodes, i)

            if biases is None:
                biases = np.zeros(weights.shape[1], dtype=np.float64)

            layers.append(DenseLayer(weights, biases, activation))

        elif node.op_type in ("Flatten", "Reshape", "Identity", "Dropout"):
            i += 1  # Skip structural ops
        else:
            raise ValueError(
                f"Unsupported ONNX operation: {node.op_type}. "
                f"Only sequential dense networks are supported."
            )

    if not layers:
        raise ValueError("No dense layers found in ONNX model")

    return NetworkDefinition(layers, input_bounds=input_bounds)


def _extract_gemm(
    node, initializers: dict, nodes: list, idx: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract weights and biases from a Gemm or MatMul node."""
    if node.op_type == "Gemm":
        # Gemm: Y = alpha * A @ B + beta * C
        # Weights can be input[1], biases input[2]
        trans_b = False
        for attr in node.attribute:
            if attr.name == "transB":
                trans_b = bool(attr.i)

        w_name = node.input[1]
        W = np.asarray(initializers[w_name], dtype=np.float64)
        if trans_b:
            W = W.T

        biases = None
        if len(node.input) >= 3 and node.input[2] in initializers:
            biases = np.asarray(initializers[node.input[2]], dtype=np.float64)

        return W, biases

    else:  # MatMul
        # One of the inputs should be in initializers (the weight matrix)
        for inp in node.input:
            if inp in initializers:
                W = np.asarray(initializers[inp], dtype=np.float64)
                return W, None
        raise ValueError(f"MatMul node has no weight initializer: {node.input}")


def _extract_add_bias(node, initializers: dict) -> np.ndarray | None:
    """Extract bias from an Add node."""
    for inp in node.input:
        if inp in initializers:
            return np.asarray(initializers[inp], dtype=np.float64)
    return None


def _extract_activation(nodes: list, idx: int) -> tuple[Activation, int]:
    """Check if the next node is an activation function."""
    if idx < len(nodes) and nodes[idx].op_type in _ONNX_ACTIVATION_MAP:
        return _ONNX_ACTIVATION_MAP[nodes[idx].op_type], idx + 1
    return Activation.LINEAR, idx
