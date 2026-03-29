"""Neural network embedding for discopt optimization models.

Embed trained neural networks as algebraic constraints in MINLP models,
enabling optimization over ML surrogates with global optimality guarantees.

Example
-------
>>> import discopt.modeling as dm
>>> from discopt.nn import NNFormulation, NetworkDefinition, DenseLayer, Activation
>>>
>>> m = dm.Model("nn_opt")
>>> net = NetworkDefinition([
...     DenseLayer(W1, b1, Activation.RELU),
...     DenseLayer(W2, b2, Activation.LINEAR),
... ], input_bounds=(lb, ub))
>>>
>>> nn = NNFormulation(m, net, strategy="relu_bigm")
>>> nn.formulate()
>>> m.minimize(dm.sum(nn.outputs))
>>> m.subject_to(nn.inputs[0] >= 1.0)
>>> result = m.solve()
"""

from discopt.nn.bounds import LayerBounds, propagate_bounds
from discopt.nn.formulations.base import NNFormulation
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.scaling import OffsetScaling

__all__ = [
    "Activation",
    "DenseLayer",
    "LayerBounds",
    "NNFormulation",
    "NetworkDefinition",
    "OffsetScaling",
    "propagate_bounds",
]


# Lazy import for optional ONNX dependency
def load_onnx(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Load an ONNX model into a NetworkDefinition. Requires ``pip install discopt[nn]``."""
    from discopt.nn.readers.onnx_reader import load_onnx as _load_onnx

    return _load_onnx(*args, **kwargs)
