"""Tests for the discopt.nn neural network embedding module."""

from __future__ import annotations

import numpy as np
import pytest
from discopt.nn.bounds import propagate_bounds
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition
from discopt.nn.scaling import OffsetScaling

# ---------------------------------------------------------------------------
# NetworkDefinition tests
# ---------------------------------------------------------------------------


class TestNetworkDefinition:
    def test_basic_construction(self):
        net = NetworkDefinition([DenseLayer(np.eye(3), np.zeros(3), Activation.LINEAR)])
        assert net.input_size == 3
        assert net.output_size == 3
        assert net.n_layers == 1

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="weights must be 2-D"):
            DenseLayer(np.zeros(5), np.zeros(5), Activation.LINEAR)

    def test_layer_mismatch(self):
        with pytest.raises(ValueError, match="expects 2 inputs but layer 0 has 3"):
            NetworkDefinition(
                [
                    DenseLayer(np.ones((3, 3)), np.zeros(3), Activation.RELU),
                    DenseLayer(np.ones((2, 1)), np.zeros(1), Activation.LINEAR),
                ]
            )

    def test_forward_linear(self):
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([0.5, -0.5])
        net = NetworkDefinition([DenseLayer(W, b, Activation.LINEAR)])
        x = np.array([1.0, 1.0])
        result = net.forward(x)
        expected = x @ W + b
        np.testing.assert_allclose(result, expected)

    def test_forward_relu(self):
        W = np.array([[1.0], [-1.0]])
        b = np.array([-0.5])
        net = NetworkDefinition([DenseLayer(W, b, Activation.RELU)])
        # x @ W + b = [1, -1] @ [1; -1] + (-0.5) = 1 + 1 - 0.5 = 1.5
        assert net.forward(np.array([1.0, -1.0]))[0] == pytest.approx(1.5)
        # x @ W + b = [-1, 1] @ [1; -1] + (-0.5) = -1 - 1 - 0.5 = -2.5 -> relu -> 0
        assert net.forward(np.array([-1.0, 1.0]))[0] == pytest.approx(0.0)

    def test_forward_sigmoid(self):
        W = np.array([[1.0]])
        b = np.array([0.0])
        net = NetworkDefinition([DenseLayer(W, b, Activation.SIGMOID)])
        result = net.forward(np.array([0.0]))
        assert result[0] == pytest.approx(0.5)

    def test_forward_two_layers(self):
        net = NetworkDefinition(
            [
                DenseLayer(np.array([[1.0, -1.0]]), np.array([0.0, 0.0]), Activation.RELU),
                DenseLayer(np.array([[1.0], [1.0]]), np.array([0.0]), Activation.LINEAR),
            ]
        )
        # x=2: layer1 = relu([2, -2]) = [2, 0], layer2 = [2, 0] @ [[1],[1]] = 2
        assert net.forward(np.array([2.0]))[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Bound propagation tests
# ---------------------------------------------------------------------------


class TestBoundPropagation:
    def test_linear_layer(self):
        W = np.array([[1.0, -1.0], [0.5, 2.0]])
        b = np.array([1.0, 0.0])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.LINEAR)],
            input_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )
        bounds = propagate_bounds(net)
        # zhat_0 = W[:,0]^T @ x + b[0] = 1*x1 + 0.5*x2 + 1
        #   lb = 1*0 + 0.5*0 + 1 = 1, ub = 1*1 + 0.5*1 + 1 = 2.5
        assert bounds[0].pre_lb[0] == pytest.approx(1.0)
        assert bounds[0].pre_ub[0] == pytest.approx(2.5)
        # zhat_1 = -1*x1 + 2*x2 + 0
        #   lb = -1*1 + 2*0 = -1, ub = -1*0 + 2*1 = 2
        assert bounds[0].pre_lb[1] == pytest.approx(-1.0)
        assert bounds[0].pre_ub[1] == pytest.approx(2.0)

    def test_relu_bounds(self):
        W = np.array([[1.0]])
        b = np.array([-0.5])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.RELU)],
            input_bounds=(np.array([-1.0]), np.array([2.0])),
        )
        bounds = propagate_bounds(net)
        # pre: [-1.5, 1.5], post: relu -> [0, 1.5]
        assert bounds[0].pre_lb[0] == pytest.approx(-1.5)
        assert bounds[0].pre_ub[0] == pytest.approx(1.5)
        assert bounds[0].post_lb[0] == pytest.approx(0.0)
        assert bounds[0].post_ub[0] == pytest.approx(1.5)

    def test_requires_input_bounds(self):
        net = NetworkDefinition([DenseLayer(np.eye(2), np.zeros(2), Activation.LINEAR)])
        with pytest.raises(ValueError, match="input_bounds must be set"):
            propagate_bounds(net)


# ---------------------------------------------------------------------------
# Scaling tests
# ---------------------------------------------------------------------------


class TestScaling:
    def test_basic(self):
        s = OffsetScaling(
            x_offset=np.array([1.0]),
            x_factor=np.array([2.0]),
            y_offset=np.array([0.0]),
            y_factor=np.array([1.0]),
        )
        assert s.x_offset[0] == 1.0
        assert s.x_factor[0] == 2.0

    def test_zero_factor_rejected(self):
        with pytest.raises(ValueError, match="x_factor must not contain zeros"):
            OffsetScaling(
                x_offset=np.array([0.0]),
                x_factor=np.array([0.0]),
                y_offset=np.array([0.0]),
                y_factor=np.array([1.0]),
            )


# ---------------------------------------------------------------------------
# Formulation tests (using discopt Model)
# ---------------------------------------------------------------------------


class TestFullSpaceFormulation:
    def test_linear_network(self):
        """1-layer linear network: output should equal W @ x + b."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        W = np.array([[2.0, -1.0], [0.5, 3.0]])
        b = np.array([1.0, -1.0])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.LINEAR)],
            input_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        )

        m = dm_.Model("test_linear")
        nn = NNFormulation(m, net, strategy="full_space")
        nn.formulate()

        # Fix inputs to known values and minimize output[0]
        m.subject_to(nn.inputs[0] == 0.5)
        m.subject_to(nn.inputs[1] == 0.5)
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"

        expected = np.array([0.5, 0.5]) @ W + b
        assert result.objective == pytest.approx(expected[0], abs=1e-4)

    def test_sigmoid_network(self):
        """1-layer sigmoid network."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        W = np.array([[1.0]])
        b = np.array([0.0])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.SIGMOID)],
            input_bounds=(np.array([-2.0]), np.array([2.0])),
        )

        m = dm_.Model("test_sigmoid")
        nn = NNFormulation(m, net, strategy="full_space")
        nn.formulate()

        m.subject_to(nn.inputs[0] == 0.0)
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        # sigmoid(0) = 0.5
        assert result.objective == pytest.approx(0.5, abs=1e-4)

    def test_rejects_relu(self):
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        net = NetworkDefinition(
            [DenseLayer(np.eye(2), np.zeros(2), Activation.RELU)],
            input_bounds=(np.zeros(2), np.ones(2)),
        )
        m = dm_.Model("test")
        with pytest.raises(ValueError, match="does not support"):
            NNFormulation(m, net, strategy="full_space").formulate()


class TestReluBigMFormulation:
    def test_always_active(self):
        """When input bounds ensure zhat >= 0, no binaries needed."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        W = np.array([[1.0]])
        b = np.array([1.0])  # zhat = x + 1, with x in [0, 1] -> zhat in [1, 2]
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.RELU)],
            input_bounds=(np.array([0.0]), np.array([1.0])),
        )

        m = dm_.Model("test_active")
        nn = NNFormulation(m, net, strategy="relu_bigm")
        nn.formulate()

        m.subject_to(nn.inputs[0] == 0.5)
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        # relu(0.5 + 1) = 1.5
        assert result.objective == pytest.approx(1.5, abs=1e-4)

    def test_always_inactive(self):
        """When input bounds ensure zhat <= 0, output is 0."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        W = np.array([[1.0]])
        b = np.array([-5.0])  # zhat = x - 5, with x in [0, 1] -> zhat in [-5, -4]
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.RELU)],
            input_bounds=(np.array([0.0]), np.array([1.0])),
        )

        m = dm_.Model("test_inactive")
        nn = NNFormulation(m, net, strategy="relu_bigm")
        nn.formulate()

        m.subject_to(nn.inputs[0] == 0.5)
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        # relu(0.5 - 5) = relu(-4.5) = 0
        assert result.objective == pytest.approx(0.0, abs=1e-4)

    def test_mixed_two_layer(self):
        """Two-layer ReLU network, verify output matches numpy forward pass."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        np.random.seed(42)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(3)
        W2 = np.random.randn(3, 1)
        b2 = np.random.randn(1)

        net = NetworkDefinition(
            [
                DenseLayer(W1, b1, Activation.RELU),
                DenseLayer(W2, b2, Activation.LINEAR),
            ],
            input_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        )

        x_test = np.array([0.3, -0.7])
        expected = net.forward(x_test)[0]

        m = dm_.Model("test_2layer")
        nn = NNFormulation(m, net, strategy="relu_bigm")
        nn.formulate()

        m.subject_to(nn.inputs[0] == x_test[0])
        m.subject_to(nn.inputs[1] == x_test[1])
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=60)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(expected, abs=1e-3)

    def test_requires_input_bounds(self):
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        net = NetworkDefinition(
            [DenseLayer(np.eye(2), np.zeros(2), Activation.RELU)],
        )
        m = dm_.Model("test")
        with pytest.raises(ValueError, match="requires input_bounds"):
            NNFormulation(m, net, strategy="relu_bigm").formulate()


class TestReducedSpaceFormulation:
    def test_tanh_network(self):
        """1-layer tanh network."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        W = np.array([[1.0]])
        b = np.array([0.0])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.TANH)],
            input_bounds=(np.array([-2.0]), np.array([2.0])),
        )

        m = dm_.Model("test_tanh")
        nn = NNFormulation(m, net, strategy="reduced_space")
        nn.formulate()

        m.subject_to(nn.inputs[0] == 1.0)
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(np.tanh(1.0), abs=1e-4)


# ---------------------------------------------------------------------------
# Scaling integration test
# ---------------------------------------------------------------------------


class TestScalingIntegration:
    def test_scaled_linear(self):
        """Linear network with input/output scaling."""
        import discopt.modeling as dm_
        from discopt.nn import NNFormulation

        W = np.array([[2.0]])
        b = np.array([0.0])
        net = NetworkDefinition(
            [DenseLayer(W, b, Activation.LINEAR)],
            input_bounds=(np.array([0.0]), np.array([10.0])),
        )
        scaling = OffsetScaling(
            x_offset=np.array([5.0]),
            x_factor=np.array([5.0]),
            y_offset=np.array([100.0]),
            y_factor=np.array([10.0]),
        )

        m = dm_.Model("test_scaled")
        nn = NNFormulation(m, net, strategy="full_space", scaling=scaling)
        nn.formulate()

        # Input x=10 -> scaled = (10-5)/5 = 1.0 -> nn: 2*1+0=2 -> output: 2*10+100=120
        m.subject_to(nn.inputs[0] == 10.0)
        m.minimize(nn.outputs[0])

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(120.0, abs=1e-3)
