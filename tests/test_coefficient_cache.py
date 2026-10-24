"""CD per-sweep shortcuts: reused coefficients equal recomputed ones and follow every change."""
from __future__ import annotations

import pytest
import torch

from model.function.interaction import (
    DoubleExponentialNonLinearInteraction,
    ResistiveBiasInteraction,
    SumSeparableFunction,
    ZeroCoefficient,
)
from model.resistive.interaction import DenseResistive
from model.resistive.layer import NonlinearResistiveLayer, ResistiveInputLayer
from model.variable.parameter import Bias, DenseWeight

PARAMS = {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.8}
VOLTAGE_AMP, CURRENT_AMP = 4.0, 1.0


def _stack():
    generator = torch.Generator().manual_seed(0)
    inputs = ResistiveInputLayer((6,), gain=2.0)
    hidden = NonlinearResistiveLayer((4,), non_linearity="double_diode_exponential")
    output = NonlinearResistiveLayer((2,), non_linearity="double_diode_exponential")
    for depth, layer in enumerate((inputs, hidden, output)):
        layer._depth_index = depth
    w1 = DenseWeight((6,), (4,), gain=1.0, device="cpu")
    w2 = DenseWeight((4,), (2,), gain=1.0, device="cpu")
    w1.state = torch.rand(6, 4, generator=generator, dtype=torch.float64)
    w2.state = torch.rand(4, 2, generator=generator, dtype=torch.float64)
    bias = Bias((4,), 0.0, "cpu")
    bias.state = torch.rand(4, generator=generator, dtype=torch.float64)
    first = DenseResistive(inputs, hidden, w1, VOLTAGE_AMP, CURRENT_AMP)
    second = DenseResistive(hidden, output, w2, VOLTAGE_AMP, CURRENT_AMP)
    interactions = [first, second, ResistiveBiasInteraction(hidden, bias, scale=0.5),
                    DoubleExponentialNonLinearInteraction(hidden, PARAMS, VOLTAGE_AMP, CURRENT_AMP)]
    energy = SumSeparableFunction([inputs, hidden, output], [w1, w2, bias], interactions)
    inputs.set_input(torch.rand(1, 3, generator=generator, dtype=torch.float64))
    hidden.state = torch.rand(1, 4, generator=generator, dtype=torch.float64)
    output.state = torch.rand(1, 2, generator=generator, dtype=torch.float64)
    return energy, first, second, (inputs, hidden, output), (w1, w2, bias)


def _expected_hidden_a(w1, w2, bias):
    return (0.5 * w1.state.sum(dim=0) + 0.5 * w2.state.sum(dim=1) * VOLTAGE_AMP * CURRENT_AMP
            + bias.state * 0.5).unsqueeze(0)


def test_a_is_reused_until_the_weights_change():
    energy, first, _, (_, hidden, _), (w1, w2, bias) = _stack()
    a_fn = energy.a_coef_fn(hidden)
    torch.testing.assert_close(a_fn(), _expected_hidden_a(w1, w2, bias), rtol=0, atol=1e-15)
    assert first._a_coef_layer_post() is first._a_coef_layer_post()
    w1.state.mul_(2.0)  # in place: the version counter invalidates the cached value
    torch.testing.assert_close(a_fn(), _expected_hidden_a(w1, w2, bias), rtol=0, atol=1e-15)
    w2.state = w2.state + 1.0  # replaced tensor
    bias.state = bias.state * 3.0
    torch.testing.assert_close(a_fn(), _expected_hidden_a(w1, w2, bias), rtol=0, atol=1e-15)


def test_nothing_is_reused_while_autograd_records_through_the_weights():
    _, first, _, _, (w1, _, _) = _stack()
    w1.state.requires_grad_(True)
    a = first._a_coef_layer_post()
    assert a is not first._a_coef_layer_post()
    a.sum().backward()
    torch.testing.assert_close(w1.state.grad, torch.full_like(w1.state, 0.5))
    with torch.no_grad():
        assert first._a_coef_layer_post() is first._a_coef_layer_post()


def test_inference_tensor_weights_are_not_cached():
    _, first, _, _, (w1, _, _) = _stack()
    with torch.inference_mode():
        w1.state = w1.state.clone()
        assert first._a_coef_layer_post() is not first._a_coef_layer_post()


def test_input_term_follows_the_input_values():
    _, first, _, (inputs, _, _), (w1, _, _) = _stack()
    b = first._b_coef_layer_post()
    torch.testing.assert_close(b, -inputs.state @ w1.state, rtol=0, atol=0)
    assert first._b_coef_layer_post() is b  # unchanged input: reused
    inputs._state = inputs.state.clone()  # a new input tensor is a new input
    rebuilt = first._b_coef_layer_post()
    assert rebuilt is not b
    torch.testing.assert_close(rebuilt, b, rtol=0, atol=0)
    inputs.state.mul_(2.0)  # in place: the version counter invalidates the term
    torch.testing.assert_close(first._b_coef_layer_post(), -inputs.state @ w1.state, rtol=0, atol=0)
    with torch.inference_mode():  # benchmark inputs are inference tensors
        inputs.set_input(torch.full((1, 3), 0.25, dtype=torch.float64))
        torch.testing.assert_close(first._b_coef_layer_post(), -inputs.state @ w1.state, rtol=0, atol=0)
    w1.state.add_(1.0)
    torch.testing.assert_close(first._b_coef_layer_post(), -inputs.state @ w1.state, rtol=0, atol=0)


def test_zero_coefficients_are_skipped_but_remain_callable():
    energy, first, second, (_, hidden, _), _ = _stack()
    zero = DoubleExponentialNonLinearInteraction(hidden, PARAMS, VOLTAGE_AMP, CURRENT_AMP).b_coef_fn(hidden)
    assert isinstance(zero, ZeroCoefficient)
    torch.testing.assert_close(zero(), torch.zeros_like(hidden.state), rtol=0, atol=0)
    expected = first._b_coef_layer_post() + second._b_coef_layer_pre()
    torch.testing.assert_close(energy.b_coef_fn(hidden)(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("non_linearity", ["double_diode_exponential", "single_diode_exponential", "linear"])
def test_activate_is_the_identity_without_clipping(non_linearity):
    layer = NonlinearResistiveLayer((4,), non_linearity=non_linearity)
    layer.state = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])
    assert layer.activate() is layer.state


def test_perfect_diode_still_clamps_each_half():
    layer = NonlinearResistiveLayer((4,), non_linearity="perfect_diode")
    layer.state = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])
    torch.testing.assert_close(layer.activate(), torch.tensor([[0.0, 2.0, -3.0, 0.0]]))
