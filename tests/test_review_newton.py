"""Numerical regression checks ported from the review solver tests."""
import numpy as np


import pytest


from scipy.optimize import brentq


from repro.solvers.solver import DoubleShockley, LayeredDRN, coordinate_descent, newton


def make_circuit(seed=12, widths=(4, 5, 3, 2), av=4.0, ai=1.0, law=None):
    rng = np.random.default_rng(seed)
    weights = [rng.uniform(0.02, 0.2, (a, b)) for a, b in zip(widths[:-1], widths[1:])]
    return LayeredDRN(weights, law or DoubleShockley(1e-3, 0.1, 0.3), av, ai)


@pytest.mark.parametrize("av,ai", [(1.0, 1.0), (4.0, 1.0), (0.7, 2.0)])
def test_analytic_jacobian_matches_central_differences(av, ai):
    circuit = make_circuit(av=av, ai=ai)
    rng = np.random.default_rng(501)
    v = rng.uniform(-0.7, 0.7, circuit.size)
    forcing = circuit.forcing(rng.normal(size=4))
    step = 2e-6
    difference = np.column_stack(
        [
            (
                circuit.residual(v + step * e, forcing)
                - circuit.residual(v - step * e, forcing)
            )
            / (2 * step)
            for e in np.eye(circuit.size)
        ]
    )
    np.testing.assert_allclose(circuit.jacobian(v), difference, rtol=2e-8, atol=3e-10)


@pytest.mark.parametrize("widths", [(4, 2), (4, 5, 2), (4, 5, 3, 2), (4, 2, 3, 4, 2)])
@pytest.mark.parametrize("av,ai", [(1.0, 1.0), (4.0, 1.0), (0.7, 2.0)])
def test_block_cholesky_matches_full_nonsymmetric_jacobian(widths, av, ai):
    circuit = make_circuit(widths=widths, av=av, ai=ai)
    rng = np.random.default_rng(11)
    v, rhs = rng.uniform(-0.6, 0.6, circuit.size), rng.normal(size=circuit.size)
    _, derivative = circuit.residual_derivative(v, rhs)
    step = circuit.linear_step(derivative, rhs)
    np.testing.assert_allclose(circuit.jacobian(v) @ step, rhs, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        step, circuit.linear_step(derivative, rhs, "dense_lu"), rtol=2e-12, atol=2e-12
    )


def test_scalar_full_law_against_independent_bracketed_roots():
    law = DoubleShockley(1e-5, 0.07, 0.6)
    diagonal = np.geomspace(1e-4, 1e2, 17)
    forcing = np.linspace(-12, 12, 17)
    expected = [
        brentq(
            lambda v: d * v
            + 2
            * law.saturation_current
            * np.exp(-law.offset_voltage / law.thermal_voltage)
            * np.sinh(v / law.thermal_voltage)
            - b,
            -30,
            30,
            xtol=1e-14,
        )
        for d, b in zip(diagonal, forcing)
    ]
    np.testing.assert_allclose(
        law.coordinate_root(diagonal, forcing), expected, rtol=1e-13, atol=1e-13
    )


def test_linear_network_is_solved_in_one_newton_step_from_zero():
    circuit = make_circuit(law=DoubleShockley(0, 0.1, 0.3))
    forcing = circuit.forcing(np.array([-2.0, 3.0, -1.0, 0.5]))
    result = newton(circuit, forcing)
    assert result.converged and result.iterations == 1
    np.testing.assert_allclose(
        result.voltages,
        np.linalg.solve(circuit.jacobian(np.zeros(circuit.size)), forcing),
        rtol=1e-13,
        atol=1e-13,
    )


def test_voltage_change_stop_needs_a_confirming_step_and_matches_current_stop():
    circuit = make_circuit(law=DoubleShockley(0, 0.1, 0.3))
    forcing = circuit.forcing(np.array([-2.0, 3.0, -1.0, 0.5]))
    # From zeros the first step changes V by more than atol, so linear networks take two steps.
    linear = newton(circuit, forcing, voltage_rtol=1e-5, voltage_atol=1e-6)
    assert linear.converged and linear.iterations == 2
    circuit = make_circuit()
    forcing = circuit.forcing(np.array([-7.0, 3.0, -2.0, 0.5]))
    by_voltage = newton(circuit, forcing, voltage_rtol=1e-5, voltage_atol=1e-6)
    by_current = newton(circuit, forcing, tolerance=1e-12, residual_kind="current")
    assert by_voltage.converged and by_voltage.iterations >= 2
    np.testing.assert_allclose(by_voltage.voltages, by_current.voltages, atol=1e-6)
    with pytest.raises(ValueError):
        newton(circuit, forcing, voltage_rtol=0.0, voltage_atol=0.0)


def test_global_newton_and_cd_reach_same_equilibrium_from_zero():
    circuit = make_circuit()
    forcing = circuit.forcing(np.array([-7.0, 3.0, -2.0, 0.5]))
    nr = newton(circuit, forcing, tolerance=1e-12)
    cd = coordinate_descent(circuit, forcing, tolerance=1e-12, relaxation=1.2)
    assert nr.converged and cd.converged
    np.testing.assert_allclose(nr.voltages, cd.voltages, atol=2e-11, rtol=0)
    assert nr.backtracks > 0


@pytest.mark.parametrize("inputs", [[-7.0, 3.0, -2.0, 0.5], [1e4, 2e4, 1e4, 2e4]])
def test_dense_newton_preserves_equilibrium_and_damping(inputs):
    circuit = make_circuit()
    force = circuit.forcing(np.array(inputs))
    dense = newton(
        circuit, force, tolerance=1e-8, residual_kind="current", backend="dense_lu"
    )
    block = newton(circuit, force, tolerance=1e-8, residual_kind="current")
    assert dense.converged and block.converged
    assert dense.iterations == block.iterations and dense.backtracks == block.backtracks
    assert dense.backtracks > 0
    np.testing.assert_allclose(dense.voltages, block.voltages, atol=1e-10, rtol=1e-11)


def test_damping_rejects_overflowing_trial_without_clipping_current():
    circuit = make_circuit(law=DoubleShockley(1e-6, 0.05, 0.8))
    forcing = circuit.forcing(np.array([1e4, 2e4, 1e4, 2e4]))
    result = newton(circuit, forcing, tolerance=1e-9)
    assert result.converged and result.backtracks > 0
    assert np.isfinite(result.voltages).all()


def test_nonzero_bias_current_and_arbitrary_input_are_preserved():
    circuit = make_circuit()
    biased = LayeredDRN(
        circuit.weights,
        circuit.law,
        circuit.av,
        circuit.ai,
        [np.full(w, 0.03) for w in circuit.widths],
    )
    x = np.array([0.1, -0.2, 0.4, -0.8])
    np.testing.assert_allclose(biased.forcing(x) - circuit.forcing(x), 0.03)
    result = newton(biased, biased.forcing(x), tolerance=1e-12)
    assert result.converged
    assert (
        biased.residual_voltage_norm(
            biased.residual(result.voltages, biased.forcing(x))
        )
        <= 1e-12
    )


def test_iteration_exhaustion_is_explicit_and_never_success():
    circuit = make_circuit()
    result = newton(circuit, circuit.forcing(np.ones(4)), max_iterations=0)
    assert not result.converged and result.reason == "maximum_iterations"


@pytest.mark.parametrize(
    "method,budget", [(newton, "max_iterations"), (coordinate_descent, "max_sweeps")]
)
@pytest.mark.parametrize(
    "conductance,current,passes_current,passes_voltage",
    [(0.1, 2e-6, True, False), (10.0, 2e-5, False, True)],
)
def test_current_tolerance_is_in_amperes_without_conductance_scaling(
    method, budget, conductance, current, passes_current, passes_voltage
):
    circuit = LayeredDRN([np.array([[conductance]])], DoubleShockley())
    for kind, passes in (("current", passes_current), ("voltage", passes_voltage)):
        result = method(
            circuit,
            np.array([current]),
            tolerance=1e-5,
            residual_kind=kind,
            **{budget: 0},
        )
        assert result.converged == passes
        assert result.max_current_residual == current
        assert result.residual_voltage == current / conductance


@pytest.mark.parametrize(
    "weights",
    [[np.zeros((2, 3))], [np.ones((2, 3)), np.ones((2, 4))], [np.full((2, 3), -1.0)]],
)
def test_invalid_circuits_are_rejected(weights):
    with pytest.raises(ValueError):
        LayeredDRN(weights, DoubleShockley())
