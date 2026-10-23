"""Adaptive relaxation of the CD minimizer: estimator against SOR theory, and the opt-in API."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.minimizer import _EquilibriumMinimizer
from repro.minimizer_factory import build_minimizer


def sor_updates(mu, omega, sweeps, modes=(1.0,)):
    """Updates of red-black SOR on the 2-cyclic system [[I, -B], [-B^T, I]], B = mu * diag(modes)."""
    b = mu * torch.tensor(modes, dtype=torch.float64)
    x1 = torch.ones_like(b)
    x2 = torch.linspace(0.3, 1.0, len(modes), dtype=torch.float64)
    previous, updates = torch.cat((x1, x2)), []
    for _ in range(sweeps):
        x1 = (1 - omega) * x1 + omega * b * x2
        x2 = (1 - omega) * x2 + omega * b * x1
        state = torch.cat((x1, x2))
        updates.append(state - previous)
        previous = state
    return updates


def estimate(updates, omega):
    probe = SimpleNamespace(_updates=updates[-3:], _omega=omega)
    return _EquilibriumMinimizer._two_term_mu2(probe)


@pytest.mark.parametrize("omega", [1.0, 1.2, 1.6, 1.85])
def test_two_term_estimate_recovers_mu_on_both_sides_of_the_optimum(omega):
    # mu = 0.9 puts omega_opt at 1.393: real dominant eigenvalue below, complex pair above.
    mu2 = estimate(sor_updates(0.9, omega, 12), omega)
    assert mu2 == pytest.approx(0.81, rel=1e-9)


def test_two_term_estimate_tracks_the_dominant_mode_when_under_relaxed():
    mu2 = estimate(sor_updates(0.95, 1.0, 60, modes=(1.0, 0.6, 0.3)), 1.0)
    assert mu2 == pytest.approx(0.95**2, rel=1e-3)
    omega = 2 / (1 + math.sqrt(1 - mu2))
    assert omega == pytest.approx(2 / (1 + math.sqrt(1 - 0.95**2)), rel=1e-3)


def _minimizer(relaxation, method="voltage_change"):
    a = torch.full((1, 2), 10., dtype=torch.float64)
    b = torch.tensor([[-4000., 4000.]], dtype=torch.float64)
    layer = NonlinearResistiveLayer((2,), non_linearity="single_diode_exponential")
    layer.state = torch.zeros_like(b)

    class Quadratic:
        def a_coef_fn(self, _):
            return lambda: a

        def b_coef_fn(self, _):
            return lambda: b

        def grad_layer_fn(self, _):
            return lambda: 2 * a * layer.state + b

        def layers(self):
            return [layer]

        def params(self):
            return []

    simulation = dict(nonlinearity="single_diode_exponential",
                      physical=dict(saturation_current=1e-6, thermal_voltage=.05, offset_voltage=0.),
                      amplification=dict(voltage_factor=1., current_factor=1.),
                      updater=dict(method="lambert_w_v1", dtype="float64", backend="torchlambertw",
                                   relaxation=relaxation, linear_coefficient_clamp=1e6,
                                   quadratic_coefficient_min=1e-30, exponent_clip=100.,
                                   asymptotic_threshold=1e10, asymptotic_terms=4, polish=False))
    equilibrium = dict(method=method, max_sweeps=1, relative_tolerance=1e-5, absolute_tolerance=1e-6,
                       initial_state="zeros", update_order="asynchronous")
    return build_minimizer(fn=Quadratic(), free_layers=[layer], simulation=simulation,
                           equilibrium=equilibrium, repo_root=Path.cwd())


def test_adaptive_relaxation_requires_relaxing_updaters():
    with pytest.raises(ValueError, match="over-relaxed updaters"):
        _minimizer(1.0).set_adaptive_relaxation()


def test_plain_call_resets_and_resume_keeps_the_factor():
    minimizer = _minimizer(1.2)
    assert minimizer.relaxation_factor is None
    minimizer.set_adaptive_relaxation(initial=1.0)
    assert minimizer.relaxation_factor == 1.0
    assert all(u._overrelaxation_factor == 1.0 for u in minimizer._updaters)
    minimizer._set_relaxation_factor(1.5)
    minimizer.compute_equilibrium(resume=True)
    assert minimizer.relaxation_factor == 1.5
    minimizer.compute_equilibrium()
    assert minimizer.relaxation_factor == 1.0


@pytest.mark.parametrize("initial, maximum", [(0.9, 1.95), (1.5, 1.4), (1.0, 2.0)])
def test_invalid_relaxation_bounds_are_rejected(initial, maximum):
    with pytest.raises(ValueError, match="initial"):
        _minimizer(1.2).set_adaptive_relaxation(initial=initial, maximum=maximum)
