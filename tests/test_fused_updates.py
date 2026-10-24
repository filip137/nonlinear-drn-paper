"""Fused (branch-free, synchronization-free) Shockley updates against the torch path."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from model.resistive import minimizer as module
from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.minimizer import (
    ConfigurableExponentialSingleDiodeUpdater,
    ExponentialSingleDiodeUpdater,
    Float64ExponentialDoubleDiodeUpdater,
    MinimizerSettings,
    NonFiniteDiodeError,
    OverRelaxedFloat64DoubleDiodeUpdater,
    OverRelaxedSingleDiodeUpdater,
)
from repro.minimizer_factory import build_minimizer

PARAMS = {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.8}


class _Quadratic:
    def __init__(self, a, b, layer=None):
        self.a, self.b, self.layer = a, b, layer

    def grad_layer_fn(self, layer):
        return lambda: 2 * self.a * layer.state + self.b

    def a_coef_fn(self, layer):
        return lambda: self.a

    def b_coef_fn(self, layer):
        return lambda: self.b

    def layers(self):
        return [self.layer]

    def params(self):
        return []


def _settings(polish, *, z_thresh=1e10, terms=4):
    return MinimizerSettings(
        rel_tol=None, vn_tol=None, use_polish=polish, max_newton_iters=8 if polish else None, z_thresh=z_thresh,
        exp_clip=165.0, experimental_newton_tol=None, b_clamp=1e6, pwl_extrapolation=None,
        pwl_nonconvergence_policy=None, lambertw_backend="torchlambertw", lambertw_asymptotic_terms=terms,
        single_diode_min_a=1e-30, single_diode_polish_abs_tol=1e-10 if polish else None,
        single_diode_polish_rel_tol=1e-10 if polish else None,
        double_diode_polish_residual_tol=1e-12 if polish else None,
    )


def _case(batch=3):
    """Coefficients spanning inactive nodes, both branches and diode-free exponents up to 1e5."""
    a = torch.tensor([[1e-6], [1e-4], [1e-2]], dtype=torch.float64)[:batch]
    floor = PARAMS["I_s"] * math.exp(-PARAMS["V_off"] / PARAMS["V_t"])
    b = torch.cat((torch.tensor([[-2*floor, -0.5*floor, 0., 0.5*floor, 2*floor]], dtype=torch.float64).expand(batch, -1),
                   -1e4*a, -30*a, -6*a, -a, a, 6*a, 30*a, 1e4*a), dim=1)
    return a.expand_as(b).clone(), b


def _updater(cls, a, b, settings, family, Is=PARAMS["I_s"], omega=None):
    layer = NonlinearResistiveLayer(b.shape[1:], batch_size=b.shape[0], non_linearity=family)
    layer.state = torch.linspace(-0.5, 0.5, b.numel(), dtype=torch.float64).reshape(b.shape)
    kwargs = {} if omega is None else {"overrelaxation_factor": omega}
    return cls(layer, _Quadratic(a, b, layer), dict(PARAMS, I_s=Is), settings, **kwargs)


@pytest.mark.parametrize("polish", [False, True])
@pytest.mark.parametrize("cls,omega", [(Float64ExponentialDoubleDiodeUpdater, None),
                                       (OverRelaxedFloat64DoubleDiodeUpdater, 1.4)])
def test_double_diode_fused_matches_the_torch_path(polish, cls, omega):
    a, b = _case()
    torch_path = _updater(cls, a, b, _settings(polish), "double_diode_exponential", omega=omega)
    fused = _updater(cls, a, b, _settings(polish), "double_diode_exponential", omega=omega)
    fused._fused = module._double_diode_fused
    torch.testing.assert_close(fused.pre_activate(), torch_path.pre_activate(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("polish", [False, True])
@pytest.mark.parametrize("Is", [PARAMS["I_s"], 0.0])
@pytest.mark.parametrize("cls,omega", [(ExponentialSingleDiodeUpdater, None),
                                       (ConfigurableExponentialSingleDiodeUpdater, None),
                                       (OverRelaxedSingleDiodeUpdater, 1.4)])
def test_single_diode_fused_matches_the_torch_path(polish, Is, cls, omega):
    a, b = _case()
    torch_path = _updater(cls, a, b, _settings(polish), "single_diode_exponential", Is=Is, omega=omega)
    fused = _updater(cls, a, b, _settings(polish), "single_diode_exponential", Is=Is, omega=omega)
    fused._fused = module._single_diode_fused
    torch.testing.assert_close(fused.pre_activate(), torch_path.pre_activate(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("family,cls", [("double_diode_exponential", Float64ExponentialDoubleDiodeUpdater),
                                        ("single_diode_exponential", ConfigurableExponentialSingleDiodeUpdater)])
def test_torch_compile_gives_the_eager_fused_result(family, cls):
    a, b = _case(batch=2)
    core = module._double_diode_fused if family.startswith("double") else module._single_diode_fused
    eager = _updater(cls, a, b, _settings(True), family)
    eager._fused = core
    compiled = _updater(cls, a, b, _settings(True), family)
    compiled._fused = module._fused_function(core, True)
    torch.testing.assert_close(compiled.pre_activate(), eager.pre_activate(), rtol=1e-14, atol=1e-14)


def _minimizer(b, relaxation=1.2, method="voltage_change"):
    a = torch.full_like(b, 10.)
    layer = NonlinearResistiveLayer((b.shape[1],), non_linearity="single_diode_exponential")
    layer.state = torch.zeros_like(b)
    simulation = dict(nonlinearity="single_diode_exponential",
                      physical=dict(saturation_current=1e-6, thermal_voltage=.05, offset_voltage=.8),
                      amplification=dict(voltage_factor=1., current_factor=1.),
                      updater=dict(method="lambert_w_v1", dtype="float64", backend="torchlambertw",
                                   relaxation=relaxation, linear_coefficient_clamp=1e6,
                                   quadratic_coefficient_min=1e-30, exponent_clip=100.,
                                   asymptotic_threshold=1e10, asymptotic_terms=4, polish=False))
    equilibrium = (dict(method=method, max_sweeps=3, relative_tolerance=1e-5, absolute_tolerance=1e-6,
                        initial_state="zeros", update_order="asynchronous") if method == "voltage_change" else
                   dict(method="fixed_sweeps", sweeps=3, initial_state="zeros", update_order="asynchronous"))
    native = build_minimizer(fn=_Quadratic(a, b, layer), free_layers=[layer], simulation=simulation,
                             equilibrium=equilibrium, repo_root=Path.cwd())
    return native, layer


@pytest.mark.parametrize("method", ["voltage_change", "fixed_sweeps"])
def test_minimizer_switches_to_fused_updates_and_back(method):
    b = torch.tensor([[-4000., 30., -2., 4000.]], dtype=torch.float64)
    states = []
    for fused in (False, True):
        native, layer = _minimizer(b, method=method)
        if fused:
            assert native.use_fused_updates() == 1
        native.compute_equilibrium()
        states.append(layer.state.clone())
        if fused:
            assert native.use_fused_updates(False) == 1 and native._updaters[0]._fused is None
    torch.testing.assert_close(states[1], states[0], rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("method", ["voltage_change", "fixed_sweeps"])
def test_non_finite_voltages_raise_at_the_sweep(method):
    native, _ = _minimizer(torch.tensor([[float("nan"), 0.]], dtype=torch.float64), method=method)
    native.use_fused_updates()
    with pytest.raises(NonFiniteDiodeError):
        native.compute_equilibrium()


def test_fused_updates_and_compiled_kernels_are_exclusive():
    from model.resistive import _kernels
    native, _ = _minimizer(torch.tensor([[-4000., 4000.]], dtype=torch.float64))
    native.use_fused_updates()
    if _kernels.available():
        native.use_compiled_kernels()
        assert native._updaters[0]._fused is None and native._updaters[0]._compiled is not None
    native.use_fused_updates()
    assert native._updaters[0]._compiled is None and native._updaters[0]._fused is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("family,cls,core", [
    ("double_diode_exponential", Float64ExponentialDoubleDiodeUpdater, "_double_diode_fused"),
    ("single_diode_exponential", ConfigurableExponentialSingleDiodeUpdater, "_single_diode_fused")])
def test_fused_updates_on_cuda_match_cpu(family, cls, core):
    a, b = _case()
    cpu = _updater(cls, a, b, _settings(True), family)
    cpu._fused = getattr(module, core)
    gpu = _updater(cls, a.cuda(), b.cuda(), _settings(True), family)
    gpu._layer.state = gpu._layer.state.cuda()
    gpu._fused = getattr(module, core)
    torch.testing.assert_close(gpu.pre_activate().cpu(), cpu.pre_activate(), rtol=1e-13, atol=1e-13)
