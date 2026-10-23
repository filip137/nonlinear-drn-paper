from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.optimize import brentq

from model.function.interaction import DoubleExponentialNonLinearInteraction
from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.minimizer import (
    ExponentialDoubleDiodeUpdater,
    Float32ExponentialDoubleDiodeUpdater,
    Float64ExponentialDoubleDiodeUpdater,
    MinimizerSettings,
    OverRelaxedFloat64DoubleDiodeUpdater,
)


PARAMS = {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.8}
UPDATERS = [
    ExponentialDoubleDiodeUpdater,
    Float64ExponentialDoubleDiodeUpdater,
    Float32ExponentialDoubleDiodeUpdater,
]


class _Quadratic:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def grad_layer_fn(self, layer):
        return lambda: 2 * self.a * layer.state + self.b

    def a_coef_fn(self, layer):
        return lambda: self.a

    def b_coef_fn(self, layer):
        return lambda: self.b


def _updater(cls, a, b, *, params=PARAMS, polish=False, omega=1.0):
    layer = NonlinearResistiveLayer(
        b.shape[1:], batch_size=b.shape[0], non_linearity="double_diode_exponential"
    )
    layer.state = torch.zeros_like(b)
    layer._depth_index = 1
    settings = MinimizerSettings(
        rel_tol=None,
        vn_tol=None,
        use_polish=polish,
        max_newton_iters=32 if polish else None,
        z_thresh=1e10,
        exp_clip=70.0,
        experimental_newton_tol=None,
        b_clamp=1e6,
        pwl_extrapolation=None,
        pwl_nonconvergence_policy=None,
        lambertw_backend="torchlambertw",
        lambertw_asymptotic_terms=4,
        single_diode_min_a=None,
        single_diode_polish_abs_tol=None,
        single_diode_polish_rel_tol=None,
        double_diode_polish_residual_tol=1e-16 if polish else None,
    )
    kwargs = {}
    if cls in (Float32ExponentialDoubleDiodeUpdater, OverRelaxedFloat64DoubleDiodeUpdater):
        kwargs["overrelaxation_factor"] = omega
    return cls(layer, _Quadratic(a, b), params, settings, **kwargs)


def _bracketed_root(a, b, params):
    """Independent scalar reference: bracket KCL directly, without Lambert W."""
    Is, vt, off = (params[key] for key in ("I_s", "V_t", "V_off"))
    c = Is * math.exp(-off / vt)
    if abs(b) <= c:
        return 0.0
    if Is == 0:
        return -b / (2 * a)
    sign = -math.copysign(1.0, b)
    upper = min(abs(b) / (2 * a), off + vt * math.log(abs(b) / Is))

    def residual(t):
        return 2 * a * t - abs(b) + Is * math.exp((t - off) / vt)

    t = brentq(residual, 0.0, upper * (1 + 1e-12), xtol=1e-15, rtol=1e-14)
    return sign * t


@pytest.mark.parametrize("cls", UPDATERS)
@pytest.mark.parametrize("polish", [False, True])
def test_coordinate_update_matches_bracketed_kcl(cls, polish):
    dtype = cls._work_dtype
    c = PARAMS["I_s"] * math.exp(-PARAMS["V_off"] / PARAMS["V_t"])
    a = torch.tensor([[1e-6], [1e-4], [1e-2]], dtype=dtype)
    b = torch.tensor([[-2*c, -0.5*c, 0, 0.5*c, 2*c]], dtype=dtype).expand(3, -1)
    # Ordinary operating points and large Lambert-W arguments in both signs.
    b = torch.cat((b, -6*a, -a, a, 6*a), dim=1)
    a = a.expand_as(b)
    updater = _updater(cls, a, b, polish=polish)
    result = updater.pre_activate()
    expected = np.array([
        _bracketed_root(float(aa), float(bb), PARAMS)
        for aa, bb in zip(a.flatten(), b.flatten())
    ]).reshape(b.shape)
    # The unpolished large-z expansion has a finite truncation error.
    atol = 4e-7 if dtype == torch.float32 else (2e-13 if polish else 2e-7)
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=atol)
    assert result.dtype == b.dtype
    assert torch.all(result * b <= 0)


@pytest.mark.parametrize("cls", UPDATERS)
@pytest.mark.parametrize("polish", [False, True])
def test_root_with_diode_free_exponent_beyond_exp_clip(cls, polish):
    # Diode-free voltages A = |b|/2a of 5-5000 V put (A - V_off)/V_t at 84-1e5,
    # far past exp_clip = 70; the root itself sits near V_off + V_t log(|b|/I_s).
    dtype = cls._work_dtype
    a = torch.tensor([[1e-6], [1e-4], [1e-2]], dtype=dtype)
    b = torch.cat((-1e4 * a, -1e3 * a, -1e2 * a, -10 * a, 10 * a, 1e2 * a, 1e3 * a, 1e4 * a), dim=1)
    a = a.expand_as(b)
    result = _updater(cls, a, b, polish=polish).pre_activate()
    expected = np.array([
        _bracketed_root(float(aa), float(bb), PARAMS)
        for aa, bb in zip(a.flatten(), b.flatten())
    ]).reshape(b.shape)
    if dtype == torch.float32:
        atol = 1e-6 if polish else 1e-4
    else:
        atol = 1e-12 if polish else 1e-5
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=atol)
    assert np.abs(expected).max() < 2.0


@pytest.mark.parametrize("cls", UPDATERS)
@pytest.mark.parametrize("polish", [False, True])
def test_zero_branch_includes_both_subgradient_endpoints(cls, polish):
    # V_off = 0 makes the endpoints exactly representable in each work dtype.
    params = dict(PARAMS, I_s=0.125, V_off=0.0)
    b = torch.tensor([[-0.125, -0.05, 0, 0.05, 0.125]], dtype=cls._work_dtype)
    updater = _updater(cls, torch.ones_like(b), b, params=params, polish=polish)
    assert torch.equal(updater.pre_activate(), torch.zeros_like(b))


@pytest.mark.parametrize("cls", UPDATERS)
def test_zero_saturation_current_reduces_to_quadratic(cls):
    a = torch.tensor([[0.1, 0.2, 0.3]], dtype=cls._work_dtype)
    b = torch.tensor([[-0.1, 0, 0.1]], dtype=cls._work_dtype)
    updater = _updater(cls, a, b, params=dict(PARAMS, I_s=0.0), polish=True)
    torch.testing.assert_close(updater.pre_activate(), -b / (2*a))


@pytest.mark.parametrize("cls", [
    Float32ExponentialDoubleDiodeUpdater,
    OverRelaxedFloat64DoubleDiodeUpdater,
])
def test_relaxation_is_applied_after_the_corrected_coordinate_minimum(cls):
    b = torch.tensor([[-1e-5, 0, 1e-5]], dtype=cls._work_dtype)
    a = torch.full_like(b, 1e-4)
    plain = _updater(cls, a, b, omega=1.0)
    relaxed = _updater(cls, a, b, omega=1.2)
    relaxed._layer.state = torch.full_like(b, 0.3)
    expected = relaxed._layer.state + 1.2 * (plain.pre_activate() - relaxed._layer.state)
    torch.testing.assert_close(relaxed.pre_activate(), expected)


def test_float64_work_preserves_float32_state_dtype():
    b = torch.tensor([[-1e-5, 0, 1e-5]], dtype=torch.float32)
    updater = _updater(Float64ExponentialDoubleDiodeUpdater, torch.full_like(b, 1e-4), b)
    assert updater.pre_activate().dtype == torch.float32


@pytest.mark.parametrize("depth,voltage_amp,current_amp", [(1, 1., 1.), (3, 4., 1.)])
def test_energy_is_current_primitive_and_gradient_is_monotone(depth, voltage_amp, current_amp):
    params = dict(PARAMS, I_s=1e-3, V_off=0.1)
    v = torch.tensor(
        [[-0.3], [-0.1], [-0.02], [-1e-12], [0], [1e-12], [0.02], [0.1], [0.3]],
        dtype=torch.float64, requires_grad=True,
    )
    layer = NonlinearResistiveLayer((1,), batch_size=len(v), non_linearity="double_diode_exponential")
    layer.state = v
    layer._depth_index = depth
    interaction = DoubleExponentialNonLinearInteraction(layer, params, voltage_amp, current_amp)
    energy = interaction.eval()
    grad, = torch.autograd.grad(energy.sum(), v)
    analytical = interaction.grad_layer_fn(layer)()
    torch.testing.assert_close(grad, analytical, rtol=2e-14, atol=1e-18)
    assert torch.all(torch.diff(analytical[:, 0]) >= 0)
    assert energy[4].item() == grad[4].item() == 0.0
    assert torch.all(energy >= 0)

    scale = (current_amp / voltage_amp) ** (depth - 1)
    expected = [
        scale * quad(
            lambda t: params["I_s"] * math.exp((t-params["V_off"]) / params["V_t"]),
            0, abs(float(voltage.detach())),
            epsabs=1e-20, epsrel=1e-12,
        )[0]
        for voltage in v[:, 0]
    ]
    np.testing.assert_allclose(energy.detach().numpy(), expected, rtol=2e-14, atol=1e-28)


def test_coordinate_update_minimizes_the_same_energy_used_in_training():
    params = dict(PARAMS, I_s=1e-3, V_off=0.1)
    b = torch.tensor([[-3e-3, -5e-5, 0, 5e-5, 3e-3]], dtype=torch.float64)
    a = torch.full_like(b, 1e-2)
    updater = _updater(Float64ExponentialDoubleDiodeUpdater, a, b, params=params, polish=True)
    layer = updater._layer
    interaction = DoubleExponentialNonLinearInteraction(layer, params, 1.0, 1.0)
    optimum = updater.pre_activate()

    def energy_at(v):
        layer.state = v
        return (a*v*v + b*v).sum(dim=1) + interaction.eval()

    minimum = energy_at(optimum)
    for perturbation in (-0.1, -1e-5, 1e-5, 0.1):
        assert torch.all(energy_at(optimum + perturbation) > minimum)


from model.resistive import _kernels  # noqa: E402
from model.resistive.minimizer import NonFiniteDiodeError  # noqa: E402

compiled = pytest.mark.skipif(not _kernels.available(), reason="the compiled kernels need a C compiler")


def _kernel_case(broadcast):
    a = torch.tensor([[1e-6], [1e-4], [1e-2]], dtype=torch.float64)
    c = PARAMS["I_s"] * math.exp(-PARAMS["V_off"] / PARAMS["V_t"])
    b = torch.cat((torch.tensor([[-2*c, -0.5*c, 0., 0.5*c, 2*c]], dtype=torch.float64).expand(3, -1),
                   -1e4*a, -6*a, -a, a, 6*a, 1e4*a), dim=1)
    return (a[1:2].expand(1, b.shape[1]).clone() if broadcast else a.expand_as(b).clone()), b


@compiled
@pytest.mark.parametrize("cls,omega", [(ExponentialDoubleDiodeUpdater, 1.0), (Float64ExponentialDoubleDiodeUpdater, 1.0),
                                       (OverRelaxedFloat64DoubleDiodeUpdater, 1.4)])
@pytest.mark.parametrize("polish", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
def test_compiled_kernel_matches_the_torch_path(cls, omega, polish, broadcast):
    a, b = _kernel_case(broadcast)
    torch_path = _updater(cls, a, b, polish=polish, omega=omega)
    kernel = _updater(cls, a, b, polish=polish, omega=omega)
    kernel._compiled = _kernels.load()
    for updater in (torch_path, kernel):
        updater._layer.state = torch.linspace(-0.5, 0.5, b.numel(), dtype=torch.float64).reshape(b.shape)
    expected, actual = torch_path.pre_activate(), kernel.pre_activate()
    # Unpolished, both refine W to FP64 (the shared Lambert-W rule).
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    if omega == 1.0:
        aa = a.expand_as(b)
        reference = np.array([_bracketed_root(float(x), float(y), PARAMS) for x, y in zip(aa.flatten(), b.flatten())])
        np.testing.assert_allclose(actual.numpy().ravel(), reference, rtol=2e-6, atol=2e-13)


@compiled
def test_compiled_kernel_rejects_non_finite_coefficients():
    a, b = _kernel_case(False)
    b[0, 0] = float("nan")
    updater = _updater(Float64ExponentialDoubleDiodeUpdater, a, b, polish=True)
    updater._compiled = _kernels.load()
    with pytest.raises(NonFiniteDiodeError):
        updater.pre_activate()


@compiled
def test_compiled_kernel_leaves_float32_work_and_autograd_to_torch():
    a, b = _kernel_case(False)
    float32 = _updater(Float32ExponentialDoubleDiodeUpdater, a.float(), b.float(), polish=True)
    expected = float32.pre_activate()
    float32._compiled = _kernels.load()
    torch.testing.assert_close(float32.pre_activate(), expected, rtol=0, atol=0)
    b = b.clone().requires_grad_(True)
    recorded = _updater(Float64ExponentialDoubleDiodeUpdater, a, b, polish=True)
    recorded._compiled = _kernels.load()
    assert recorded.pre_activate().grad_fn is not None
