"""Full-law scalar references and factory regressions for single Shockley CD."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
import torch

from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.minimizer import (
    ConfigurableExponentialSingleDiodeUpdater,
    ExponentialSingleDiodeUpdater,
    MinimizerSettings,
    NonFiniteDiodeError,
    OverRelaxedSingleDiodeUpdater,
)
from repro.minimizer_factory import build_minimizer


CLASSES = [ExponentialSingleDiodeUpdater, ConfigurableExponentialSingleDiodeUpdater,
           OverRelaxedSingleDiodeUpdater]


class Quadratic:
    def __init__(self, a, b, layer):
        self.a, self.b, self.layer = a, b, layer

    def a_coef_fn(self, layer):
        return lambda: self.a

    def b_coef_fn(self, layer):
        return lambda: self.b

    def grad_layer_fn(self, layer):
        return lambda: 2*self.a*layer.state+self.b

    def layers(self):
        return [self.layer]

    def params(self):
        return []


def make_updater(cls, a, b, *, off=0., Is=1e-6, polish=False, omega=1., threshold=1e10, clip=100.):
    layer = NonlinearResistiveLayer(b.shape[1:], batch_size=b.shape[0],
                                   non_linearity='single_diode_exponential')
    layer.state = torch.zeros_like(b)
    settings = MinimizerSettings(
        rel_tol=None, vn_tol=None, use_polish=polish,
        max_newton_iters=32 if polish else None, z_thresh=threshold, exp_clip=clip,
        experimental_newton_tol=None, b_clamp=1e6, pwl_extrapolation=None,
        pwl_nonconvergence_policy=None, lambertw_backend='torchlambertw',
        lambertw_asymptotic_terms=4, single_diode_min_a=1e-30,
        single_diode_polish_abs_tol=1e-10 if polish else None,
        single_diode_polish_rel_tol=1e-10 if polish else None,
        double_diode_polish_residual_tol=None,
    )
    kwargs = {'overrelaxation_factor': omega} if cls is OverRelaxedSingleDiodeUpdater else {}
    return cls(layer, Quadratic(a,b,layer), {'I_s':Is,'V_t':.05,'V_off':off},settings,**kwargs)


def bracketed_root(a, b, sign, off, Is=1e-6):
    """Independent solve of d*u+Is*expm1((u-off)/vt)=-sign*b."""
    d, force, vt = 2*a, -sign*b, .05
    if not Is:
        return -b/d
    lo = min(force/d, off)
    hi = min((force+Is)/d, off+vt*math.log1p(max(force-d*off,0)/Is))
    fn = lambda u: d*u-force+Is*math.expm1((u-off)/vt)
    # Roundoff at the analytical bounds can reverse an endpoint sign.
    margin = 16*np.finfo(float).eps*max(1.,abs(lo),abs(hi))
    u = brentq(fn, lo-margin, hi+margin, xtol=1e-13, rtol=1e-14)
    return sign*u


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('polish', [False,True])
@pytest.mark.parametrize('off', [0.,.8])
def test_full_law_roots_across_overflow_and_reverse_leakage(cls,dtype,polish,off):
    a=torch.tensor([[.0005],[.5],[10.]],dtype=dtype)
    values=torch.tensor([[-4000.,-30.,-.01,0.,.01,30.,4000.]],dtype=dtype)
    b=values.repeat(3,2)
    updater=make_updater(cls,a,b,off=off,polish=polish)
    actual=updater.pre_activate()
    aa=a.expand_as(b).double().numpy();bb=b.double().numpy()
    signs=np.ones(b.shape);signs[:,b.shape[1]//2:]=-1
    expected=np.array([bracketed_root(x,y,s,off) for x,y,s in zip(aa.flat,bb.flat,signs.flat)]).reshape(b.shape)
    assert torch.isfinite(actual).all() and actual.dtype==dtype
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-7 if dtype==torch.float32 else 2e-10,
                               rtol=2e-7 if dtype==torch.float32 else 2e-13)
    v=actual.double().numpy()
    residual=2*aa*v+bb+signs*1e-6*np.expm1((signs*v-off)/.05)
    np.testing.assert_allclose(residual/(1+abs(bb)+abs(2*aa*v)),0.,
                               atol=2e-6 if dtype==torch.float32 else 2e-12)


@pytest.mark.parametrize('cls',CLASSES)
def test_stable_reconstruction_when_linear_voltage_is_enormous(cls):
    # linear root is 1e20 V but the conducting diode root is around 1 V.
    b=torch.tensor([[-1000.,1000.]],dtype=torch.float64)
    a=torch.full_like(b,5e-18)
    result=make_updater(cls,a,b).pre_activate()
    expected=np.array([[bracketed_root(5e-18,-1000,1,0),bracketed_root(5e-18,1000,-1,0)]])
    np.testing.assert_allclose(result.numpy(),expected,rtol=1e-13,atol=1e-12)


@pytest.mark.parametrize('threshold',[1.0001,1e10,1e300])
def test_backend_never_receives_overflowing_fp32_arguments(threshold):
    b=torch.tensor([[-.01,-10.,-4000.,.01,10.,4000.]],dtype=torch.float64)
    a=torch.full_like(b,.5)
    updater=make_updater(ConfigurableExponentialSingleDiodeUpdater,a,b,threshold=threshold)
    backend=updater._lambertw
    def checked(z):
        assert torch.isfinite(z.float()).all()
        return backend(z)
    updater._lambertw=checked
    result=updater.pre_activate()
    signs=[1]*3+[-1]*3
    expected=[bracketed_root(.5,float(bb),s,0) for bb,s in zip(b[0],signs)]
    np.testing.assert_allclose(result.numpy(),[expected],atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('polish',[False,True])
def test_exponent_clip_cannot_change_the_full_law_root(polish):
    a=torch.full((1,4),10.,dtype=torch.float64)
    b=torch.tensor([[-4000.,-30.,4000.,30.]],dtype=torch.float64)
    small=make_updater(ConfigurableExponentialSingleDiodeUpdater,a,b,clip=1.,polish=polish).pre_activate()
    large=make_updater(ConfigurableExponentialSingleDiodeUpdater,a,b,clip=10000.,polish=polish).pre_activate()
    torch.testing.assert_close(small,large,rtol=0,atol=0)


@pytest.mark.parametrize('cls',CLASSES)
def test_zero_saturation_current_is_linear(cls):
    b=torch.tensor([[-20.,0.,20.,0.]],dtype=torch.float64)
    a=torch.full_like(b,.5)
    torch.testing.assert_close(make_updater(cls,a,b,Is=0,polish=True).pre_activate(),-b/(2*a))


def test_relaxation_is_applied_after_stable_root():
    a=torch.full((1,2),10.,dtype=torch.float64)
    b=torch.tensor([[-4000.,4000.]],dtype=torch.float64)
    plain=make_updater(OverRelaxedSingleDiodeUpdater,a,b)
    relaxed=make_updater(OverRelaxedSingleDiodeUpdater,a,b,omega=1.2)
    relaxed._layer.state.fill_(.3)
    expected=.3+1.2*(plain.pre_activate()-.3)
    torch.testing.assert_close(relaxed.pre_activate(),expected)


@pytest.mark.parametrize('relaxation',[1.,1.2])
def test_training_factory_routes_to_the_stable_kernel(relaxation):
    a=torch.full((1,2),10.,dtype=torch.float64)
    b=torch.tensor([[-4000.,4000.]],dtype=torch.float64)
    layer=NonlinearResistiveLayer((2,),non_linearity='single_diode_exponential')
    layer.state=torch.zeros_like(b)
    fn=Quadratic(a,b,layer)
    simulation=dict(nonlinearity='single_diode_exponential',
        physical=dict(saturation_current=1e-6,thermal_voltage=.05,offset_voltage=0.),
        amplification=dict(voltage_factor=1.,current_factor=1.),
        updater=dict(method='lambert_w_v1',dtype='float64',backend='torchlambertw',
                     relaxation=relaxation,linear_coefficient_clamp=1e6,
                     quadratic_coefficient_min=1e-30,exponent_clip=100.,
                     asymptotic_threshold=1e10,asymptotic_terms=4,polish=False))
    native=build_minimizer(fn=fn,free_layers=[layer],simulation=simulation,
        equilibrium=dict(method='fixed_sweeps',sweeps=1,initial_state='zeros',update_order='asynchronous'),
        repo_root=Path.cwd())
    native.compute_equilibrium()
    expected=relaxation*np.array([[bracketed_root(10,-4000,1,0),bracketed_root(10,4000,-1,0)]])
    np.testing.assert_allclose(layer.state.numpy(),expected,rtol=1e-12,atol=1e-12)


def test_nonfinite_coefficients_raise_explicitly():
    a=torch.ones((1,2),dtype=torch.float64)
    b=torch.tensor([[float('nan'),0.]],dtype=torch.float64)
    with pytest.raises(NonFiniteDiodeError):
        make_updater(ConfigurableExponentialSingleDiodeUpdater,a,b).pre_activate()


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('polish', [False, True])
def test_one_pass_matches_solving_both_orientations(cls, polish):
    # Every node is solved once in its own orientation instead of both ways.
    a = torch.tensor([[.0005], [.5], [10.]], dtype=torch.float64)
    b = torch.tensor([[-4000., -30., -.01, 0., .01, 30., 4000., -2., 3., .5]], dtype=torch.float64).repeat(3, 1)
    updater = make_updater(cls, a, b, off=.8, polish=polish)
    settings = dict(z_thresh=1e10, polish=polish, abs_tol=1e-10 if polish else None,
                    rel_tol=1e-10 if polish else None, exp_clip=100., a_min=1e-30,
                    max_inner_iter=32 if polish else None)
    aa = a.expand_as(b)
    forward = updater.lambert_single_forward(aa, b, 1e-6, .8, .05, **settings)
    reverse = updater.lambert_single_reverse(aa, b, 1e-6, .8, .05, **settings)
    reverse_half = torch.arange(b.shape[1]) >= b.shape[1]//2
    torch.testing.assert_close(updater.pre_activate(), torch.where(reverse_half, reverse, forward), rtol=0, atol=1e-13)


from model.resistive import _kernels  # noqa: E402

compiled = pytest.mark.skipif(not _kernels.available(), reason='the compiled kernels need a C compiler')


@compiled
@pytest.mark.parametrize('cls,omega', [(ExponentialSingleDiodeUpdater, 1.), (ConfigurableExponentialSingleDiodeUpdater, 1.),
                                       (OverRelaxedSingleDiodeUpdater, 1.4)])
@pytest.mark.parametrize('polish', [False, True])
@pytest.mark.parametrize('off', [0., .8])
@pytest.mark.parametrize('Is', [1e-6, 0.])
def test_compiled_kernel_matches_the_torch_path(cls, omega, polish, off, Is):
    a = torch.tensor([[.0005], [.5], [10.]], dtype=torch.float64)
    b = torch.tensor([[-4000., -30., -.01, 0., .01, 30., 4000., -2., 3., .5]], dtype=torch.float64).repeat(3, 1)
    torch_path = make_updater(cls, a, b, off=off, Is=Is, polish=polish, omega=omega)
    kernel = make_updater(cls, a, b, off=off, Is=Is, polish=polish, omega=omega)
    kernel._compiled = _kernels.load()
    for updater in (torch_path, kernel):
        updater._layer.state = torch.linspace(-.5, .5, b.numel(), dtype=torch.float64).reshape(b.shape)
    torch.testing.assert_close(kernel.pre_activate(), torch_path.pre_activate(), rtol=1e-13, atol=1e-12)


@compiled
def test_compiled_kernel_rejects_non_finite_coefficients_too():
    a = torch.ones((1, 2), dtype=torch.float64)
    b = torch.tensor([[float('nan'), 0.]], dtype=torch.float64)
    updater = make_updater(ConfigurableExponentialSingleDiodeUpdater, a, b)
    updater._compiled = _kernels.load()
    with pytest.raises(NonFiniteDiodeError):
        updater.pre_activate()


@compiled
@pytest.mark.parametrize('relaxation', [1., 1.2])
def test_minimizer_switches_to_the_compiled_kernels(relaxation):
    a = torch.full((1, 4), 10., dtype=torch.float64)
    b = torch.tensor([[-4000., 30., -2., 4000.]], dtype=torch.float64)
    states = []
    for enabled in (False, True):
        layer = NonlinearResistiveLayer((4,), non_linearity='single_diode_exponential')
        layer.state = torch.zeros_like(b)
        simulation = dict(nonlinearity='single_diode_exponential',
            physical=dict(saturation_current=1e-6, thermal_voltage=.05, offset_voltage=.8),
            amplification=dict(voltage_factor=1., current_factor=1.),
            updater=dict(method='lambert_w_v1', dtype='float64', backend='torchlambertw', relaxation=relaxation,
                         linear_coefficient_clamp=1e6, quadratic_coefficient_min=1e-30, exponent_clip=100.,
                         asymptotic_threshold=1e10, asymptotic_terms=4, polish=False))
        native = build_minimizer(fn=Quadratic(a, b, layer), free_layers=[layer], simulation=simulation,
            equilibrium=dict(method='fixed_sweeps', sweeps=2, initial_state='zeros', update_order='asynchronous'),
            repo_root=Path.cwd())
        if enabled:
            assert native.use_compiled_kernels() == 1
            assert native._updaters[0]._compiled is not None
        native.compute_equilibrium()
        states.append(layer.state.clone())
        if enabled:
            assert native.use_compiled_kernels(False) == 1 and native._updaters[0]._compiled is None
    torch.testing.assert_close(states[1], states[0], rtol=1e-13, atol=1e-12)
