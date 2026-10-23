"""The double- and single-diode coordinate solves share one Lambert-W logic.

The double-diode magnitude equation 2a t - |b| + I_s exp((t - V_off)/V_t) = 0 is the
single diode's forward equation with b = I_s - |b| (and the reversed one with
b = |b| - I_s, v = -t). Solved without polish steps, both families expose their raw
Lambert-W evaluation, so equal voltages mean the same threshold, series and branch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import lambertw
import torch

from model.resistive import _kernels
from model.resistive import minimizer as module
from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.minimizer import (
    ConfigurableExponentialSingleDiodeUpdater,
    Float64ExponentialDoubleDiodeUpdater,
    MinimizerSettings,
)

PARAMS = {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.8}


class _Quadratic:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def grad_layer_fn(self, layer):
        return lambda: 2 * self.a * layer.state + self.b

    def a_coef_fn(self, layer):
        return lambda: self.a

    def b_coef_fn(self, layer):
        return lambda: self.b


def _settings(z_thresh, polish_steps, terms=4, exp_clip=100.0):
    return MinimizerSettings(
        rel_tol=None, vn_tol=None, use_polish=True, max_newton_iters=polish_steps, z_thresh=z_thresh,
        exp_clip=exp_clip, experimental_newton_tol=None, b_clamp=1e6, pwl_extrapolation=None,
        pwl_nonconvergence_policy=None, lambertw_backend="torchlambertw", lambertw_asymptotic_terms=terms,
        single_diode_min_a=1e-30, single_diode_polish_abs_tol=1e-12, single_diode_polish_rel_tol=1e-12,
        double_diode_polish_residual_tol=1e-16,
    )


def _pair(settings, compiled=False):
    """Matching double- and single-diode layers whose diode-free exponents span 1e-3 to 1e5."""
    a = torch.full((1, 16), 1e-3, dtype=torch.float64)
    drive = 2e-3 * (PARAMS["V_off"] + PARAMS["V_t"] * torch.tensor(
        [0.5, 5, 15, 21, 23.5, 30, 200, 1e5], dtype=torch.float64))
    b_double = torch.cat((-drive, drive)).unsqueeze(0)
    b_single = torch.cat((PARAMS["I_s"] - drive, drive - PARAMS["I_s"])).unsqueeze(0)
    updaters = []
    for cls, b, law in ((Float64ExponentialDoubleDiodeUpdater, b_double, "double_diode_exponential"),
                        (ConfigurableExponentialSingleDiodeUpdater, b_single, "single_diode_exponential")):
        layer = NonlinearResistiveLayer((16,), non_linearity=law)
        layer.state = torch.zeros_like(b)
        updater = cls(layer, _Quadratic(a, b), PARAMS, settings)
        if compiled:
            updater._compiled = _kernels.load()
        updaters.append(updater)
    return updaters


@pytest.mark.parametrize("z_thresh", [1e10, 1e300])
@pytest.mark.parametrize("terms", [1, 4])
def test_both_families_take_the_same_branch_and_series(z_thresh, terms):
    # No polish steps: the voltages are the raw Lambert-W reconstructions. z_thresh = 1e300
    # exercises the shared cap at the FP32 range of the backend.
    double, single = _pair(_settings(z_thresh, polish_steps=0, terms=terms))
    torch.testing.assert_close(single.pre_activate(), double.pre_activate(), rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not _kernels.available(), reason="the compiled kernels need a C compiler")
@pytest.mark.parametrize("polish_steps", [0, 32])
def test_compiled_kernels_share_the_logic_too(polish_steps):
    double, single = _pair(_settings(1e10, polish_steps), compiled=True)
    torch.testing.assert_close(single.pre_activate(), double.pre_activate(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("z_thresh", [1.0001, 1e10, 1e300])
def test_shared_lambertw_matches_scipy_on_both_sides_of_the_threshold(z_thresh):
    log_z = torch.tensor([-800., -40., -1., 0., 1e-3, 1., 7.9, 8.1, 20., 23., 24., 80., 90., 700., 1e5],
                         dtype=torch.float64)
    backend = module._load_lambertw_backend("torchlambertw")
    w = module._shockley_lambertw(log_z, backend, z_thresh=z_thresh, terms=4, refine=True)
    # Reference from log z: W = log z - log W solved in mpmath-free FP64 for the large values.
    expected = np.array([lambertw(math.exp(x)).real if x < 700 else float("nan") for x in log_z.tolist()])
    large = ~np.isfinite(expected)
    np.testing.assert_allclose(w.numpy()[~large], expected[~large], rtol=1e-14, atol=1e-300)
    l = log_z.numpy()[large]
    np.testing.assert_allclose(w.numpy()[large] + np.log(w.numpy()[large]), l, rtol=1e-15)


@pytest.mark.parametrize("polish_steps", [0, 32])
def test_exp_clip_changes_neither_family(polish_steps):
    for small, large in zip(_pair(_settings(1e10, polish_steps, exp_clip=1.0)),
                            _pair(_settings(1e10, polish_steps, exp_clip=1e4))):
        torch.testing.assert_close(small.pre_activate(), large.pre_activate(), rtol=0, atol=0)
