"""Check the compiled coordinates against independent physical KCL and Newton."""
import numpy as np
import pytest
from scipy.optimize import brentq

from repro.solvers import (
    AdaptiveRelaxation,
    DoubleShockley,
    FastCD,
    LayeredDRN,
    SingleDiodeDRN,
    SingleShockley,
    newton,
)


@pytest.fixture(autouse=True)
def kernel_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DRN_KERNEL_CACHE", str(tmp_path / "kernels"))


def network(family, physical=None):
    rng = np.random.default_rng(43)
    weights = [rng.uniform(0.01, 0.1, (a, b)) for a, b in ((4, 6), (6, 4), (4, 2))]
    physical = physical or dict(
        saturation_current=1e-6, thermal_voltage=0.05, offset_voltage=0.8
    )
    circuit = (
        SingleDiodeDRN(weights, physical, 4.0, 1.0)
        if family == "single"
        else LayeredDRN(weights, DoubleShockley(**physical), 4.0, 1.0)
    )
    return weights, physical, circuit


@pytest.mark.parametrize("family", ["single", "double"])
@pytest.mark.parametrize(
    "options", [{}, {"small": False}, {"warm": False}, {"polish": True}]
)
def test_fast_cd_reaches_independently_evaluated_equilibrium(family, options):
    weights, physical, circuit = network(family)
    fast = FastCD(weights, family, 4.0, 1.0, physical=physical, **options)
    inputs = np.array([10.0, -7.0, 3.0, -5.0])
    forcing = circuit.forcing(inputs)
    steps = fast.solve(
        forcing[circuit.slices[0]],
        AdaptiveRelaxation(),
        rtol=1e-12,
        atol=1e-13,
        max_sweeps=4096,
    )
    assert fast.converged and steps > 1
    assert np.max(np.abs(circuit.residual(fast.state, forcing))) < 2e-10
    reference = newton(circuit, forcing, tolerance=1e-12, residual_kind="current")
    assert reference.converged
    np.testing.assert_allclose(fast.state, reference.voltages, atol=2e-9, rtol=0)
    expected = fast.state
    fast.solve(-forcing[circuit.slices[0]], AdaptiveRelaxation())
    repeated = fast.solve(
        forcing[circuit.slices[0]],
        AdaptiveRelaxation(),
        rtol=1e-12,
        atol=1e-13,
        max_sweeps=4096,
    )
    assert repeated == steps
    np.testing.assert_array_equal(fast.state, expected)


@pytest.mark.parametrize("family", ["single", "double"])
def test_custom_device_parameters_are_used(family):
    physical = dict(saturation_current=1e-8, thermal_voltage=0.06, offset_voltage=0.9)
    weights, _, circuit = network(family, physical)
    forcing = circuit.forcing(np.array([4.0, -5.0, 8.0, -1.0]))
    fast = FastCD(weights, family, 4.0, 1.0, physical=physical)
    fast.solve(forcing[circuit.slices[0]], 1.2, rtol=1e-12, atol=1e-13, max_sweeps=4096)
    assert fast.converged
    assert np.max(np.abs(circuit.residual(fast.state, forcing))) < 2e-10


def test_single_orientations_against_bracketed_roots():
    diagonal = np.geomspace(1e-3, 1e2, 31)
    forcing = np.linspace(-10, 10, 31)
    for sign in (1.0, -1.0):
        law = SingleShockley(signs=tuple(np.full(31, sign)))
        expected = []
        for d, f in zip(diagonal, forcing):
            bound = abs(f) / d + 2
            with np.errstate(over="ignore"):
                expected.append(
                    brentq(
                        lambda v: d * v
                        - f
                        + sign * 1e-6 * np.expm1((sign * v - 0.8) / 0.05),
                        -bound,
                        bound,
                        xtol=5e-14,
                    )
                )
        np.testing.assert_allclose(
            law.coordinate_root(diagonal, forcing), expected, atol=1e-12, rtol=3e-14
        )


def test_exhaustion_and_invalid_inputs_do_not_report_success():
    weights, _, circuit = network("double")
    fast = FastCD(weights, "double", 4.0, 1.0)
    forcing = circuit.forcing(np.ones(4))[circuit.slices[0]]
    assert fast.solve(forcing, 1.0, max_sweeps=0) == 0
    assert not fast.converged and fast.reason == "maximum_sweeps"
    assert fast.solve(forcing, 1.0, max_sweeps=1, rtol=0.0, atol=0.0) == 1
    assert not fast.converged
    for feed in (np.zeros(5), np.full(6, np.nan), np.zeros((1, 6))):
        with pytest.raises(ValueError):
            fast.solve(feed, 1.0)
    for kwargs in (
        {"max_sweeps": -1},
        {"max_sweeps": 1.5},
        {"rtol": np.nan},
        {"atol": -1.0},
    ):
        with pytest.raises(ValueError):
            fast.solve(forcing, 1.0, **kwargs)
    with pytest.raises(ValueError):
        FastCD(weights, "typo", 4.0, 1.0)
    with pytest.raises(ValueError):
        FastCD([np.array([[-1.0]])], "double")


def test_missing_compiler_is_actionable(monkeypatch, tmp_path):
    from repro.solvers import _kernels

    monkeypatch.setattr(_kernels, "_library", None)
    monkeypatch.setenv("CC", str(tmp_path / "missing-compiler"))
    with pytest.raises(_kernels.KernelUnavailable, match="not usable"):
        _kernels.load_kernels()
