"""Numerical checks of the review's measured-table CD and modified Newton."""
import numpy as np
import pytest

from repro.solvers import (
    BlockCholesky,
    DoubleShockley,
    LayeredDRN,
    PWLCoordinateDescent,
    TableLaw,
    newton,
    newton_cached,
)


def circuit():
    grid = np.array([-2.0, -0.5, 0.0, 0.7, 2.0])
    nmos = np.array([-1.0, -0.05, 0.0, 0.2, 3.0]) * 1e-7
    pmos = np.array([-4.0, -0.2, -0.01, 0.05, 0.8]) * 1e-7
    law = TableLaw.paired(grid, nmos, pmos, [4, 2])
    rng = np.random.default_rng(12)
    weights = [rng.uniform(0.05, 0.2, (a, b)) for a, b in ((3, 4), (4, 2), (2, 2))]
    return LayeredDRN(weights, law, 4.0, 1.0)


def test_table_law_units_knots_and_linear_extrapolation():
    law = TableLaw(
        [-1.0, 0.0, 1.0],
        [[-2e-7, 0.0, 4e-7], [-3e-7, -1e-7, 0.0]],
        [0, 1],
        current_unit_A=1e-7,
    )
    current, derivative = law.current_derivative(np.array([[-2.0, 2.0], [0.0, 0.0]]))
    np.testing.assert_allclose(current, [[-4.0, 1.0], [0.0, -1.0]])
    np.testing.assert_allclose(derivative, [[2.0, 1.0], [4.0, 1.0]])
    with pytest.raises(ValueError):
        law.current_derivative(np.zeros(3))
    with pytest.raises(ValueError):
        TableLaw([0.0, 0.0], [[0.0, 1.0]], [0])
    with pytest.raises(ValueError):
        TableLaw([0.0, 1.0], [[1.0, 0.0]], [0])


@pytest.mark.parametrize("adaptive", [False, True])
def test_pwl_cd_newton_and_cached_newton_agree(adaptive):
    c = circuit()
    force = c.forcing(np.array([3.0, -2.0, 1.0]))
    cd = PWLCoordinateDescent(c, adaptive=adaptive, compiled=False)
    result = cd.solve(force, tolerance=1e-11)
    block = newton(c, force, residual_kind="current", tolerance=1e-11)
    dense = newton(
        c, force, residual_kind="current", tolerance=1e-11, backend="dense_lu"
    )
    cached = newton_cached(c, force, tolerance=1e-11, theta=0.9)
    assert all(r.converged for r in (result, block, dense, cached))
    for r in (result, dense, cached):
        np.testing.assert_allclose(r.voltages, block.voltages, atol=1e-9, rtol=0)
    first = result.voltages.copy()
    cd.solve(-force)
    np.testing.assert_array_equal(cd.solve(force, tolerance=1e-11).voltages, first)


def test_compiled_pwl_sweep_matches_eager(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    c = circuit()
    force = c.forcing(np.array([3.0, -2.0, 1.0]))
    eager = PWLCoordinateDescent(c, adaptive=False, compiled=False)
    compiled = PWLCoordinateDescent(c, adaptive=False, compiled=True)
    eager.reset(force)
    compiled.reset(force)
    for _ in range(6):
        eager.step()
        compiled.step()
        np.testing.assert_allclose(
            compiled.state(), eager.state(), rtol=2e-14, atol=2e-14
        )


def test_cached_factors_match_fresh_linear_solve_and_zero_budget():
    c = circuit()
    force = c.forcing(np.array([3.0, -2.0, 1.0]))
    _, derivative = c.residual_derivative(np.zeros(c.size), force)
    np.testing.assert_allclose(
        BlockCholesky(c, derivative).solve(force),
        c.linear_step(derivative, force),
        atol=1e-14,
    )
    result = newton_cached(c, force, max_iterations=0)
    assert not result.converged and result.iterations == result.factorizations == 0
    assert result.reason == "maximum_iterations"
    # PMOS has nonzero current at zero voltage.
    assert not newton_cached(c, np.zeros(c.size), max_iterations=0).converged
    zero = LayeredDRN([np.ones((1, 1))], DoubleShockley())
    assert newton_cached(zero, np.zeros(1), max_iterations=0).converged
    with pytest.raises(ValueError):
        newton_cached(c, force, theta=0.0)
    with pytest.raises(ValueError):
        newton_cached(c, force[:-1])
