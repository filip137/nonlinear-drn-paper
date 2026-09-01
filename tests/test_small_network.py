from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from repro import (
    DEFAULT_SHOCKLEY_PARAMETERS,
    SmallNetworkResult,
    simulate_small_network,
)


ROOT = Path(__file__).resolve().parents[1]
LAYER_SIZES = [2, 4, 2]
CONDUCTANCES = [
    np.array(
        [
            [1.0, 0.2, 0.8, 0.4],
            [0.3, 1.1, 0.5, 0.9],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.8, 0.2],
            [0.4, 1.0],
            [1.1, 0.3],
            [0.2, 0.7],
        ],
        dtype=np.float32,
    ),
]
INPUT_VOLTAGES = np.array(
    [
        [0.2, -0.1],
        [0.7, 0.3],
        [-0.4, 0.8],
    ],
    dtype=np.float32,
)


def _simulate(**overrides) -> SmallNetworkResult:
    arguments = {
        "layer_sizes": LAYER_SIZES,
        "conductances": CONDUCTANCES,
        "input_voltages": INPUT_VOLTAGES,
        "non_linearity": "double",
    }
    arguments.update(overrides)
    return simulate_small_network(**arguments)


@pytest.mark.parametrize("non_linearity", ["single", "double", "pwl"])
def test_small_network_supports_each_paper_nonlinearity(
    non_linearity: str,
) -> None:
    result = _simulate(non_linearity=non_linearity)

    assert isinstance(result, SmallNetworkResult)
    assert result.converged is True
    assert 1 <= result.sweeps <= 128
    assert len(result.hidden_voltages) == 1
    assert result.hidden_voltages[0].shape == (3, 4)
    assert result.output_voltages.shape == (3, 2)
    assert np.isfinite(result.hidden_voltages[0]).all()
    assert np.isfinite(result.output_voltages).all()


def test_input_voltages_are_physical_nodes_without_channel_duplication() -> None:
    result = simulate_small_network(
        layer_sizes=[2, 2, 1],
        conductances=[
            [[1.0, 0.2], [0.3, 1.1]],
            [[0.8], [1.2]],
        ],
        input_voltages=[0.2, -0.1],
        non_linearity="double",
    )

    assert result.hidden_voltages[0].shape == (1, 2)
    assert result.output_voltages.shape == (1, 1)


def test_small_network_supports_multiple_hidden_layers() -> None:
    result = simulate_small_network(
        layer_sizes=[2, 2, 2, 1],
        conductances=[
            [[1.0, 0.4], [0.3, 1.2]],
            [[0.8, 0.2], [0.5, 1.0]],
            [[0.7], [1.1]],
        ],
        input_voltages=[[0.2, -0.1], [0.7, 0.3]],
        non_linearity="double",
    )

    assert result.converged is True
    assert [voltages.shape for voltages in result.hidden_voltages] == [
        (2, 2),
        (2, 2),
    ]
    assert result.output_voltages.shape == (2, 1)


def test_fixed_equilibrium_runs_all_default_sweeps() -> None:
    result = _simulate(adaptive_equilibrium=False)

    assert result.converged is True
    assert result.sweeps == 128
    assert result.final_max_voltage_change <= 1e-6


def test_batched_and_individual_inputs_match_in_fixed_mode() -> None:
    batched = _simulate(adaptive_equilibrium=False)
    individual = [
        _simulate(input_voltages=row, adaptive_equilibrium=False)
        for row in INPUT_VOLTAGES
    ]

    expected_hidden = np.concatenate(
        [result.hidden_voltages[0] for result in individual], axis=0
    )
    expected_output = np.concatenate(
        [result.output_voltages for result in individual], axis=0
    )
    np.testing.assert_allclose(
        batched.hidden_voltages[0], expected_hidden, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(
        batched.output_voltages, expected_output, rtol=2e-7, atol=2e-7
    )


def test_linear_output_is_the_amplified_conductance_weighted_average() -> None:
    result = _simulate(adaptive_equilibrium=False)
    output_conductances = CONDUCTANCES[-1]
    expected = 4.0 * (
        result.hidden_voltages[0] @ output_conductances
    ) / output_conductances.sum(axis=0)

    np.testing.assert_allclose(result.output_voltages, expected, rtol=1e-6, atol=1e-6)


def test_custom_shockley_dictionary_changes_voltages_without_mutating_default() -> None:
    default_before = dict(DEFAULT_SHOCKLEY_PARAMETERS)
    default_result = _simulate()
    custom_result = _simulate(
        shockley_parameters={"I_s": 1e-2, "V_t": 0.05, "V_off": 0.8}
    )

    assert not np.allclose(
        default_result.output_voltages,
        custom_result.output_voltages,
    )
    assert dict(DEFAULT_SHOCKLEY_PARAMETERS) == default_before


def test_sweep_cap_warns_and_returns_latest_voltages() -> None:
    with pytest.warns(RuntimeWarning, match="did not satisfy"):
        result = _simulate(max_sweeps=1)

    assert result.converged is False
    assert result.sweeps == 1
    assert np.isfinite(result.output_voltages).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_sizes": [2, 3, 2], "non_linearity": "single"}, "even"),
        ({"conductances": [np.ones((2, 3)), CONDUCTANCES[1]]}, "shape"),
        (
            {
                "conductances": [
                    np.array([[-1.0, 0.2, 0.8, 0.4], [0.3, 1.1, 0.5, 0.9]]),
                    CONDUCTANCES[1],
                ]
            },
            "non-negative",
        ),
        ({"input_voltages": [0.2]}, "input_voltages"),
        ({"shockley_parameters": {"I_s": 1e-6, "V_t": 0.05}}, "exactly"),
        ({"adaptive_equilibrium": 1}, "boolean"),
        ({"max_sweeps": 0}, "positive integer"),
        ({"relative_tolerance": 0.0}, "finite and positive"),
    ],
)
def test_small_network_rejects_invalid_arguments(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _simulate(**overrides)


def test_small_network_rejects_isolated_free_nodes() -> None:
    isolated_output = CONDUCTANCES[1].copy()
    isolated_output[:, 1] = 0.0

    with pytest.raises(ValueError, match="isolated node"):
        _simulate(conductances=[CONDUCTANCES[0], isolated_output])


def test_small_network_rejects_missing_pwl_curve(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="existing .npz"):
        _simulate(non_linearity="pwl", iv_data_path=tmp_path / "missing.npz")


def test_editable_example_runs_without_writing_files(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "small_network.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "hidden_voltages[1]:" in completed.stdout
    assert "output_voltages:" in completed.stdout
    assert "converged: True" in completed.stdout
    assert list(tmp_path.iterdir()) == []
