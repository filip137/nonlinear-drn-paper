from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from repro import make_input_voltage_sweep, simulate_small_network
from repro.small_network import SmallNetworkResult, load_small_network_config


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "small_network" / "example.json"


def _config(family: str = "double_diode_exponential") -> dict:
    document = load_small_network_config(EXAMPLE, repo_root=ROOT)
    profile_name = {
        "single_diode_exponential": "default_single_shockley.json",
        "double_diode_exponential": "default_double_shockley.json",
        "experimental": "default_pwl.json",
    }[family]
    profile = json.loads(
        (ROOT / "configs" / "simulator" / profile_name).read_text(encoding="utf-8")
    )
    document["simulation"] = profile["simulation"]
    return document


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (("float32", np.float32), ("float64", np.float64)),
)
def test_input_voltage_sweep_builds_cartesian_grid_with_explicit_dtype(
    dtype: str,
    numpy_dtype: type[np.generic],
) -> None:
    sweep = make_input_voltage_sweep(
        2,
        voltage_min=-1.0,
        voltage_max=1.0,
        num_points=3,
        dtype=dtype,
    )
    assert sweep.dtype == numpy_dtype
    np.testing.assert_array_equal(
        sweep,
        np.array(
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=numpy_dtype,
        ),
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "input_size": 0,
                "voltage_min": -1,
                "voltage_max": 1,
                "num_points": 3,
                "dtype": "float32",
            },
            "positive integer",
        ),
        (
            {
                "input_size": 2,
                "voltage_min": -1,
                "voltage_max": 1,
                "num_points": 1,
                "dtype": "float32",
            },
            "at least 2",
        ),
        (
            {
                "input_size": 2,
                "voltage_min": 1,
                "voltage_max": 1,
                "num_points": 3,
                "dtype": "float32",
            },
            "greater than",
        ),
        (
            {
                "input_size": 2,
                "voltage_min": float("nan"),
                "voltage_max": 1,
                "num_points": 3,
                "dtype": "float32",
            },
            "finite",
        ),
        (
            {
                "input_size": 4,
                "voltage_min": -1,
                "voltage_max": 1,
                "num_points": 100,
                "dtype": "float32",
            },
            "too large",
        ),
        (
            {
                "input_size": 2,
                "voltage_min": -1,
                "voltage_max": 1,
                "num_points": 3,
                "dtype": "float16",
            },
            "dtype",
        ),
    ],
)
def test_input_voltage_sweep_rejects_invalid_requests(arguments, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_input_voltage_sweep(**arguments)


@pytest.mark.parametrize(
    "family",
    [
        "single_diode_exponential",
        "double_diode_exponential",
        "experimental",
    ],
)
def test_complete_config_supports_each_paper_family(family: str) -> None:
    result = simulate_small_network(_config(family), repo_root=ROOT)
    assert isinstance(result, SmallNetworkResult)
    assert result.converged is True
    assert 1 <= result.sweeps <= 128
    assert result.hidden_voltages[0].shape == (3, 4)
    assert result.output_voltages.shape == (3, 2)
    assert result.receipt["resolved_config_sha256"]
    assert result.receipt["execution"]["name"] == "reference_cpu"


def test_config_path_and_object_are_numerically_identical() -> None:
    from_path = simulate_small_network(EXAMPLE, repo_root=ROOT)
    from_object = simulate_small_network(_config(), repo_root=ROOT)
    np.testing.assert_array_equal(
        from_path.output_voltages,
        from_object.output_voltages,
    )
    assert from_path.receipt["source_documents"][0]["owner"] == "small_network"
    assert (
        from_path.receipt["source_documents"][0]["path"]
        == "configs/small_network/example.json"
    )
    assert len(from_path.receipt["source_documents"][0]["sha256"]) == 64
    assert from_object.receipt["source_documents"] == []


def test_generated_input_rows_are_part_of_the_config() -> None:
    document = _config()
    document["network"]["input_voltages"] = make_input_voltage_sweep(
        2,
        voltage_min=-1,
        voltage_max=1,
        num_points=3,
        dtype="float32",
    ).tolist()
    result = simulate_small_network(document, repo_root=ROOT)
    assert result.output_voltages.shape == (9, 2)


def test_fixed_equilibrium_runs_exact_configured_sweeps() -> None:
    document = _config()
    document["equilibrium"] = {
        "initial_state": "zeros",
        "method": "fixed_sweeps",
        "update_order": "asynchronous",
        "sweeps": 7,
    }
    result = simulate_small_network(document, repo_root=ROOT)
    assert result.sweeps == 7
    assert result.converged is None


def test_physical_parameter_change_is_explicit_and_effective() -> None:
    baseline = simulate_small_network(_config(), repo_root=ROOT)
    changed = _config()
    changed["simulation"]["physical"]["saturation_current"] = 1e-2
    result = simulate_small_network(changed, repo_root=ROOT)
    assert not np.allclose(baseline.output_voltages, result.output_voltages)


def test_input_gain_is_explicit_and_effective() -> None:
    baseline = simulate_small_network(_config(), repo_root=ROOT)
    changed = _config()
    changed["network"]["input_gain"] = 2.0
    result = simulate_small_network(changed, repo_root=ROOT)
    assert not np.array_equal(baseline.output_voltages, result.output_voltages)


def test_hostile_scientific_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = simulate_small_network(_config(), repo_root=ROOT)
    monkeypatch.setenv("DRN_B_CLAMP", "2")
    monkeypatch.setenv("LABS_IV_EXTRAPOLATION", "linear")
    monkeypatch.setenv("LABS_IV_CURVE_PATH", "/does/not/exist.npz")
    hostile = simulate_small_network(_config(), repo_root=ROOT)
    np.testing.assert_array_equal(
        baseline.output_voltages,
        hostile.output_voltages,
    )


def test_schema_rejects_aliases_and_unknown_fields() -> None:
    document = _config()
    document["simulation"]["nonlinearity"] = "double"
    with pytest.raises(ValueError, match="nonlinearity"):
        simulate_small_network(document, repo_root=ROOT)
    document = _config()
    document["simulation"]["updater"]["overrelated"] = True
    with pytest.raises(ValueError, match="updater"):
        simulate_small_network(document, repo_root=ROOT)


def test_runtime_relational_checks_reject_bad_matrix_shape() -> None:
    document = _config()
    document["network"]["conductances"][0][0].pop()
    with pytest.raises(ValueError, match="shape"):
        simulate_small_network(document, repo_root=ROOT)


def test_missing_pwl_curve_fails_before_science() -> None:
    document = _config("experimental")
    document["simulation"]["updater"]["curve"] = {
        "path": "missing.npz",
        "sha256": "0" * 64,
    }
    with pytest.raises(FileNotFoundError, match="configured scientific asset"):
        simulate_small_network(document, repo_root=ROOT)


def test_legacy_granular_api_is_not_accepted() -> None:
    with pytest.raises(TypeError):
        simulate_small_network(  # type: ignore[call-arg]
            layer_sizes=[2, 2, 1],
            conductances=[[[1, 1], [1, 1]], [[1], [1]]],
            input_voltages=[[0, 0]],
            non_linearity="double",
        )


def test_example_script_default_is_read_only(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "small_network.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "input_voltage_sets: 3" in completed.stdout
    assert "converged: True" in completed.stdout
    assert "config_sha256:" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_example_script_writes_complete_generated_sweep_config(tmp_path: Path) -> None:
    generated = tmp_path / "sweep.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "small_network.py"),
            "--sweep-inputs",
            "--sweep-min",
            "-0.5",
            "--sweep-max",
            "0.5",
            "--sweep-points",
            "3",
            "--write-config",
            str(generated),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "input_voltage_sets: 9" in completed.stdout
    payload = json.loads(generated.read_text())
    assert payload["schema_version"] == 2
    assert len(payload["network"]["input_voltages"]) == 9
    assert payload["provenance"]["generation_overrides"][0]["pointer"] == (
        "/network/input_voltages"
    )
