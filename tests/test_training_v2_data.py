from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from repro.strict_config import load_validated_json


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "configs" / "train"
SIMULATOR_DIR = ROOT / "configs" / "simulator"

TRAINING_FILES = (
    "default_custom_iv.json",
    "default_double_shockley.json",
    "default_mnist_custom_iv.json",
    "default_mnist_double_shockley.json",
    "default_mnist_single_shockley.json",
    "default_single_shockley.json",
    "digits_double_shockley.json",
    "digits_pwl.json",
    "digits_single_shockley.json",
    "mnist_paper_double_shockley.json",
)
SIMULATOR_FILES = (
    "default_double_shockley.json",
    "default_mnist_double_shockley.json",
    "default_pwl.json",
    "default_single_shockley.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SIMULATOR_FILES)
def test_simulator_profile_satisfies_strict_v2_schema(name: str) -> None:
    document = load_validated_json(
        SIMULATOR_DIR / name,
        "simulator-v2.schema.json",
        repo_root=ROOT,
    )
    assert set(document) == {
        "$schema",
        "schema_version",
        "kind",
        "description",
        "simulation",
        "provenance",
    }
    assert document["schema_version"] == 2
    assert document["kind"] == "simulator_profile"


@pytest.mark.parametrize("name", TRAINING_FILES)
def test_training_source_satisfies_strict_v2_schema(name: str) -> None:
    document = load_validated_json(
        TRAIN_DIR / name,
        "training-v2.schema.json",
        repo_root=ROOT,
    )
    assert document["schema_version"] == 2
    assert document["kind"] == "training"
    assert "simulation" not in document
    assert "execution" not in document
    assert set(document["simulation_ref"]) == {"path", "sha256"}
    assert set(document["execution_ref"]) == {"path", "sha256"}


@pytest.mark.parametrize("name", TRAINING_FILES)
def test_training_references_name_exact_file_bytes(name: str) -> None:
    document = _load(TRAIN_DIR / name)
    for field in ("simulation_ref", "execution_ref"):
        reference = document[field]
        target = ROOT / reference["path"]
        assert target.is_file()
        assert reference["sha256"] == _sha256(target)


def test_piecewise_linear_curve_reference_names_exact_asset_bytes() -> None:
    profile = _load(SIMULATOR_DIR / "default_pwl.json")
    reference = profile["simulation"]["updater"]["curve"]
    target = ROOT / reference["path"]
    assert reference == {
        "path": "data/assets/experimental_curve_voff_0.8_200_points.npz",
        "sha256": "3d7dd78182fc399fecdf1f6a99a3b9b71ffde131fd32e1bd62bde2689be136ea",
    }
    assert _sha256(target) == reference["sha256"]


EXPECTED_TRAINING = {
    "default_custom_iv.json": {
        "data": "sklearn_digits",
        "shapes": [[128], [32], [20]],
        "input_gain": 10.0,
        "batch": 10,
        "epochs": 15,
        "rates": [0.01, 0.01, 0.01],
        "gamma": 1.0,
        "nudging": 0.1,
        "simulation": "default_pwl.json",
    },
    "default_double_shockley.json": {
        "data": "sklearn_digits",
        "shapes": [[128], [32], [20]],
        "input_gain": 10.0,
        "batch": 10,
        "epochs": 15,
        "rates": [0.01, 0.01, 0.01],
        "gamma": 1.0,
        "nudging": 0.1,
        "simulation": "default_double_shockley.json",
    },
    "default_mnist_custom_iv.json": {
        "data": "torchvision_mnist",
        "shapes": [[2, 28, 28], [100], [20]],
        "input_gain": 10.0,
        "batch": 10,
        "epochs": 15,
        "rates": [0.01, 0.01, 0.01],
        "gamma": 1.0,
        "nudging": 0.1,
        "simulation": "default_pwl.json",
    },
    "default_mnist_double_shockley.json": {
        "data": "torchvision_mnist",
        "shapes": [[2, 28, 28], [100], [20]],
        "input_gain": 50.0,
        "batch": 10,
        "epochs": 100,
        "rates": [0.15, 0.08, 0.05],
        "gamma": 0.99,
        "nudging": 0.05,
        "simulation": "default_mnist_double_shockley.json",
    },
    "default_mnist_single_shockley.json": {
        "data": "torchvision_mnist",
        "shapes": [[2, 28, 28], [100], [10]],
        "input_gain": 10.0,
        "batch": 10,
        "epochs": 15,
        "rates": [0.005, 0.005, 0.0],
        "gamma": 1.0,
        "nudging": 1.0,
        "simulation": "default_single_shockley.json",
    },
    "default_single_shockley.json": {
        "data": "sklearn_digits",
        "shapes": [[128], [32], [10]],
        "input_gain": 10.0,
        "batch": 10,
        "epochs": 15,
        "rates": [0.005, 0.005, 0.0],
        "gamma": 1.0,
        "nudging": 1.0,
        "simulation": "default_single_shockley.json",
    },
    "digits_double_shockley.json": {
        "data": "sklearn_digits",
        "shapes": [[128], [32], [20]],
        "input_gain": 10.0,
        "batch": 32,
        "epochs": 15,
        "rates": [0.01, 0.01, 0.01],
        "gamma": 1.0,
        "nudging": 0.1,
        "simulation": "default_double_shockley.json",
    },
    "digits_pwl.json": {
        "data": "sklearn_digits",
        "shapes": [[128], [32], [20]],
        "input_gain": 10.0,
        "batch": 32,
        "epochs": 15,
        "rates": [0.01, 0.01, 0.01],
        "gamma": 1.0,
        "nudging": 0.1,
        "simulation": "default_pwl.json",
    },
    "digits_single_shockley.json": {
        "data": "sklearn_digits",
        "shapes": [[128], [64], [10]],
        "input_gain": 10.0,
        "batch": 10,
        "epochs": 15,
        "rates": [0.005, 0.005, 0.0],
        "gamma": 1.0,
        "nudging": 1.0,
        "simulation": "default_single_shockley.json",
    },
    "mnist_paper_double_shockley.json": {
        "data": "torchvision_mnist",
        "shapes": [[2, 28, 28], [100], [20]],
        "input_gain": 50.0,
        "batch": 16,
        "epochs": 100,
        "rates": [0.15, 0.08, 0.05],
        "gamma": 0.99,
        "nudging": 0.05,
        "simulation": "default_mnist_double_shockley.json",
    },
}


@pytest.mark.parametrize("name", TRAINING_FILES)
def test_training_v2_preserves_captured_effective_values(name: str) -> None:
    document = _load(TRAIN_DIR / name)
    expected = EXPECTED_TRAINING[name]
    model = document["model"]
    training = document["training"]

    assert document["data"]["source"] == expected["data"]
    assert model["layer_shapes"] == expected["shapes"]
    assert model["input_gain"] == expected["input_gain"]
    assert training["loader"]["batch_size"] == expected["batch"]
    assert training["epochs"] == expected["epochs"]
    assert training["optimizer"]["learning_rates"] == expected["rates"]
    assert training["scheduler"]["gamma"] == expected["gamma"]
    assert training["equilibrium_propagation"] == {
        "gradient_formula": "standard",
        "nudging": expected["nudging"],
        "nudging_mode": "cost",
        "variant": "centered",
    }
    assert document["equilibrium"] == {
        "initial_state": "zeros",
        "method": "fixed_sweeps",
        "sweeps": 4,
        "update_order": "asynchronous",
    }
    assert Path(document["simulation_ref"]["path"]).name == expected["simulation"]


@pytest.mark.parametrize("name", TRAINING_FILES)
def test_training_v2_materializes_previously_implicit_policy(name: str) -> None:
    document = _load(TRAIN_DIR / name)
    model = document["model"]
    training = document["training"]

    assert training["batch_limits"] == {"train": None, "evaluation": None}
    assert model["state_dtype"] == "float32"
    assert model["topology"] == "dense"
    assert model["input_encoding"] == "signed_pair"
    assert model["amplification_learning"] == {
        "current_factor": False,
        "input_gain": False,
        "voltage_factor": False,
    }
    assert model["bias"] == {
        "bounds": {"maximum": None, "minimum": 0.0},
        "enabled": True,
        "initialization": {"method": "constant", "value": 0.0},
        "interaction": "linear",
        "scale_mode": "legacy",
    }
    assert model["signed_weights"] is False
    assert model["weight_bounds"] == {"minimum": 1e-7, "maximum": 100.0}
    assert model["weight_initialization"] == "kaiming_uniform"
    assert model["loss"] == {"method": "squared_error", "reduction": "mean"}
    assert training["loader"] | {"batch_size": 0} == {
        "batch_size": 0,
        "train_shuffle": True,
        "evaluation_shuffle": False,
        "drop_last": False,
    }
    assert training["optimizer"] | {"learning_rates": []} == {
        "dampening": 0.0,
        "differentiable": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "method": "sgd",
        "learning_rates": [],
        "momentum": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "zero_grad_set_to_none": True,
    }
    assert training["scheduler"] | {"gamma": 0.0} == {
        "gamma": 0.0,
        "initial_epoch": -1,
        "method": "exponential",
        "step_timing": "after_epoch",
    }
    assert training["checkpoint"] == {
        "save_final": True,
        "save_best": {
            "metric": "evaluation_accuracy",
            "mode": "max",
            "tie_break": "first",
        },
    }
    assert document["execution_ref"]["path"] == (
        "configs/execution/reference_cpu.json"
    )

    data = document["data"]
    assert data["seed"] == 0
    assert data["subset"] == {"method": "all"}
    if data["source"] == "sklearn_digits":
        assert data["preprocessing"] == {
            "dtype": "float32",
            "method": "affine",
            "divisor": 16.0,
            "multiplier": 2.0,
            "offset": -1.0,
        }
        assert data["split"] == {
            "method": "seeded_random",
            "rounding": "floor",
            "train_fraction": 0.8,
        }
    else:
        assert data["path"] == "data/external/mnist"
        assert data["preprocessing"] == {
            "dtype": "float32",
            "method": "normalized_tensor",
            "mean": 0.1307,
            "standard_deviation": 0.3081,
            "scale": 0.3,
        }
        assert data["split"] == {"method": "official"}


def test_simulator_profiles_materialize_only_active_family_settings() -> None:
    single = _load(SIMULATOR_DIR / "default_single_shockley.json")["simulation"]
    double = _load(SIMULATOR_DIR / "default_double_shockley.json")["simulation"]
    mnist_double = _load(
        SIMULATOR_DIR / "default_mnist_double_shockley.json"
    )["simulation"]
    pwl = _load(SIMULATOR_DIR / "default_pwl.json")["simulation"]

    shared_shockley_keys = {
        "method",
        "backend",
        "dtype",
        "relaxation",
        "linear_coefficient_clamp",
        "exponent_clip",
        "asymptotic_threshold",
        "asymptotic_terms",
        "polish",
    }
    assert set(single["updater"]) == shared_shockley_keys | {
        "quadratic_coefficient_min"
    }
    assert set(double["updater"]) == shared_shockley_keys
    assert set(mnist_double["updater"]) == shared_shockley_keys
    assert set(pwl["updater"]) == {
        "method",
        "curve",
            "extrapolation",
            "nonconvergence_policy",
        "damping",
        "max_steps",
        "voltage_tolerance",
        "relaxation",
    }

    assert single["updater"] == {
        "method": "lambert_w_v1",
        "backend": "torchlambertw",
        "dtype": "float64",
        "relaxation": 1.2,
        "linear_coefficient_clamp": 1e6,
        "quadratic_coefficient_min": 1e-30,
        "exponent_clip": 100.0,
        "asymptotic_threshold": 1e10,
        "asymptotic_terms": 4,
        "polish": False,
    }
    assert double["updater"]["dtype"] == "float64"
    assert double["updater"]["relaxation"] == 1.2
    assert double["updater"]["exponent_clip"] == 165.0
    assert mnist_double["updater"]["relaxation"] == 1.0
    assert pwl["updater"] | {"curve": {}} == {
        "method": "piecewise_linear_newton_v1",
        "curve": {},
            "extrapolation": "clamp",
            "nonconvergence_policy": "accept_last",
            "damping": 1.0,
        "max_steps": 32,
        "voltage_tolerance": 0.01,
        "relaxation": 1.0,
    }


def test_v2_sources_contain_no_flat_legacy_scientific_fields() -> None:
    forbidden = {
        "non_linearity",
        "minimizer_impl",
        "single_diode_updater",
        "double_diode_updater",
        "overrelaxation_factor",
        "adaptive_equilibrium",
        "rel_tol",
        "vn_tol",
        "use_polish",
        "max_newton_iters",
        "z_thresh",
        "exp_clip",
        "damping",
        "experimental_newton_max_steps",
        "experimental_newton_tol",
        "simulator_profile",
    }
    for directory, names in (
        (TRAIN_DIR, TRAINING_FILES),
        (SIMULATOR_DIR, SIMULATOR_FILES),
    ):
        for name in names:
            assert forbidden.isdisjoint(_load(directory / name))
