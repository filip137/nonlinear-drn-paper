from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from repro.strict_config import (
    ConfigReference,
    ConfigurationCompositionError,
    ConfigurationOverrideError,
    ConfigurationParseError,
    ConfigurationReferenceError,
    ConfigurationValidationError,
    JsonPointerOverride,
    apply_json_pointer_overrides,
    canonical_json_bytes,
    compose_owned_sections,
    document_sha256,
    file_sha256,
    load_json,
    load_validated_json,
    make_reference,
    resolve_reference,
    resolve_config_source,
    validate_document,
)


ROOT = Path(__file__).resolve().parents[1]
CPU_PROFILE = ROOT / "configs" / "execution" / "reference_cpu.json"


def _single_simulation() -> dict:
    return {
        "nonlinearity": "single_diode_exponential",
        "physical": {
            "saturation_current": 1e-6,
            "thermal_voltage": 0.05,
            "offset_voltage": 0.8,
        },
        "amplification": {"voltage_factor": 4.0, "current_factor": 1.0},
        "updater": {
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
        },
    }


def _training_source() -> dict:
    return {
        "$schema": "https://raw.githubusercontent.com/filip137/nonlinear-drn-paper/main/configs/schema/training-v2.schema.json",
        "schema_version": 2,
        "kind": "training",
        "description": "Strict test training source.",
        "data": {
            "source": "sklearn_digits",
            "preprocessing": {
                "dtype": "float32",
                "method": "affine",
                "divisor": 16.0,
                "multiplier": 2.0,
                "offset": -1.0,
            },
            "subset": {"method": "all"},
            "split": {
                "method": "seeded_random",
                "train_fraction": 0.8,
                "rounding": "floor",
            },
            "seed": 0,
        },
        "model": {
            "layer_shapes": [[128], [32], [10]],
            "state_dtype": "float32",
            "weight_gains": [1.0, 1.0],
            "weight_bounds": {"minimum": 1e-7, "maximum": 100.0},
            "weight_initialization": "kaiming_uniform",
            "input_gain": 10.0,
            "input_encoding": "signed_pair",
            "amplification_learning": {
                "input_gain": False,
                "voltage_factor": False,
                "current_factor": False,
            },
            "topology": "dense",
            "bias": {
                "enabled": True,
                "scale_mode": "legacy",
                "interaction": "linear",
                "initialization": {"method": "constant", "value": 0.0},
                "bounds": {"minimum": 0.0, "maximum": None},
            },
            "signed_weights": False,
            "output": {"encoding": "single_ended", "classes": 10},
            "loss": {"method": "squared_error", "reduction": "mean"},
        },
        "training": {
            "epochs": 15,
            "batch_limits": {"train": None, "evaluation": None},
            "loader": {
                "batch_size": 10,
                "train_shuffle": True,
                "evaluation_shuffle": False,
                "drop_last": False,
            },
            "equilibrium_propagation": {
                "variant": "centered",
                "nudging": 1.0,
                "nudging_mode": "cost",
                "gradient_formula": "standard",
            },
            "optimizer": {
                "method": "sgd",
                "learning_rates": [0.005, 0.005, 0.0],
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "zero_grad_set_to_none": True,
                "nesterov": False,
                "maximize": False,
                "foreach": False,
                "differentiable": False,
                "fused": False,
            },
            "scheduler": {
                "method": "exponential",
                "gamma": 1.0,
                "step_timing": "after_epoch",
                "initial_epoch": -1,
            },
            "checkpoint": {
                "save_final": True,
                "save_best": {
                    "metric": "evaluation_accuracy",
                    "mode": "max",
                    "tie_break": "first",
                },
            },
        },
        "equilibrium": {
            "method": "fixed_sweeps",
            "initial_state": "zeros",
            "update_order": "asynchronous",
            "sweeps": 4,
        },
        "simulation_ref": {"path": "configs/simulator/example.json", "sha256": "0" * 64},
        "execution_ref": {"path": "configs/execution/reference_cpu.json", "sha256": "1" * 64},
        "provenance": {"source": "Unit test fixture."},
    }


def test_execution_profiles_validate_at_runtime() -> None:
    for path in sorted((ROOT / "configs" / "execution").glob("*.json")):
        loaded = load_validated_json(path, "execution-v2", repo_root=ROOT)
        assert loaded["schema_version"] == 2
        assert loaded["execution"]["threads"]["torch_intraop"] == 1
        assert loaded["execution"]["dataloader"]["num_workers"] == 0


def test_training_source_schema_is_strict_and_family_neutral() -> None:
    source = _training_source()
    validate_document(source, "training-v2", repo_root=ROOT)

    unsupported_bias_alias = copy.deepcopy(source)
    unsupported_bias_alias["model"]["bias"]["scale_mode"] = "fan_in"
    with pytest.raises(ConfigurationValidationError, match="scale_mode"):
        validate_document(unsupported_bias_alias, "training-v2", repo_root=ROOT)

    source["training"]["optimizer"]["momentun"] = 0.0
    with pytest.raises(ConfigurationValidationError, match=r"optimizer.*momentun|momentun"):
        validate_document(source, "training-v2", repo_root=ROOT)


def test_family_schema_rejects_inactive_updater_fields() -> None:
    simulation = _single_simulation()
    simulation["updater"]["residual_tolerance"] = {
        "rule": "batch_scaled_absolute",
        "coefficient": 1e-6,
    }
    profile = {
        "$schema": "https://raw.githubusercontent.com/filip137/nonlinear-drn-paper/main/configs/schema/simulator-v2.schema.json",
        "schema_version": 2,
        "kind": "simulator_profile",
        "description": "Invalid cross-family fixture.",
        "simulation": simulation,
        "provenance": {},
    }

    with pytest.raises(ConfigurationValidationError, match="residual_tolerance"):
        validate_document(profile, "simulator-v2", repo_root=ROOT)


def test_wrong_scalar_types_are_not_coerced() -> None:
    source = _training_source()
    source["equilibrium"]["sweeps"] = "4"

    with pytest.raises(ConfigurationValidationError, match=r"sweeps.*integer|integer"):
        validate_document(source, "training-v2", repo_root=ROOT)


def test_canonical_serialization_and_hash_ignore_mapping_order() -> None:
    left = {"z": [3, 2, 1], "a": {"β": True, "x": 1.0}}
    right = {"a": {"x": 1.0, "β": True}, "z": [3, 2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert document_sha256(left) == document_sha256(right)
    assert canonical_json_bytes(left).decode("utf-8") == '{"a":{"x":1.0,"β":true},"z":[3,2,1]}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_serialization_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ConfigurationParseError, match="Non-finite"):
        canonical_json_bytes({"scientific_value": value})


def test_json_loader_rejects_duplicate_keys_and_constants(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed": 0, "seed": 1}', encoding="utf-8")
    with pytest.raises(ConfigurationParseError, match="duplicate.*seed"):
        load_json(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"tolerance": NaN}', encoding="utf-8")
    with pytest.raises(ConfigurationParseError, match="non-finite.*NaN"):
        load_json(non_finite)


def test_reference_requires_exact_bytes_and_stays_in_repository(tmp_path: Path) -> None:
    target = tmp_path / "profile.json"
    target.write_text('{"simulation":{"method":"test"}}\n', encoding="utf-8")
    reference = make_reference(target, repo_root=tmp_path)
    resolved = resolve_reference(reference, repo_root=tmp_path)
    assert resolved.document == {"simulation": {"method": "test"}}
    assert reference.sha256 == file_sha256(target)

    target.write_text('{"simulation":{"method":"changed"}}\n', encoding="utf-8")
    with pytest.raises(ConfigurationReferenceError, match="SHA-256 mismatch"):
        resolve_reference(reference, repo_root=tmp_path)

    with pytest.raises(ConfigurationReferenceError, match="escapes repository"):
        resolve_reference(
            ConfigReference(path="../outside.json", sha256="0" * 64),
            repo_root=tmp_path,
        )


def test_owned_composition_is_order_independent_and_rejects_collisions(
    tmp_path: Path,
) -> None:
    simulator = tmp_path / "simulator.json"
    simulator.write_text(json.dumps({"simulation": {"family": "single"}}), encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({"execution": {"device": "cpu"}}), encoding="utf-8")
    references = {
        "simulation": make_reference(simulator, repo_root=tmp_path),
        "execution": make_reference(execution, repo_root=tmp_path),
    }

    result = compose_owned_sections(
        {"schema_version": 2, "kind": "training"},
        references,
        repo_root=tmp_path,
    )
    assert result.document["simulation"] == {"family": "single"}
    assert result.document["execution"] == {"device": "cpu"}
    assert [record.owner for record in result.references] == ["execution", "simulation"]

    with pytest.raises(ConfigurationCompositionError, match="collision.*simulation"):
        compose_owned_sections(
            {"simulation": {}},
            {"simulation": references["simulation"]},
            repo_root=tmp_path,
        )


def test_high_level_source_resolution_validates_both_shapes_and_records_sources() -> None:
    source = ROOT / "configs" / "train" / "default_single_shockley.json"
    result = resolve_config_source(
        source,
        schema="training-v2",
        reference_fields={
            "simulation_ref": "simulation",
            "execution_ref": "execution",
        },
        reference_schemas={
            "simulation": "simulator-v2",
            "execution": "execution-v2",
        },
        repo_root=ROOT,
    )

    assert "simulation_ref" not in result.document
    assert "execution_ref" not in result.document
    assert result.document["simulation"]["nonlinearity"] == "single_diode_exponential"
    assert result.document["execution"]["name"] == "reference_cpu"
    assert result.document["provenance"]["config_sources"] == [
        record.as_dict() for record in result.references
    ]

    reloaded = resolve_config_source(
        result.document,
        schema="training-v2",
        reference_fields={
            "simulation_ref": "simulation",
            "execution_ref": "execution",
        },
        repo_root=ROOT,
    )
    assert reloaded.document == result.document
    assert reloaded.references == ()


def test_json_pointer_overrides_replace_only_explicit_existing_targets() -> None:
    source = {"equilibrium": {"sweeps": 16}, "model": {"layers": [128, 20]}}
    original = copy.deepcopy(source)
    updated = apply_json_pointer_overrides(
        source,
        {
            "/equilibrium/sweeps": 32,
            "/model/layers/1": 10,
        },
    )

    assert updated == {"equilibrium": {"sweeps": 32}, "model": {"layers": [128, 10]}}
    assert source == original

    with pytest.raises(ConfigurationOverrideError, match="does not exist"):
        apply_json_pointer_overrides(source, {"/equilibrium/tolerance": 1e-5})


def test_json_pointer_overrides_reject_order_dependent_parent_child_pairs() -> None:
    overrides = [
        JsonPointerOverride("/equilibrium", {"sweeps": 4}),
        JsonPointerOverride("/equilibrium/sweeps", 8),
    ]
    with pytest.raises(ConfigurationOverrideError, match="order-dependent"):
        apply_json_pointer_overrides({"equilibrium": {"sweeps": 16}}, overrides)


def test_small_network_schema_requires_complete_numeric_inputs() -> None:
    execution = load_json(CPU_PROFILE)["execution"]
    config = {
        "$schema": "https://raw.githubusercontent.com/filip137/nonlinear-drn-paper/main/configs/schema/small-network-v2.schema.json",
        "schema_version": 2,
        "kind": "small_network",
        "description": "Complete hand-specified network fixture.",
        "network": {
            "layer_sizes": [2, 2, 1],
            "conductances": [
                [[1.0, 0.2], [0.3, 1.1]],
                [[0.8], [1.2]],
            ],
            "input_voltages": [[0.2, -0.1]],
            "input_gain": 1.0,
            "input_encoding": "direct",
            "bias": {"method": "none"},
            "seed": 0,
            "state_dtype": "float32",
        },
        "simulation": _single_simulation(),
        "equilibrium": {
            "method": "voltage_change",
            "initial_state": "zeros",
            "update_order": "asynchronous",
            "max_sweeps": 128,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-6,
        },
        "execution": execution,
        "provenance": {"source": "Unit test fixture."},
    }
    validate_document(config, "small-network-v2", repo_root=ROOT)

    unsupported_precision = copy.deepcopy(config)
    unsupported_precision["simulation"]["updater"]["dtype"] = "float32"
    with pytest.raises(ConfigurationValidationError, match=r"dtype: 'float64'"):
        validate_document(
            unsupported_precision,
            "small-network-v2",
            repo_root=ROOT,
        )

    unsupported_update_order = copy.deepcopy(config)
    unsupported_update_order["equilibrium"]["update_order"] = "synchronous"
    with pytest.raises(
        ConfigurationValidationError,
        match=r"update_order: 'asynchronous' was expected",
    ):
        validate_document(
            unsupported_update_order,
            "small-network-v2",
            repo_root=ROOT,
        )

    del config["network"]["input_voltages"]
    with pytest.raises(ConfigurationValidationError, match="input_voltages"):
        validate_document(config, "small-network-v2", repo_root=ROOT)
