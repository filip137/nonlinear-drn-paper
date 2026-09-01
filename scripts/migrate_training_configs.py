#!/usr/bin/env python3
"""Migrate the bundled training and simulator presets to strict config v2.

The migration is deterministic and idempotent: simulator profiles are written
first, their exact file hashes are embedded in each training source, and every
result is validated before the command succeeds.  The script intentionally
only handles the repository's bundled presets; it is not a compatibility
loader for scientific execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repro" / "vendor"))

from repro.strict_config import load_json, load_validated_json, validate_document


TRAINING_SCHEMA = (
    "https://raw.githubusercontent.com/filip137/nonlinear-drn-paper/main/"
    "configs/schema/training-v2.schema.json"
)
SIMULATOR_SCHEMA = (
    "https://raw.githubusercontent.com/filip137/nonlinear-drn-paper/main/"
    "configs/schema/simulator-v2.schema.json"
)
EXECUTION_SCHEMA = "execution-v2.schema.json"
TRAINING_SCHEMA_FILE = "training-v2.schema.json"
SIMULATOR_SCHEMA_FILE = "simulator-v2.schema.json"
MIGRATION_TOOL = "scripts/migrate_training_configs.py"
SHOCKLEY_MATERIALIZED_DEFAULTS = [
    "simulation.updater.backend",
    "simulation.updater.linear_coefficient_clamp",
    "simulation.updater.asymptotic_terms",
]
TRAINING_MATERIALIZED_DEFAULTS = [
    "data.preprocessing.dtype",
    "data.preprocessing.method",
    "data.split.method",
    "data.subset.method",
    "equilibrium.initial_state",
    "model.amplification_learning",
    "model.bias.bounds",
    "model.bias.enabled",
    "model.bias.initialization",
    "model.bias.interaction",
    "model.bias.scale_mode",
    "model.input_encoding",
    "model.loss",
    "model.output",
    "model.signed_weights",
    "model.state_dtype",
    "model.topology",
    "training.batch_limits",
    "training.checkpoint",
    "training.equilibrium_propagation.gradient_formula",
    "training.equilibrium_propagation.nudging_mode",
    "training.loader.drop_last",
    "training.loader.evaluation_shuffle",
    "training.loader.train_shuffle",
    "training.optimizer.dampening",
    "training.optimizer.differentiable",
    "training.optimizer.foreach",
    "training.optimizer.fused",
    "training.optimizer.maximize",
    "training.optimizer.method",
    "training.optimizer.momentum",
    "training.optimizer.nesterov",
    "training.optimizer.weight_decay",
    "training.optimizer.zero_grad_set_to_none",
    "training.scheduler.initial_epoch",
    "training.scheduler.method",
    "training.scheduler.step_timing",
]


def _simulator_materialized_defaults(family: str) -> list[str]:
    if family == "experimental":
        return [
            "simulation.updater.extrapolation",
            "simulation.updater.nonconvergence_policy",
        ]
    defaults = list(SHOCKLEY_MATERIALIZED_DEFAULTS)
    if family == "single_diode_exponential":
        defaults.append("simulation.updater.quadratic_coefficient_min")
    return defaults


def _training_materialized_defaults(document: dict[str, Any]) -> list[str]:
    defaults = list(TRAINING_MATERIALIZED_DEFAULTS)
    if document["data"]["source"] == "sklearn_digits":
        defaults.extend(
            (
                "data.preprocessing.divisor",
                "data.preprocessing.multiplier",
                "data.preprocessing.offset",
                "data.split.rounding",
            )
        )
    return defaults

SIMULATOR_FILES = (
    "default_double_shockley.json",
    "default_mnist_double_shockley.json",
    "default_pwl.json",
    "default_single_shockley.json",
)
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

TRAINING_SIMULATOR = {
    "default_custom_iv.json": "configs/simulator/default_pwl.json",
    "default_double_shockley.json": "configs/simulator/default_double_shockley.json",
    "default_mnist_custom_iv.json": "configs/simulator/default_pwl.json",
    "default_mnist_double_shockley.json": (
        "configs/simulator/default_mnist_double_shockley.json"
    ),
    "default_mnist_single_shockley.json": (
        "configs/simulator/default_single_shockley.json"
    ),
    "default_single_shockley.json": "configs/simulator/default_single_shockley.json",
    "digits_double_shockley.json": "configs/simulator/default_double_shockley.json",
    "digits_pwl.json": "configs/simulator/default_pwl.json",
    "digits_single_shockley.json": "configs/simulator/default_single_shockley.json",
    "mnist_paper_double_shockley.json": (
        "configs/simulator/default_mnist_double_shockley.json"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Reference target does not exist: {relative}.")
    return {"path": relative, "sha256": _sha256(path)}


def _provenance(legacy: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "migration": {
            "tool": MIGRATION_TOOL,
            "from_schema_version": 1,
            "to_schema_version": 2,
        }
    }
    source = legacy.get("source")
    if isinstance(source, str) and source.strip():
        if source == (
            "Derived from the validated Digits measured/PWL anchor; replace "
            "iv_data_path or pass --iv-data-path for a new curve."
        ):
            source = (
                "Derived from the validated Digits measured/PWL anchor; select a "
                "different hashed simulator profile to use another curve."
            )
        result["source"] = source
    expected = legacy.get("expected_results")
    if isinstance(expected, dict):
        result["expected_results"] = expected
    return result


def _stamp_migration_provenance(
    document: dict[str, Any], *, materialized_defaults: list[str]
) -> None:
    for dotted_path in materialized_defaults:
        value: Any = document
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(
                    "Migration provenance names a field absent from the "
                    f"document: {dotted_path}."
                )
            value = value[part]
    provenance = document.setdefault("provenance", {})
    provenance["migration"] = {
        "tool": MIGRATION_TOOL,
        "from_schema_version": 1,
        "to_schema_version": 2,
    }
    provenance["materialized_defaults"] = sorted(set(materialized_defaults))


def _shockley_updater(
    *,
    family: str,
    dtype: str,
    relaxation: float,
    exponent_clip: float,
    asymptotic_threshold: float,
) -> dict[str, Any]:
    updater: dict[str, Any] = {
        "asymptotic_terms": 4,
        "asymptotic_threshold": asymptotic_threshold,
        "backend": "torchlambertw",
        "dtype": dtype,
        "exponent_clip": exponent_clip,
        "linear_coefficient_clamp": 1e6,
        "method": "lambert_w_v1",
        "polish": False,
        "relaxation": relaxation,
    }
    if family == "single_diode_exponential":
        updater["quadratic_coefficient_min"] = 1e-30
    return updater


def _migrate_simulator(root: Path, path: Path) -> dict[str, Any]:
    legacy = _read_json(path)
    if legacy.get("schema_version") == 2:
        if legacy["simulation"]["nonlinearity"] == "experimental":
            legacy["simulation"]["updater"]["nonconvergence_policy"] = "accept_last"
        defaults = _simulator_materialized_defaults(
            legacy["simulation"]["nonlinearity"]
        )
        _stamp_migration_provenance(legacy, materialized_defaults=defaults)
        validate_document(legacy, SIMULATOR_SCHEMA_FILE, repo_root=root)
        return legacy

    family = legacy["non_linearity"]
    amplification = {
        "current_factor": legacy["current_amp"],
        "voltage_factor": legacy["voltage_amp"],
    }
    if family == "experimental":
        curve_path = str(legacy["iv_data_path"])
        simulation = {
            "amplification": amplification,
            "nonlinearity": family,
            "physical": {"representation": "measured_piecewise_linear"},
            "updater": {
                "curve": _reference(root, curve_path),
                "damping": legacy["damping"],
                "extrapolation": "clamp",
                "max_steps": legacy["experimental_newton_max_steps"],
                "method": "piecewise_linear_newton_v1",
                "nonconvergence_policy": "accept_last",
                "relaxation": (
                    legacy["overrelaxation_factor"]
                    if legacy["double_diode_updater"] == "overrelaxed"
                    else 1.0
                ),
                "voltage_tolerance": legacy["experimental_newton_tol"],
            },
        }
    else:
        diode = legacy["exponential_diode_param"]
        if family == "single_diode_exponential":
            updater_name = legacy["single_diode_updater"]
            dtype = "float64"
            relaxation = (
                legacy["overrelaxation_factor"]
                if updater_name == "overrelaxed"
                else 1.0
            )
        else:
            updater_name = legacy["double_diode_updater"]
            dtype = "float32" if updater_name.startswith("float32") else "float64"
            relaxation = (
                legacy["overrelaxation_factor"]
                if updater_name.endswith("_overrelaxed")
                else 1.0
            )
        simulation = {
            "amplification": amplification,
            "nonlinearity": family,
            "physical": {
                "offset_voltage": diode["V_off"],
                "saturation_current": diode["I_s"],
                "thermal_voltage": diode["V_t"],
            },
            "updater": _shockley_updater(
                family=family,
                dtype=dtype,
                relaxation=relaxation,
                exponent_clip=legacy["exp_clip"],
                asymptotic_threshold=legacy["z_thresh"],
            ),
        }

    migrated = {
        "$schema": SIMULATOR_SCHEMA,
        "schema_version": 2,
        "kind": "simulator_profile",
        "description": legacy["description"],
        "simulation": simulation,
        "provenance": _provenance(legacy),
    }
    defaults = _simulator_materialized_defaults(family)
    _stamp_migration_provenance(migrated, materialized_defaults=defaults)
    validate_document(migrated, SIMULATOR_SCHEMA_FILE, repo_root=root)
    return migrated


def _data(legacy: dict[str, Any]) -> dict[str, Any]:
    dataset = legacy["dataset"]
    seed = legacy["seed"]
    if dataset["name"] == "digits":
        count = dataset.get("num_samples")
        subset = (
            {"method": "all"}
            if count is None
            else {"method": "seeded_random", "count": count}
        )
        return {
            "preprocessing": {
                "divisor": 16.0,
                "dtype": "float32",
                "method": "affine",
                "multiplier": 2.0,
                "offset": -1.0,
            },
            "seed": seed,
            "source": "sklearn_digits",
            "split": {
                "method": "seeded_random",
                "train_fraction": dataset["train_fraction"],
                "rounding": "floor",
            },
            "subset": subset,
        }

    train_samples = dataset.get("train_samples")
    test_samples = dataset.get("test_samples")
    subset = (
        {"method": "all"}
        if train_samples is None and test_samples is None
        else {
            "method": "prefix",
            "train_count": train_samples,
            "evaluation_count": test_samples,
        }
    )
    if dataset.get("normalize", True):
        preprocessing = {
            "dtype": "float32",
            "mean": dataset["normalize_mean"],
            "method": "normalized_tensor",
            "scale": dataset["normalize_scale"],
            "standard_deviation": dataset["normalize_standard_deviation"],
        }
    else:
        preprocessing = {"dtype": "float32", "method": "to_tensor"}
    return {
        "path": dataset["root"],
        "preprocessing": preprocessing,
        "seed": seed,
        "source": "torchvision_mnist",
        "split": {"method": "official"},
        "subset": subset,
    }


def _migrate_training(
    root: Path,
    path: Path,
    *,
    simulation_ref: dict[str, str],
    execution_ref: dict[str, str],
) -> dict[str, Any]:
    legacy = _read_json(path)
    if legacy.get("schema_version") == 2:
        migrated = dict(legacy)
        migrated["simulation_ref"] = simulation_ref
        migrated["execution_ref"] = execution_ref
        migrated["training"] = dict(migrated["training"])
        migrated["training"]["batch_limits"] = {
            "evaluation": None,
            "train": None,
        }
        equilibrium_propagation = dict(
            migrated["training"]["equilibrium_propagation"]
        )
        equilibrium_propagation.setdefault("nudging_mode", "cost")
        equilibrium_propagation.setdefault("gradient_formula", "standard")
        migrated["training"]["equilibrium_propagation"] = equilibrium_propagation
        migrated["model"] = dict(migrated["model"])
        migrated["model"]["input_encoding"] = "signed_pair"
        migrated["model"]["topology"] = "dense"
        migrated["model"]["amplification_learning"] = {
            "input_gain": False,
            "voltage_factor": False,
            "current_factor": False,
        }
        migrated["model"]["bias"] = {
            "enabled": True,
            "scale_mode": migrated["model"]["bias"]["scale_mode"],
            "interaction": migrated["model"]["bias"]["interaction"],
            "initialization": {"method": "constant", "value": 0.0},
            "bounds": {"minimum": 0.0, "maximum": None},
        }
        if migrated["data"]["source"] == "sklearn_digits":
            migrated["data"]["split"]["rounding"] = "floor"
        migrated["equilibrium"]["initial_state"] = "zeros"
        migrated["training"]["optimizer"].update(
            {
                "dampening": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": False,
                "differentiable": False,
                "fused": False,
            }
        )
        migrated["training"]["scheduler"]["initial_epoch"] = -1
        migrated["training"]["checkpoint"]["save_best"]["tie_break"] = "first"
        _stamp_migration_provenance(
            migrated, materialized_defaults=_training_materialized_defaults(migrated)
        )
        validate_document(migrated, TRAINING_SCHEMA_FILE, repo_root=root)
        return migrated

    layer_shapes = legacy["layer_shapes"]
    output_width = 1
    for value in layer_shapes[-1]:
        output_width *= value
    if output_width == 10:
        output_encoding = "single_ended"
    elif output_width == 20:
        output_encoding = "differential_pair"
    else:
        raise ValueError(
            f"Expected output width 10 or 20 in {path}; got {output_width}."
        )

    migrated = {
        "$schema": TRAINING_SCHEMA,
        "schema_version": 2,
        "kind": "training",
        "description": legacy["description"],
        "data": _data(legacy),
        "model": {
            "bias": {
                "enabled": True,
                "interaction": "linear",
                "scale_mode": "legacy",
                "initialization": {"method": "constant", "value": 0.0},
                "bounds": {"minimum": 0.0, "maximum": None},
            },
            "input_encoding": "signed_pair",
            "topology": "dense",
            "amplification_learning": {
                "input_gain": False,
                "voltage_factor": False,
                "current_factor": False,
            },
            "input_gain": legacy["input_gain"],
            "layer_shapes": layer_shapes,
            "loss": {"method": "squared_error", "reduction": "mean"},
            "output": {"classes": 10, "encoding": output_encoding},
            "signed_weights": False,
            "state_dtype": "float32",
            "weight_bounds": {
                "maximum": legacy["weight_max"],
                "minimum": legacy["weight_min"],
            },
            "weight_gains": legacy["weight_gains"],
            "weight_initialization": legacy["weight_init_mode"],
        },
        "training": {
            "batch_limits": {"evaluation": None, "train": None},
            "checkpoint": {
                "save_best": {
                    "metric": "evaluation_accuracy",
                    "mode": "max",
                    "tie_break": "first",
                },
                "save_final": True,
            },
            "epochs": legacy["num_epochs"],
            "equilibrium_propagation": {
                "gradient_formula": "standard",
                "nudging": legacy["nudging"],
                "nudging_mode": "cost",
                "variant": legacy["ep_variant"],
            },
            "loader": {
                "batch_size": legacy["batch_size"],
                "drop_last": False,
                "evaluation_shuffle": False,
                "train_shuffle": True,
            },
            "optimizer": {
                "learning_rates": legacy["learning_rates"],
                "method": "sgd",
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
                "gamma": legacy["scheduler_gamma"],
                "method": "exponential",
                "step_timing": "after_epoch",
                "initial_epoch": -1,
            },
        },
        "equilibrium": {
            "initial_state": "zeros",
            "method": "fixed_sweeps",
            "sweeps": legacy["num_iterations"],
            # The six legacy composed templates inherited this value from their
            # simulator profiles; all ten bundled effective configs used the
            # asynchronous odd/even order captured before migration.
            "update_order": legacy.get("mode", "asynchronous"),
        },
        "simulation_ref": simulation_ref,
        "execution_ref": execution_ref,
        "provenance": _provenance(legacy),
    }
    _stamp_migration_provenance(
        migrated, materialized_defaults=_training_materialized_defaults(migrated)
    )
    validate_document(migrated, TRAINING_SCHEMA_FILE, repo_root=root)
    return migrated


def migrate(root: Path) -> None:
    root = root.expanduser().resolve()
    simulator_dir = root / "configs" / "simulator"
    training_dir = root / "configs" / "train"
    execution_path = root / "configs" / "execution" / "reference_cpu.json"

    load_validated_json(execution_path, EXECUTION_SCHEMA, repo_root=root)
    execution_ref = _reference(root, "configs/execution/reference_cpu.json")

    for name in SIMULATOR_FILES:
        path = simulator_dir / name
        _write_json(path, _migrate_simulator(root, path))

    for name in TRAINING_FILES:
        path = training_dir / name
        simulation_path = TRAINING_SIMULATOR[name]
        simulation_ref = _reference(root, simulation_path)
        _write_json(
            path,
            _migrate_training(
                root,
                path,
                simulation_ref=simulation_ref,
                execution_ref=execution_ref,
            ),
        )

    for name in SIMULATOR_FILES:
        load_validated_json(
            simulator_dir / name,
            SIMULATOR_SCHEMA_FILE,
            repo_root=root,
        )
    for name in TRAINING_FILES:
        path = training_dir / name
        load_validated_json(
            path,
            TRAINING_SCHEMA_FILE,
            repo_root=root,
        )
        # Exercise the same hash-resolution, relational, and nested-asset gate
        # used by the public training entry point before claiming success.
        from repro.train import resolve_training_config

        resolve_training_config(path, repo_root=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root containing configs/ (default: inferred from this script).",
    )
    args = parser.parse_args()
    migrate(args.repo_root)
    print("migrated and validated 4 simulator profiles and 10 training sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
