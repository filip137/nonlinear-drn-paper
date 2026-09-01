#!/usr/bin/env python3
"""Migrate the bundled numerical-replay catalogue to strict schema version 2.

The migration intentionally preserves the effective values of the legacy
loader, including values that used to come from Python fallbacks or the CLI
environment.  Fields that the replay path never consumed are retained only as
provenance and cannot affect execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE_PATH = Path("data/manifest.json")
MIGRATION_TOOL = "scripts/migrate_replay_configs.py"
COMPARISON_POLICY = {
    "accumulation_dtype": "float64",
    "percentile_method": "linear",
    "percentiles": [50, 60, 70, 80, 90, 95, 99],
    "policy_version": 1,
    "reference_data_fingerprint": "a4c367b1b40ae59417bb00dc2cca03ed3cdf3b686976e16ce557b89c0b7e5742",
    "reference_split_fingerprint": "02a37047fc2fa86bf100d808f89f7fa19f27e8644b74a7962a2e4fadd0289b07",
    "relative_error_epsilon": 1e-12,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _finite_number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected legacy {name} to be numeric. Provided value: {value!r}.")
    if not math.isfinite(float(value)):
        raise ValueError(f"Expected legacy {name} to be finite. Provided value: {value!r}.")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = float(_finite_number(value, name))
    if result <= 0.0:
        raise ValueError(f"Expected legacy {name} to be positive. Provided value: {value!r}.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Expected legacy {name} to be a positive integer. Provided value: {value!r}.")
    return value


def _required(payload: dict[str, Any], name: str) -> Any:
    if name not in payload or payload[name] is None:
        raise ValueError(f"Expected legacy field {name!r} to be present. Provided value: {payload.get(name)!r}.")
    return payload[name]


def _shockley_physical(payload: dict[str, Any]) -> dict[str, float]:
    diode = _required(payload, "exponential_diode_param")
    if not isinstance(diode, dict):
        raise ValueError("Expected legacy exponential_diode_param to be an object.")
    return {
        "offset_voltage": _positive_float(_required(diode, "V_off"), "exponential_diode_param.V_off"),
        "saturation_current": _positive_float(_required(diode, "I_s"), "exponential_diode_param.I_s"),
        "thermal_voltage": _positive_float(_required(diode, "V_t"), "exponential_diode_param.V_t"),
    }


def _polish(payload: dict[str, Any], *, family: str) -> bool | dict[str, Any]:
    enabled = _required(payload, "use_polish")
    if not isinstance(enabled, bool):
        raise ValueError(f"Expected legacy use_polish to be boolean. Provided value: {enabled!r}.")
    if not enabled:
        return False
    max_steps = _positive_int(_required(payload, "max_newton_iters"), "max_newton_iters")
    if family == "double_diode_exponential":
        if max_steps <= 1:
            raise ValueError(
                "Cannot preserve a legacy double-Shockley polish cap below two: "
                "the legacy loop performed max_newton_iters - 1 steps."
            )
        max_steps -= 1
    result: dict[str, Any] = {
        "method": "newton_v1",
        "max_steps": max_steps,
    }
    if family == "single_diode_exponential":
        result.update({"absolute_tolerance": 1e-6, "relative_tolerance": 1e-6})
    else:
        result["residual_tolerance"] = {
            "rule": "batch_scaled_absolute",
            "coefficient": 1e-6,
        }
    return result


def _shockley_updater(payload: dict[str, Any], family: str, consumed: set[str]) -> dict[str, Any]:
    if family == "double_diode_exponential":
        selector_field = "double_diode_updater"
        selector = _required(payload, selector_field)
        mapping = {
            "custom": ("float64", 1.0),
            "float64": ("float64", 1.0),
            "float32": ("float32", 1.0),
            "float64_overrelaxed": ("float64", None),
            "float32_overrelaxed": ("float32", None),
            "overrelaxed": ("float32", None),
            "overrelated": ("float32", None),
        }
    else:
        selector_field = "single_diode_updater"
        selector = _required(payload, selector_field)
        mapping = {
            "custom": ("float64", 1.0),
            "overrelaxed": ("float64", None),
            "overrelated": ("float64", None),
        }
    if selector not in mapping:
        raise ValueError(
            f"Cannot losslessly migrate legacy {selector_field}={selector!r}; "
            f"supported selectors are {sorted(mapping)}."
        )
    dtype, fixed_relaxation = mapping[selector]
    consumed.add(selector_field)
    if fixed_relaxation is None:
        relaxation = _positive_float(_required(payload, "overrelaxation_factor"), "overrelaxation_factor")
        consumed.add("overrelaxation_factor")
    else:
        relaxation = fixed_relaxation

    polish = _polish(payload, family=family)
    consumed.update({"use_polish", "z_thresh", "exp_clip"})
    if polish is not False:
        consumed.add("max_newton_iters")
    updater: dict[str, Any] = {
        "asymptotic_terms": 4,
        "asymptotic_threshold": _positive_float(_required(payload, "z_thresh"), "z_thresh"),
        "backend": "torchlambertw",
        "dtype": dtype,
        "exponent_clip": _positive_float(_required(payload, "exp_clip"), "exp_clip"),
        "linear_coefficient_clamp": 1e6,
        "method": "lambert_w_v1",
        "polish": polish,
        "relaxation": relaxation,
    }
    if family == "single_diode_exponential":
        updater["quadratic_coefficient_min"] = 1e-30
    return updater


def _pwl_updater(
    root: Path,
    payload: dict[str, Any],
    consumed: set[str],
    materialized_defaults: list[str],
) -> dict[str, Any]:
    curve_field = "iv_data_path" if payload.get("iv_data_path") else "LABS_IV_CURVE_PATH"
    curve_path = _required(payload, curve_field)
    if not isinstance(curve_path, str) or not curve_path:
        raise ValueError(f"Expected legacy {curve_field} to be a non-empty path string.")
    curve = Path(curve_path)
    resolved_curve = curve if curve.is_absolute() else root / curve
    if curve.is_absolute():
        raise ValueError(
            "Replay configuration contains an absolute measured-I-V path and is not portable: "
            f"{curve}."
        )
    if not resolved_curve.is_file():
        raise FileNotFoundError(f"Expected measured-I-V asset to exist: {resolved_curve}.")
    consumed.add(curve_field)

    selector = payload.get("double_diode_updater", "standard")
    if selector not in {"standard", "custom", "overrelaxed", "overrelated"}:
        raise ValueError(f"Cannot losslessly migrate measured-I-V updater {selector!r}.")
    if "double_diode_updater" in payload:
        consumed.add("double_diode_updater")
    if selector in {"overrelaxed", "overrelated"}:
        relaxation = _positive_float(_required(payload, "overrelaxation_factor"), "overrelaxation_factor")
        consumed.add("overrelaxation_factor")
    else:
        relaxation = 1.0

    if "damping" in payload:
        damping = _positive_float(payload["damping"], "damping")
        consumed.add("damping")
    else:
        damping = 0.5
        materialized_defaults.append("simulation.updater.damping")
    if "experimental_newton_max_steps" in payload:
        max_steps = _positive_int(payload["experimental_newton_max_steps"], "experimental_newton_max_steps")
        consumed.add("experimental_newton_max_steps")
    else:
        max_steps = 100
        materialized_defaults.append("simulation.updater.max_steps")
    if "experimental_newton_tol" in payload:
        voltage_tolerance = _positive_float(payload["experimental_newton_tol"], "experimental_newton_tol")
        consumed.add("experimental_newton_tol")
    else:
        voltage_tolerance = 1e-5
        materialized_defaults.append("simulation.updater.voltage_tolerance")

    return {
        "curve": {"path": curve.as_posix(), "sha256": _sha256(resolved_curve)},
        "damping": damping,
        "extrapolation": "clamp",
        "max_steps": max_steps,
        "method": "piecewise_linear_newton_v1",
        "nonconvergence_policy": "accept_last",
        "relaxation": relaxation,
        "voltage_tolerance": voltage_tolerance,
    }


def migrate_config(root: Path, relative_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the strict v2 replay config corresponding to one legacy payload."""

    if payload.get("schema_version") == 2:
        if payload.get("kind") != "replay":
            raise ValueError(f"Expected v2 replay kind in {relative_path!r}.")
        return payload

    family = _required(payload, "non_linearity")
    if family not in {"single_diode_exponential", "double_diode_exponential", "experimental"}:
        raise ValueError(f"Unsupported replay nonlinearity in {relative_path!r}: {family!r}.")
    if payload.get("minimizer_impl") != "custom":
        raise ValueError(
            f"Expected legacy minimizer_impl='custom' in {relative_path!r}. "
            f"Provided value: {payload.get('minimizer_impl')!r}."
        )

    consumed = {
        "adaptive_equilibrium",
        "batch_size",
        "bias_interaction_type",
        "bias_scale_mode",
        "current_amp",
        "dims",
        "input_gain",
        "minimizer_impl",
        "non_linearity",
        "num_iterations",
        "seed",
        "signed_weights",
        "voltage_amp",
        "weight_gains",
        "weight_max",
        "weight_min",
    }
    materialized_defaults = [
        "data.preprocessing",
        "data.seed",
        "data.split",
        "data.subset",
        "equilibrium.update_order",
        "model.state_dtype",
    ]

    dims = _required(payload, "dims")
    weight_gains = _required(payload, "weight_gains")
    if not isinstance(dims, list) or len(dims) < 2 or not all(isinstance(value, int) and value > 0 for value in dims):
        raise ValueError(f"Expected positive integer dims in {relative_path!r}. Provided value: {dims!r}.")
    if not isinstance(weight_gains, list) or len(weight_gains) != len(dims) - 1:
        raise ValueError(f"Expected one weight gain per edge in {relative_path!r}.")
    if dims[-1] == 10:
        output_encoding = "single_ended"
    elif dims[-1] == 20:
        output_encoding = "differential_pair"
    else:
        raise ValueError(f"Expected replay output width 10 or 20 in {relative_path!r}. Provided: {dims[-1]}.")

    adaptive = _required(payload, "adaptive_equilibrium")
    if not isinstance(adaptive, bool):
        raise ValueError(f"Expected legacy adaptive_equilibrium to be boolean in {relative_path!r}.")
    iterations = _positive_int(_required(payload, "num_iterations"), "num_iterations")
    if adaptive:
        equilibrium = {
            "absolute_tolerance": _positive_float(payload.get("vn_tol", 1e-6), "vn_tol"),
            "initial_state": "zeros",
            "max_sweeps": iterations,
            "method": "voltage_change",
            "relative_tolerance": _positive_float(payload.get("rel_tol", 1e-5), "rel_tol"),
            "update_order": "asynchronous",
        }
        if "vn_tol" in payload:
            consumed.add("vn_tol")
        else:
            materialized_defaults.append("equilibrium.absolute_tolerance")
        if "rel_tol" in payload:
            consumed.add("rel_tol")
        else:
            materialized_defaults.append("equilibrium.relative_tolerance")
    else:
        equilibrium = {
            "initial_state": "zeros",
            "method": "fixed_sweeps",
            "sweeps": iterations,
            "update_order": "asynchronous",
        }

    if family == "experimental":
        physical: dict[str, Any] = {"representation": "measured_piecewise_linear"}
        updater = _pwl_updater(root, payload, consumed, materialized_defaults)
    else:
        consumed.add("exponential_diode_param")
        materialized_defaults.extend(
            (
                "simulation.updater.backend",
                "simulation.updater.linear_coefficient_clamp",
            )
        )
        physical = _shockley_physical(payload)
        updater = _shockley_updater(payload, family, consumed)

    historical_unused = {
        key: value
        for key, value in sorted(payload.items())
        if key not in consumed
    }
    # Missing fields were effective legacy defaults, not missing scientific choices.
    if "seed" not in payload:
        materialized_defaults.append("data.seed")
    if "bias_scale_mode" not in payload:
        materialized_defaults.append("model.bias.scale_mode")
    if "bias_interaction_type" not in payload:
        materialized_defaults.append("model.bias.interaction")
    if "signed_weights" not in payload:
        materialized_defaults.append("model.signed_weights")
    materialized_defaults.append("model.weight_initialization")

    return {
        "data": {
            "loader": {
                "batch_size": _positive_int(payload.get("batch_size", 1), "batch_size"),
                "drop_last": False,
                "shuffle": False,
            },
            "preprocessing": {
                "input_dtype": "float32",
                "input_offset": -1.0,
                "input_scale": 0.125,
                "target_dtype": "int64",
            },
            "seed": int(payload.get("seed", 0) if payload.get("seed") is not None else 0),
            "source": {"loader": "sklearn.datasets.load_digits", "name": "digits"},
            "split": {
                "method": "random_fraction",
                "train_fraction": 0.8,
                "rounding": "floor",
            },
            "subset": {"max_points": 2000, "method": "random_max_points"},
        },
        "equilibrium": equilibrium,
        "kind": "replay",
        "model": {
            "bias": {
                "bounds": {"maximum": None, "minimum": 0.0},
                "enabled": True,
                "initialization": {"method": "constant", "value": 0.0},
                "interaction": str(payload.get("bias_interaction_type", "linear")),
                "scale_mode": str(payload.get("bias_scale_mode", "legacy")),
            },
            "input_gain": _positive_float(_required(payload, "input_gain"), "input_gain"),
            "input_encoding": "signed_pair",
            "topology": "dense",
            "amplification_learning": {
                "input_gain": False,
                "voltage_factor": False,
                "current_factor": False,
            },
            "layer_widths": dims,
            "loss": {"method": "squared_error", "reduction": "mean"},
            "output": {"classes": 10, "encoding": output_encoding},
            "signed_weights": payload.get("signed_weights", False),
            "state_dtype": "float32",
            "weight_bounds": {
                "maximum": _positive_float(_required(payload, "weight_max"), "weight_max"),
                "minimum": _positive_float(_required(payload, "weight_min"), "weight_min"),
            },
            "weight_gains": [float(_finite_number(value, "weight_gains[]")) for value in weight_gains],
            "weight_initialization": "kaiming_uniform",
        },
        "provenance": {
            "historical_unused": historical_unused,
            "materialized_defaults": sorted(set(materialized_defaults)),
            "migration": {"from": "legacy_flat_replay_v1", "tool": MIGRATION_TOOL},
        },
        "schema_version": 2,
        "simulation": {
            "amplification": {
                "current_factor": _positive_float(_required(payload, "current_amp"), "current_amp"),
                "voltage_factor": _positive_float(_required(payload, "voltage_amp"), "voltage_amp"),
            },
            "nonlinearity": family,
            "physical": physical,
            "updater": updater,
        },
    }


def _file_reference(root: Path, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Expected manifest asset to exist: {path}.")
    return {"path": relative_path, "sha256": _sha256(path)}


def migrate_manifest(
    root: Path,
    legacy: dict[str, Any],
    migrated_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if legacy.get("schema_version") == 2:
        return legacy
    jobs = legacy.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Expected legacy manifest jobs to be a list.")
    migrated_jobs: list[dict[str, Any]] = []
    for job in jobs:
        config_path = _required(job, "config")
        config = migrated_configs[config_path]
        base_equilibrium = config["equilibrium"]
        iterations = _positive_int(_required(job, "num_iterations"), "job.num_iterations")
        if base_equilibrium["method"] == "fixed_sweeps":
            equilibrium_override = {"sweeps": iterations}
        else:
            equilibrium_override = {"max_sweeps": iterations}
        assets: dict[str, Any] = {
            "weights": _file_reference(root, _required(job, "weights")),
        }
        reference = job.get("reference_npz")
        if reference is not None:
            assets["reference_states"] = _file_reference(root, reference)
        provenance: dict[str, Any] = {"source_run": _required(job, "source")}
        if job.get("variant") is not None:
            provenance["variant"] = job["variant"]
        migrated_jobs.append(
            {
                "assets": assets,
                "base_config": {
                    "path": config_path,
                    "sha256": _payload_sha256(config),
                },
                "group": _required(job, "group"),
                "job_id": _required(job, "job_id"),
                "overrides": {"equilibrium": equilibrium_override},
                "provenance": provenance,
            }
        )
    if len({item["job_id"] for item in migrated_jobs}) != len(migrated_jobs):
        raise ValueError("Expected manifest job_id values to be unique.")
    return {
        "comparison": COMPARISON_POLICY,
        "created_by": MIGRATION_TOOL,
        "demo": {
            "batch_size": 256,
            "job_id": "timing/double_diode_exponential/hidden_1/hidden_64",
        },
        "description": legacy.get(
            "description",
            "Standalone numerical replay catalogue for the nonlinear DRN paper.",
        ),
        "execution_ref": _file_reference(root, "configs/execution/reference_cpu.json"),
        "jobs": migrated_jobs,
        "kind": "replay_manifest",
        "name": legacy.get("name", "nonlinear-drn-paper"),
        "notes": legacy.get("notes", []),
        "schema_version": 2,
    }


def _refresh_manifest_references(root: Path, manifest: dict[str, Any]) -> None:
    manifest.setdefault(
        "execution_ref",
        {"path": "configs/execution/reference_cpu.json", "sha256": "0" * 64},
    )
    manifest["execution_ref"]["sha256"] = _sha256(root / manifest["execution_ref"]["path"])
    for job in manifest["jobs"]:
        for reference in [job["base_config"], *job["assets"].values()]:
            reference["sha256"] = _sha256(root / reference["path"])


def _upgrade_v2_configs(root: Path, manifest: dict[str, Any]) -> None:
    """Materialize policy fields added while finalizing the public v2 contract."""

    config_paths = sorted(
        {job["base_config"]["path"] for job in manifest["jobs"]}
    )
    for relative in config_paths:
        path = root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2 or payload.get("kind") != "replay":
            raise ValueError(f"Expected a replay-v2 config at {relative}.")
        model = payload["model"]
        existing = model.get("weight_initialization")
        if existing not in {None, "kaiming_uniform"}:
            raise ValueError(
                "Cannot materialize model.weight_initialization over an existing "
                f"value in {relative}: {existing!r}."
            )
        model["weight_initialization"] = "kaiming_uniform"
        model["input_encoding"] = "signed_pair"
        model["topology"] = "dense"
        model["amplification_learning"] = {
            "input_gain": False,
            "voltage_factor": False,
            "current_factor": False,
        }
        bias = model["bias"]
        model["bias"] = {
            "bounds": {"maximum": None, "minimum": 0.0},
            "enabled": True,
            "initialization": {"method": "constant", "value": 0.0},
            "interaction": bias["interaction"],
            "scale_mode": bias["scale_mode"],
        }
        defaults = payload["provenance"]["materialized_defaults"]
        defaults.append("model.weight_initialization")
        defaults.extend(
            [
                "model.bias.bounds",
                "model.bias.enabled",
                "model.bias.initialization",
                "model.input_encoding",
                "model.topology",
                "model.amplification_learning",
            ]
        )
        payload["data"]["loader"]["drop_last"] = False
        payload["data"]["split"]["rounding"] = "floor"
        payload["equilibrium"]["initial_state"] = "zeros"
        if payload["simulation"]["nonlinearity"] == "experimental":
            payload["simulation"]["updater"]["nonconvergence_policy"] = "accept_last"
            defaults.append("simulation.updater.nonconvergence_policy")
        polish = payload["simulation"]["updater"].get("polish")
        if (
            payload["simulation"]["nonlinearity"] == "double_diode_exponential"
            and isinstance(polish, dict)
            and polish["max_steps"] == 8
        ):
            polish["max_steps"] = 7
            defaults.append("simulation.updater.polish.max_steps_semantics")
        defaults.extend(
            [
                "data.loader.drop_last",
                "data.split.rounding",
                "equilibrium.initial_state",
            ]
        )
        payload["provenance"]["materialized_defaults"] = sorted(set(defaults))
        path.write_bytes(_canonical_bytes(payload))


def migrate(
    root: Path,
    *,
    check: bool,
    refresh_references: bool = False,
    upgrade_v2: bool = False,
) -> tuple[int, int]:
    manifest_path = root / MANIFEST_RELATIVE_PATH
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if legacy_manifest.get("schema_version") == 2:
        legacy_manifest["comparison"] = COMPARISON_POLICY
        if upgrade_v2:
            _upgrade_v2_configs(root, legacy_manifest)
        if refresh_references or upgrade_v2:
            _refresh_manifest_references(root, legacy_manifest)
            manifest_path.write_bytes(_canonical_bytes(legacy_manifest))
        elif not check:
            raise ValueError(
                "Replay catalogue is already schema version 2; use --check or --refresh-references."
            )
        execution_reference = legacy_manifest.get("execution_ref")
        if execution_reference is None:
            raise ValueError("Expected v2 replay manifest to declare execution_ref.")
        references = [execution_reference]
        references.extend(
            reference
            for job in legacy_manifest["jobs"]
            for reference in [job["base_config"], *job["assets"].values()]
        )
        for reference in references:
            actual = _sha256(root / reference["path"])
            if actual != reference["sha256"]:
                raise ValueError(
                    f"SHA256 mismatch for {reference['path']}: expected {reference['sha256']}, actual {actual}."
                )
        return len({job["base_config"]["path"] for job in legacy_manifest["jobs"]}), len(legacy_manifest["jobs"])

    config_paths = sorted({_required(job, "config") for job in legacy_manifest["jobs"]})
    migrated_configs = {
        relative: migrate_config(
            root,
            relative,
            json.loads((root / relative).read_text(encoding="utf-8")),
        )
        for relative in config_paths
    }
    migrated_manifest = migrate_manifest(root, legacy_manifest, migrated_configs)
    if check:
        raise ValueError("Replay catalogue still uses the legacy schema; run this script without --check.")
    for relative, payload in migrated_configs.items():
        (root / relative).write_bytes(_canonical_bytes(payload))
    manifest_path.write_bytes(_canonical_bytes(migrated_manifest))
    return len(migrated_configs), len(migrated_manifest["jobs"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Repository root.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate an already-migrated catalogue and all per-reference SHA256 values.",
    )
    parser.add_argument(
        "--refresh-references",
        action="store_true",
        help="Refresh per-config and per-asset SHA256 values after an intentional file update.",
    )
    parser.add_argument(
        "--upgrade-v2",
        action="store_true",
        help="Materialize fields added while finalizing the public v2 contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_actions = sum((args.check, args.refresh_references, args.upgrade_v2))
    if selected_actions > 1:
        raise ValueError(
            "Choose exactly one of --check, --refresh-references, or --upgrade-v2."
        )
    configs, jobs = migrate(
        args.root.resolve(),
        check=args.check,
        refresh_references=args.refresh_references,
        upgrade_v2=args.upgrade_v2,
    )
    action = (
        "verified"
        if args.check
        else "upgraded"
        if args.upgrade_v2
        else "refreshed"
        if args.refresh_references
        else "migrated"
    )
    print(f"{action} {configs} replay configs and {jobs} manifest jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
