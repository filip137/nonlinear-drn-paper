from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path

import jsonschema
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
REPLAY_SCHEMA_PATH = ROOT / "configs" / "schema" / "replay-v2.schema.json"
MANIFEST_SCHEMA_PATH = ROOT / "configs" / "schema" / "manifest-v2.schema.json"
COMMON_SCHEMA_PATH = ROOT / "configs" / "schema" / "common-v2.schema.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_paths(manifest: dict) -> list[str]:
    return sorted({job["base_config"]["path"] for job in manifest["jobs"]})


def _schema_registry() -> Registry:
    common_schema = _load(COMMON_SCHEMA_PATH)
    return Registry().with_resource(
        common_schema["$id"],
        Resource.from_contents(common_schema),
    )


def test_replay_v2_catalogue_has_exact_declared_coverage() -> None:
    manifest = _load(MANIFEST_PATH)

    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "replay_manifest"
    assert len(manifest["jobs"]) == 195
    assert len(_config_paths(manifest)) == 96
    assert Counter(job["group"] for job in manifest["jobs"]) == {
        "error_vs_iter": 108,
        "timing": 42,
        "vol_tol": 45,
    }
    assert len({job["job_id"] for job in manifest["jobs"]}) == 195

    job_ids = "\n".join(sorted(job["job_id"] for job in manifest["jobs"])) + "\n"
    assert hashlib.sha256(job_ids.encode()).hexdigest() == (
        "bf9420da87e7a2ddf6fb2972767d94127ced0df8ef0e0e8b2dcab41d0a732b6f"
    )

    rows = []
    for job in manifest["jobs"]:
        assets = job["assets"]
        override = job["overrides"]["equilibrium"]
        sweeps = override.get("sweeps", override.get("max_sweeps"))
        rows.append(
            "\t".join(
                (
                    job["job_id"],
                    job["group"],
                    job["base_config"]["path"],
                    assets["weights"]["path"],
                    assets.get("reference_states", {}).get("path", ""),
                    str(sweeps),
                )
            )
        )
    catalogue = "\n".join(sorted(rows)) + "\n"
    assert hashlib.sha256(catalogue.encode()).hexdigest() == (
        "1970c722ce84ede23c511bd96c97c4ba23e0a21e3da0d7518de0abf6809344c6"
    )


def test_replay_v2_documents_validate_against_strict_schemas() -> None:
    manifest = _load(MANIFEST_PATH)
    manifest_schema = _load(MANIFEST_SCHEMA_PATH)
    replay_schema = _load(REPLAY_SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    jsonschema.Draft202012Validator.check_schema(replay_schema)
    jsonschema.validate(manifest, manifest_schema)

    validator = jsonschema.Draft202012Validator(replay_schema, registry=_schema_registry())
    for relative in _config_paths(manifest):
        errors = sorted(validator.iter_errors(_load(ROOT / relative)), key=lambda error: error.json_path)
        assert not errors, (relative, [(error.json_path, error.message) for error in errors])


def test_replay_schema_rejects_ambiguity_and_accepts_expanded_snapshot() -> None:
    manifest = _load(MANIFEST_PATH)
    schema = _load(REPLAY_SCHEMA_PATH)
    base = _load(ROOT / manifest["jobs"][0]["base_config"]["path"])
    validator = jsonschema.Draft202012Validator(schema, registry=_schema_registry())

    unknown = deepcopy(base)
    unknown["num_iterations"] = 32
    assert list(validator.iter_errors(unknown))

    numeric_string = deepcopy(base)
    numeric_string["data"]["loader"]["batch_size"] = "256"
    assert list(validator.iter_errors(numeric_string))

    execution_profile = _load(ROOT / manifest["execution_ref"]["path"])
    expanded = deepcopy(base)
    expanded["execution"] = execution_profile["execution"]
    expanded_validator = jsonschema.Draft202012Validator(schema, registry=_schema_registry())
    assert not list(expanded_validator.iter_errors(expanded))


def test_replay_configs_materialize_every_effective_choice() -> None:
    manifest = _load(MANIFEST_PATH)
    legacy_top_level_fields = {
        "adaptive_equilibrium",
        "dims",
        "double_diode_updater",
        "exp_clip",
        "experimental_newton_max_steps",
        "experimental_newton_tol",
        "minimizer_impl",
        "non_linearity",
        "num_iterations",
        "overrelaxation_factor",
        "rel_tol",
        "single_diode_updater",
        "vn_tol",
    }
    for relative in _config_paths(manifest):
        config = _load(ROOT / relative)
        assert config["schema_version"] == 2
        assert config["kind"] == "replay"
        assert not legacy_top_level_fields.intersection(config)

        data = config["data"]
        assert data["seed"] == 0
        assert data["loader"] == {
            "batch_size": data["loader"]["batch_size"],
            "drop_last": False,
            "shuffle": False,
        }
        assert data["preprocessing"] == {
            "input_dtype": "float32",
            "input_offset": -1.0,
            "input_scale": 0.125,
            "target_dtype": "int64",
        }
        assert data["split"] == {
            "method": "random_fraction",
            "rounding": "floor",
            "train_fraction": 0.8,
        }
        assert data["subset"] == {"max_points": 2000, "method": "random_max_points"}

        model = config["model"]
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
        assert len(model["weight_gains"]) == len(model["layer_widths"]) - 1

        equilibrium = config["equilibrium"]
        assert equilibrium["initial_state"] == "zeros"
        assert equilibrium["update_order"] == "asynchronous"
        if equilibrium["method"] == "fixed_sweeps":
            assert set(equilibrium) == {
                "initial_state",
                "method",
                "sweeps",
                "update_order",
            }
        else:
            assert set(equilibrium) == {
                "absolute_tolerance",
                "initial_state",
                "max_sweeps",
                "method",
                "relative_tolerance",
                "update_order",
            }

        simulation = config["simulation"]
        updater = simulation["updater"]
        if simulation["nonlinearity"] == "experimental":
            assert set(updater) == {
                "curve",
                "damping",
                "extrapolation",
                "nonconvergence_policy",
                "max_steps",
                "method",
                "relaxation",
                "voltage_tolerance",
            }
            assert updater["method"] == "piecewise_linear_newton_v1"
        else:
            assert updater["method"] == "lambert_w_v1"
            assert updater["backend"] == "torchlambertw"
            assert updater["linear_coefficient_clamp"] == 1e6
            assert updater["asymptotic_terms"] == 4
            if updater["polish"] is False:
                assert "max_steps" not in updater


def test_hidden3_pwl_fallbacks_are_materialized_exactly() -> None:
    for exponent in range(3, 8):
        path = ROOT / (
            "data/vol_tol/configs/"
            f"experimental__hidden_3__rel_tol_1e-{exponent}.json"
        )
        config = _load(path)
        updater = config["simulation"]["updater"]
        assert updater["damping"] == 0.5
        assert updater["max_steps"] == 100
        assert updater["voltage_tolerance"] == 1e-5
        assert {
            "simulation.updater.max_steps",
            "simulation.updater.voltage_tolerance",
        }.issubset(config["provenance"]["materialized_defaults"])


def test_manifest_has_one_checksum_index_and_no_derived_job_duplicates() -> None:
    manifest = _load(MANIFEST_PATH)
    assert "checksums" not in manifest
    assert "groups" not in manifest
    assert (ROOT / "data" / "checksums.sha256").is_file()
    assert manifest["execution_ref"]["path"] == "configs/execution/reference_cpu.json"

    duplicated_fields = {
        "family",
        "hidden_layers",
        "hidden_size",
        "num_iterations",
        "overrelaxation_factor",
        "reference_npz",
        "rel_tol",
        "source",
        "variant",
        "weights",
    }
    for job in manifest["jobs"]:
        assert not duplicated_fields.intersection(job)

    demo = manifest["demo"]
    assert demo == {
        "batch_size": 256,
        "job_id": "timing/double_diode_exponential/hidden_1/hidden_64",
    }
    assert sum(job["job_id"] == demo["job_id"] for job in manifest["jobs"]) == 1


def test_manifest_references_are_portable_valid_and_hash_verified() -> None:
    manifest = _load(MANIFEST_PATH)
    root = ROOT.resolve()
    execution_ref = manifest["execution_ref"]
    execution_path = (ROOT / execution_ref["path"]).resolve()
    assert execution_path.is_relative_to(root)
    assert _sha256(execution_path) == execution_ref["sha256"]
    for job in manifest["jobs"]:
        references = [job["base_config"], *job["assets"].values()]
        for reference in references:
            path = (ROOT / reference["path"]).resolve()
            assert path.is_relative_to(root)
            assert path.is_file()
            assert _sha256(path) == reference["sha256"]

        config = _load(ROOT / job["base_config"]["path"])
        override = job["overrides"]["equilibrium"]
        if config["equilibrium"]["method"] == "fixed_sweeps":
            assert set(override) == {"sweeps"}
        else:
            assert set(override) == {"max_sweeps"}
