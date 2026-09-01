from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repro import build_training_config, run_drn, write_training_config
from repro.strict_config import (
    ConfigurationOverrideError,
    file_sha256,
    validate_document,
)
from repro.train import load_training_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "train" / "digits_double_shockley.json"


def test_builder_only_resolves_explicit_source_and_hashed_profiles() -> None:
    config = build_training_config(SOURCE, repo_root=ROOT)
    assert config["schema_version"] == 2
    assert "simulation_ref" not in config
    assert "execution_ref" not in config
    assert config["simulation"]["nonlinearity"] == "double_diode_exponential"
    assert config["execution"]["name"] == "reference_cpu"
    assert [item["owner"] for item in config["provenance"]["config_sources"]] == [
        "training_source",
        "execution",
        "simulation",
    ]
    validate_document(config, "training-v2.schema.json", repo_root=ROOT)


def test_json_pointer_overrides_are_recorded_and_revalidated() -> None:
    config = build_training_config(
        SOURCE,
        repo_root=ROOT,
        overrides=(
            "/training/epochs=1",
            "/equilibrium/sweeps=2",
        ),
    )
    assert config["training"]["epochs"] == 1
    assert config["equilibrium"]["sweeps"] == 2
    assert config["provenance"]["generation_overrides"] == [
        {"pointer": "/training/epochs", "value": 1},
        {"pointer": "/equilibrium/sweeps", "value": 2},
    ]


@pytest.mark.parametrize(
    "override",
    [
        "/training/epoch=1",
        '/training/epochs="1"',
        '/simulation/nonlinearity="double"',
        "/provenance/config_sources=[]",
        "training/epochs=1",
        "/training/epochs=NaN",
    ],
)
def test_invalid_or_ambiguous_overrides_fail_closed(override: str) -> None:
    with pytest.raises((ValueError, ConfigurationOverrideError)):
        build_training_config(SOURCE, repo_root=ROOT, overrides=[override])


def test_expanded_snapshot_reloads_without_source_resolution(tmp_path: Path) -> None:
    snapshot = write_training_config(
        SOURCE,
        tmp_path / "resolved.json",
        repo_root=ROOT,
    )
    typed, document = load_training_config(snapshot, repo_root=ROOT)
    assert typed.non_linearity == "double_diode_exponential"
    assert typed.execution["name"] == "reference_cpu"
    assert document == json.loads(snapshot.read_text())
    sources = document["provenance"]["config_sources"]
    assert [item["owner"] for item in sources] == [
        "training_source",
        "execution",
        "simulation",
    ]
    assert sources[0] == {
        "owner": "training_source",
        "path": "configs/train/digits_double_shockley.json",
        "sha256": file_sha256(SOURCE),
    }


def test_removed_compact_runner_policy_is_not_exposed() -> None:
    import repro
    import repro.runner as runner

    assert not hasattr(repro, "DRNRunSpec")
    assert not hasattr(runner, "NONLINEARITY_ALIASES")
    assert not hasattr(runner, "PARAMETER_SETS")


def test_run_drn_uses_a_saved_complete_override_snapshot(tmp_path: Path) -> None:
    result = run_drn(
        SOURCE,
        repo_root=ROOT,
        output_dir=tmp_path / "run",
        overrides=(
            "/training/epochs=1",
            "/equilibrium/sweeps=2",
            "/training/batch_limits/train=1",
            "/training/batch_limits/evaluation=1",
        ),
    )
    assert result.final_checkpoint is not None
    assert result.final_checkpoint.is_file()
    assert result.receipt_path.is_file()
    resolved = json.loads((result.output_dir / "config.resolved.json").read_text())
    assert resolved["training"]["epochs"] == 1
    assert resolved["training"]["batch_limits"] == {
        "train": 1,
        "evaluation": 1,
    }
    assert resolved["provenance"]["generation_overrides"]


def test_train_drn_cli_can_materialize_without_running(tmp_path: Path) -> None:
    destination = tmp_path / "generated.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_drn.py"),
            "--config",
            str(SOURCE),
            "--override",
            "/training/epochs=1",
            "--write-config",
            str(destination),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "resolved_config:" in completed.stdout
    payload = json.loads(destination.read_text())
    assert payload["training"]["epochs"] == 1
    assert payload["simulation"]["updater"]["method"] == "lambert_w_v1"
