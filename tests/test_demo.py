from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from repro.cli import main
from repro.digits_validate import DemoResult, run_demo, run_validation
from repro.manifest import PackManifest, ReproJob
from repro.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_demo_evaluates_bundled_checkpoint_without_output_writes() -> None:
    manifest = PackManifest.load(ROOT)
    job = manifest.demo_job()
    output_dir = job.output_dir(ROOT)
    before = set(output_dir.rglob("*")) if output_dir.exists() else None
    result = run_demo(ROOT, manifest)
    after = set(output_dir.rglob("*")) if output_dir.exists() else None
    assert after == before
    assert result.job_id == manifest.demo["job_id"]
    assert result.architecture == (128, 64, 20)
    assert result.non_linearity == "double_diode_exponential"
    assert result.num_iterations == 128
    assert result.batch_size == manifest.demo["batch_size"] == 256
    assert result.execution_profile == "reference_cpu"
    assert result.device == "cpu"
    assert (result.correct, result.total, result.accuracy) == (342, 360, 0.95)
    assert result.receipt["resolved_config_sha256"]
    assert result.receipt["split_fingerprint"]


def test_demo_selection_has_one_manifest_authority() -> None:
    manifest = PackManifest.load(ROOT)
    broken = PackManifest(
        jobs=manifest.jobs,
        demo={"job_id": "missing", "batch_size": 256},
        comparison=manifest.comparison,
        execution_ref=manifest.execution_ref,
        checksums=manifest.checksums,
        raw=manifest.raw,
    )
    with pytest.raises(LookupError, match="Bundled demo job 'missing'"):
        run_demo(ROOT, broken)


def test_demo_cli_prints_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = DemoResult(
        job_id="demo/job",
        config="config.json",
        weights="weights.pt",
        architecture=(128, 64, 20),
        non_linearity="double_diode_exponential",
        num_iterations=128,
        batch_size=256,
        execution_profile="reference_cpu",
        device="cpu",
        correct=342,
        total=360,
        accuracy=0.95,
        receipt={"resolved_config_sha256": "a" * 64},
    )
    monkeypatch.setattr("repro.digits_validate.run_demo", lambda *args, **kwargs: expected)
    monkeypatch.setattr(sys, "argv", ["scripts/reproduce.py", "demo"])
    assert main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "demo_job: demo/job",
        "config: config.json",
        "weights: weights.pt",
        "architecture: 128 -> 64 -> 20",
        "nonlinearity: double_diode_exponential",
        "iterations: 128",
        "inference_batch_size: 256",
        "execution_profile: reference_cpu",
        "device: cpu",
        "correct: 342/360",
        "accuracy: 95.00%",
    ]


def test_validation_writes_hashed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = PackManifest.load(ROOT)
    job = manifest.jobs_for_group("error_vs_iter")[0]
    monkeypatch.setattr(ReproJob, "output_dir", lambda self, root: tmp_path / "run")

    result = run_validation(ROOT, manifest, job)
    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))

    assert metadata["validation_states_sha256"] == sha256_file(result.states_npz)
    assert metadata["validation_inputs_sha256"] == sha256_file(
        result.run_dir / "validation_inputs.npz"
    )
    assert result.receipt_json.is_file()
