from __future__ import annotations

import sys
from pathlib import Path

import pytest

from repro.cli import main
from repro.digits_validate import DEMO_BATCH_SIZE, DEMO_JOB_ID, DemoResult, run_demo
from repro.manifest import PackManifest


ROOT = Path(__file__).resolve().parents[1]


def test_demo_evaluates_bundled_checkpoint_without_output_writes() -> None:
    manifest = PackManifest.load(ROOT)
    demo_job = next(job for job in manifest.jobs if job.job_id == DEMO_JOB_ID)
    output_dir = demo_job.output_dir(ROOT)
    before = set(output_dir.rglob("*")) if output_dir.exists() else None

    result = run_demo(ROOT, manifest, device="cpu")

    after = set(output_dir.rglob("*")) if output_dir.exists() else None
    assert after == before
    assert result == DemoResult(
        job_id=DEMO_JOB_ID,
        config="data/timing/configs/double_diode_exponential__hidden_1__hidden_64/config.json",
        weights="data/timing/weights/double_diode_exponential__hidden_1__hidden_64/weights.pt",
        architecture=(128, 64, 20),
        non_linearity="double_diode_exponential",
        num_iterations=128,
        batch_size=DEMO_BATCH_SIZE,
        device="cpu",
        correct=342,
        total=360,
        accuracy=0.95,
    )


def test_demo_reports_missing_manifest_job() -> None:
    manifest = PackManifest(jobs=[], checksums={}, raw={})

    with pytest.raises(LookupError, match=f"Bundled demo job {DEMO_JOB_ID!r} was not found"):
        run_demo(ROOT, manifest)


def test_demo_cli_prints_model_identity_and_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = DemoResult(
        job_id=DEMO_JOB_ID,
        config="config.json",
        weights="weights.pt",
        architecture=(128, 64, 20),
        non_linearity="double_diode_exponential",
        num_iterations=128,
        batch_size=256,
        device="cpu",
        correct=342,
        total=360,
        accuracy=0.95,
    )
    monkeypatch.setattr("repro.digits_validate.run_demo", lambda *args, **kwargs: expected)
    monkeypatch.setattr(sys, "argv", ["scripts/reproduce.py", "demo"])

    assert main() == 0
    assert capsys.readouterr().out.splitlines() == [
        f"demo_job: {DEMO_JOB_ID}",
        "config: config.json",
        "weights: weights.pt",
        "architecture: 128 -> 64 -> 20",
        "nonlinearity: double_diode_exponential",
        "iterations: 128",
        "inference_batch_size: 256",
        "device: cpu",
        "correct: 342/360",
        "accuracy: 95.00%",
    ]
