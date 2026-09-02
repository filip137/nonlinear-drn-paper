from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

from repro.train import _LiveProgress, run_training


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_name",
    [
        "digits_single_shockley.json",
        "digits_double_shockley.json",
        "digits_pwl.json",
        "default_single_shockley.json",
        "default_double_shockley.json",
        "default_custom_iv.json",
    ],
)
def test_one_batch_training_is_finite(
    tmp_path: Path,
    config_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_training(
        ROOT,
        ROOT / "configs" / "train" / config_name,
        output_dir=tmp_path / Path(config_name).stem,
        overrides=(
            "/training/epochs=1",
            "/equilibrium/sweeps=2",
            "/training/batch_limits/train=1",
            "/training/batch_limits/evaluation=1",
        ),
    )
    assert result.final_checkpoint is not None
    assert result.final_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
    for values in result.history.values():
        assert len(values) == 1
        assert math.isfinite(values[0])
    metadata = json.loads((result.output_dir / "run_metadata.json").read_text())
    assert metadata["final_checkpoint_sha256"]
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["resolved_config_sha256"]
    assert receipt["data_fingerprint"]
    assert receipt["split_fingerprint"]
    sources = {item["owner"]: item for item in receipt["source_documents"]}
    assert set(sources) == {"training", "simulation", "execution"}
    assert all(len(item["sha256"]) == 64 for item in sources.values())
    if config_name in {"digits_pwl.json", "default_custom_iv.json"}:
        curve = receipt["assets"]["iv_curve"]
        assert curve["path"] == (
            "data/assets/experimental_curve_voff_0.8_200_points.npz"
        )
        assert len(curve["sha256"]) == 64
    output = capsys.readouterr().out
    assert "train epoch=1/1 batch=1/1" in output
    assert "eval epoch=1/1 batch=1/1" in output
    assert "accuracy=" in output


def test_float64_training_config_controls_inputs_parameters_and_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_training(
        ROOT,
        ROOT / "configs" / "train" / "default_single_shockley.json",
        output_dir=tmp_path / "float64",
        overrides=(
            "/data/preprocessing/dtype=\"float64\"",
            "/model/state_dtype=\"float64\"",
            "/execution/backend/default_dtype=\"float64\"",
            "/training/epochs=1",
            "/equilibrium/sweeps=2",
            "/training/batch_limits/train=1",
            "/training/batch_limits/evaluation=1",
        ),
    )

    assert result.final_checkpoint is not None
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["execution"]["backend"]["default_dtype"] == "float64"
    assert receipt["runtime"]["default_dtype"] == "float64"
    assert all(math.isfinite(values[0]) for values in result.history.values())
    capsys.readouterr()


def test_live_progress_refreshes_every_batch_in_a_terminal() -> None:
    stream = io.StringIO()
    progress = _LiveProgress(
        "train",
        2,
        5,
        2,
        stream=stream,
        interactive=True,
    )

    progress.start()
    progress.update(1, running_loss=0.75, running_accuracy=0.5)
    progress.update(2, running_loss=0.5, running_accuracy=0.75)
    progress.close()

    output = stream.getvalue()
    assert "\rtrain epoch=2/5 batches=2 starting" in output
    assert "\rtrain epoch=2/5 batch=1/2" in output
    assert "\rtrain epoch=2/5 batch=2/2 loss=0.5 accuracy=75.00%" in output
    assert output.endswith("\n")


def test_live_progress_limits_redirected_log_updates() -> None:
    stream = io.StringIO()
    progress = _LiveProgress(
        "eval",
        1,
        1,
        100,
        stream=stream,
        interactive=False,
    )

    progress.start()
    for batch in range(1, 101):
        progress.update(batch, running_loss=1.0, running_accuracy=0.25)
    progress.close()

    output = stream.getvalue()
    assert "eval epoch=1/1 batches=100 starting" in output
    assert "batch=1/100" in output
    assert "batch=10/100" in output
    assert "batch=100/100" in output
    assert "batch=2/100" not in output
    assert "\r" not in output
