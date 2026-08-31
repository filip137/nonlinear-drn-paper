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
        device="cpu",
        output_dir=tmp_path / Path(config_name).stem,
        epochs=1,
        num_iterations=2,
        max_batches=1,
        max_eval_batches=1,
    )
    assert result.final_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
    for values in result.history.values():
        assert len(values) == 1
        assert math.isfinite(values[0])
    metadata = json.loads((result.output_dir / "run_metadata.json").read_text())
    assert metadata["final_checkpoint_sha256"]
    output = capsys.readouterr().out
    assert "train epoch=1/1 batch=1/1" in output
    assert "eval epoch=1/1 batch=1/1" in output
    assert "accuracy=" in output


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
