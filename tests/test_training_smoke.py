from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from repro.train import run_training


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_name",
    [
        "digits_single_shockley.json",
        "digits_double_shockley.json",
        "digits_pwl.json",
    ],
)
def test_one_batch_training_is_finite(tmp_path: Path, config_name: str) -> None:
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
