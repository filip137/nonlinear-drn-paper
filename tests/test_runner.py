from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from repro import DRNRunSpec, build_training_config, run_drn
from repro.runner import main
from repro.train import load_training_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "canonical", "output_width"),
    [
        ("single", "single_diode_exponential", 10),
        ("double", "double_diode_exponential", 20),
        ("pwl", "experimental", 20),
    ],
)
def test_runner_builds_each_paper_nonlinearity(
    name: str,
    canonical: str,
    output_width: int,
) -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="digits",
            hidden_sizes=(64, 32),
            learning_rate=0.01,
            non_linearity=name,
            parameter_set="paper-digits",
            epochs=3,
        ),
        repo_root=ROOT,
    )

    assert config["layer_shapes"] == [[128], [64], [32], [output_width]]
    assert config["non_linearity"] == canonical
    assert config["weight_gains"] == [1.0, 1.0, 1.0]
    assert config["learning_rates"] == [0.01] * 5
    assert config["num_epochs"] == 3
    assert config["num_iterations"] == 8
    assert config["adaptive_equilibrium"] is False
    assert config["runner"]["hidden_sizes_source"] == "user"
    assert config["runner"]["num_iterations_source"] == "digits-depth-default"
    assert config["quadratic_diode_param"]
    assert config["exponential_diode_param"]
    assert config["hard_sigmoid_param"]
    schema = json.loads(
        (ROOT / "configs" / "train" / "schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(config, schema)


@pytest.mark.parametrize(
    ("name", "expected_hidden_width", "output_width"),
    [
        ("single", 64, 10),
        ("double", 32, 20),
        ("pwl", 32, 20),
    ],
)
def test_runner_inherits_one_hidden_layer_digits_anchors(
    name: str,
    expected_hidden_width: int,
    output_width: int,
) -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="digits",
            non_linearity=name,
            parameter_set="paper-digits",
        ),
        repo_root=ROOT,
    )

    assert config["layer_shapes"] == [[128], [expected_hidden_width], [output_width]]
    assert config["num_iterations"] == 4
    assert config["runner"]["hidden_sizes"] == [expected_hidden_width]
    assert config["runner"]["hidden_sizes_source"] == "parameter-source"
    assert config["runner"]["num_iterations_source"] == "digits-depth-default"


def test_runner_builds_variable_mnist_architecture_and_layerwise_rates() -> None:
    rates = (0.15, 0.1, 0.05, 0.02, 0.01)
    config = build_training_config(
        DRNRunSpec(
            dataset="mnist",
            hidden_sizes=(100, 50),
            learning_rate=rates,
            non_linearity="double-shockley",
            parameter_set="paper-mnist-xs",
            batch_size=8,
            num_iterations=6,
            mnist_train_samples=128,
            mnist_test_samples=64,
        ),
        repo_root=ROOT,
    )

    assert config["dataset"]["name"] == "mnist"
    assert config["dataset"]["train_samples"] == 128
    assert config["dataset"]["test_samples"] == 64
    assert config["layer_shapes"] == [[2, 28, 28], [100], [50], [20]]
    assert config["learning_rates"] == list(rates)
    assert config["batch_size"] == 8
    assert config["num_iterations"] == 6
    assert config["runner"]["num_iterations_source"] == "user"
    assert config["adaptive_equilibrium"] is False
    assert config["scheduler_gamma"] == 0.99
    assert config["dataset"]["normalize_standard_deviation"] == 0.3081
    assert config["dataset"]["normalize_scale"] == 0.3


def test_runner_keeps_mnist_source_iterations_when_omitted() -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="mnist",
            non_linearity="double",
            parameter_set="paper-mnist-xs",
        ),
        repo_root=ROOT,
    )

    assert config["layer_shapes"] == [[2, 28, 28], [100], [20]]
    assert config["num_iterations"] == 4
    assert config["runner"]["hidden_sizes_source"] == "parameter-source"
    assert config["runner"]["num_iterations_source"] == "parameter-source"


def test_runner_preserves_explicit_digits_iterations_and_positional_api() -> None:
    config = build_training_config(
        DRNRunSpec(
            "digits",
            (48,),
            "double",
            num_iterations=7,
            parameter_set="paper-digits",
        ),
        repo_root=ROOT,
    )

    assert config["layer_shapes"] == [[128], [48], [20]]
    assert config["num_iterations"] == 7
    assert config["runner"]["hidden_sizes_source"] == "user"
    assert config["runner"]["num_iterations_source"] == "user"


@pytest.mark.parametrize(
    ("dataset", "parameter_set", "hidden_sizes", "expected_rates", "expected_input_gain"),
    [
        ("digits", "paper-digits", (64, 32), [0.01] * 5, 10.0),
        ("mnist", "paper-mnist-xs", (100, 50), [0.15, 0.08, 0.08, 0.05, 0.05], 50.0),
    ],
)
def test_runner_expands_known_working_defaults_when_resizing(
    dataset: str,
    parameter_set: str,
    hidden_sizes: tuple[int, ...],
    expected_rates: list[float],
    expected_input_gain: float,
) -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset=dataset,
            hidden_sizes=hidden_sizes,
            non_linearity="double",
            parameter_set=parameter_set,
        ),
        repo_root=ROOT,
    )

    assert config["learning_rates"] == expected_rates
    assert config["input_gain"] == expected_input_gain
    assert config["runner"]["learning_rate_source"] == "parameter-source"
    assert config["runner"]["input_gain_source"] == "parameter-source"


def test_runner_expands_single_shockley_weight_only_default_when_resizing() -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="digits",
            hidden_sizes=(128, 64),
            non_linearity="single",
            parameter_set="paper-digits",
        ),
        repo_root=ROOT,
    )

    assert config["learning_rates"] == [0.005, 0.005, 0.005, 0.0, 0.0]
    assert config["input_gain"] == 10.0
    assert config["adaptive_equilibrium"] is False
    assert config["runner"]["learning_rate_source"] == "parameter-source"


def test_runner_rejects_odd_single_shockley_hidden_width() -> None:
    with pytest.raises(ValueError, match="Expected every single-Shockley hidden size to be even"):
        build_training_config(
            DRNRunSpec(
                dataset="digits",
                hidden_sizes=(63,),
                learning_rate=0.01,
                non_linearity="single",
                parameter_set="paper-digits",
            ),
            repo_root=ROOT,
        )


def test_runner_requires_explicit_physical_parameter_source() -> None:
    with pytest.raises(ValueError, match="Expected exactly one of parameter_set or parameter_config"):
        build_training_config(
            DRNRunSpec(
                dataset="digits",
                hidden_sizes=(64,),
                learning_rate=0.01,
                non_linearity="double",
            ),
            repo_root=ROOT,
        )


def test_runner_dry_run_prints_resolved_config_without_training(capsys: pytest.CaptureFixture[str]) -> None:
    status = main(
        [
            "--dataset",
            "digits",
            "--learning-rate",
            "0.01",
            "--non-linearity",
            "double",
            "--parameter-set",
            "paper-digits",
            "--dry-run",
        ],
        repo_root=ROOT,
    )
    printed = json.loads(capsys.readouterr().out)

    assert status == 0
    assert printed["layer_shapes"] == [[128], [32], [20]]
    assert printed["num_iterations"] == 4
    assert printed["runner"]["parameter_source"].startswith("parameter-set:paper-digits")
    assert printed["runner"]["hidden_sizes_source"] == "parameter-source"
    assert printed["runner"]["num_iterations_source"] == "digits-depth-default"


def test_runner_executes_variable_depth_training(tmp_path: Path) -> None:
    result = run_drn(
        DRNRunSpec(
            dataset="digits",
            hidden_sizes=(16, 8),
            non_linearity="double",
            parameter_set="paper-digits",
            epochs=1,
            batch_size=8,
            num_iterations=2,
            digits_num_samples=40,
            seed=7,
        ),
        repo_root=ROOT,
        device="cpu",
        output_dir=tmp_path / "runner-training",
        max_batches=1,
        max_eval_batches=1,
    )

    generated = result.output_dir / "config.generated.json"
    loaded, _ = load_training_config(generated, repo_root=ROOT)
    metadata = json.loads((result.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert loaded.layer_shapes == [(128,), (16,), (8,), (20,)]
    assert loaded.learning_rates == [0.01] * 5
    assert metadata["adaptive_equilibrium"] is False
    assert result.final_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
