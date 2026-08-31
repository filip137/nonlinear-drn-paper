from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema
import numpy as np
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


@pytest.mark.parametrize(
    ("name", "expected_hidden_width", "output_width"),
    [
        ("single", 100, 10),
        ("double", 32, 20),
        ("pwl", 100, 20),
    ],
)
def test_default_parameter_set_supports_every_nonlinearity_on_mnist(
    name: str,
    expected_hidden_width: int,
    output_width: int,
) -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="mnist",
            non_linearity=name,
            parameter_set="default",
        ),
        repo_root=ROOT,
    )

    assert config["dataset"]["name"] == "mnist"
    assert config["layer_shapes"] == [
        [2, 28, 28],
        [expected_hidden_width],
        [output_width],
    ]
    assert config["runner"]["parameter_source"].startswith("parameter-set:default")


def test_incompatible_paper_parameter_set_lists_working_alternatives() -> None:
    with pytest.raises(ValueError) as caught:
        build_training_config(
            DRNRunSpec(
                dataset="mnist",
                non_linearity="single",
                parameter_set="paper-mnist-xs",
            ),
            repo_root=ROOT,
        )

    message = str(caught.value)
    assert "Compatible bundled parameter sets" in message
    assert "default" in message
    assert "paper-digits" in message
    assert "new experiment, not a paper reproduction" in message


def test_parameter_set_accepts_a_configuration_path() -> None:
    source = ROOT / "configs" / "train" / "default_single_shockley.json"
    config = build_training_config(
        DRNRunSpec(
            dataset="mnist",
            non_linearity="single",
            parameter_set=source,
        ),
        repo_root=ROOT,
    )

    assert config["layer_shapes"] == [[2, 28, 28], [100], [10]]
    assert config["runner"]["parameter_source"] == str(source.relative_to(ROOT))


@pytest.mark.parametrize(
    ("hidden_sizes", "expected_iterations", "expected_source"),
    [
        ((64, 32, 16), 8, "digits-depth-default"),
        ((64, 32, 16, 8), 4, "parameter-source"),
    ],
)
def test_runner_resolves_three_and_four_layer_digits_iteration_defaults(
    hidden_sizes: tuple[int, ...],
    expected_iterations: int,
    expected_source: str,
) -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="digits",
            hidden_sizes=hidden_sizes,
            non_linearity="double",
            parameter_set="paper-digits",
        ),
        repo_root=ROOT,
    )

    assert config["num_iterations"] == expected_iterations
    assert config["runner"]["num_iterations_source"] == expected_source


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


def test_runner_custom_iv_curve_overrides_parameter_source_and_resolves_from_root(
    tmp_path: Path,
) -> None:
    curve = tmp_path / "custom-curve.npz"
    np.savez(
        curve,
        i=np.array([-1.0, 0.0, 2.0]),
        v=np.array([-0.5, 0.0, 0.5]),
    )
    relative_curve = Path(os.path.relpath(curve, ROOT))

    config = build_training_config(
        DRNRunSpec(
            dataset="digits",
            non_linearity="pwl",
            parameter_set="paper-digits",
            iv_data_path=relative_curve,
        ),
        repo_root=ROOT,
    )

    assert config["iv_data_path"] == str(curve.resolve())
    assert config["runner"]["iv_data_source"] == "user"
    assert config["runner"]["parameter_source"].startswith(
        "parameter-set:paper-digits"
    )


def test_runner_records_parameter_source_for_bundled_iv_curve() -> None:
    config = build_training_config(
        DRNRunSpec(
            dataset="digits",
            non_linearity="experimental",
            parameter_set="paper-digits",
        ),
        repo_root=ROOT,
    )

    assert config["iv_data_path"] == (
        "data/assets/experimental_curve_voff_0.8_200_points.npz"
    )
    assert config["runner"]["iv_data_source"] == "parameter-source"


def test_runner_rejects_iv_curve_for_analytic_nonlinearity(tmp_path: Path) -> None:
    curve = tmp_path / "curve.npz"
    np.savez(curve, iv=np.array([[0.0, 1.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="only with the measured/PWL nonlinearity"):
        build_training_config(
            DRNRunSpec(
                dataset="digits",
                non_linearity="double",
                parameter_set="paper-digits",
                iv_data_path=curve,
            ),
            repo_root=ROOT,
        )


def test_runner_dry_run_validates_and_prints_custom_iv_curve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    curve = tmp_path / "custom-curve.npz"
    np.savez(curve, iv=np.array([[0.0, 1.0], [0.0, 1.0]]))

    status = main(
        [
            "--dataset",
            "digits",
            "--non-linearity",
            "pwl",
            "--parameter-set",
            "paper-digits",
            "--iv-data-path",
            str(curve),
            "--dry-run",
        ],
        repo_root=ROOT,
    )
    printed = json.loads(capsys.readouterr().out)

    assert status == 0
    assert printed["iv_data_path"] == str(curve.resolve())
    assert printed["runner"]["iv_data_source"] == "user"


def test_runner_dry_run_rejects_malformed_custom_iv_curve(tmp_path: Path) -> None:
    curve = tmp_path / "invalid-curve.npz"
    np.savez(curve, iv=np.array([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError, match="voltages to be strictly increasing"):
        main(
            [
                "--dataset",
                "digits",
                "--non-linearity",
                "pwl",
                "--parameter-set",
                "paper-digits",
                "--iv-data-path",
                str(curve),
                "--dry-run",
            ],
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
            "configs/train/default_double_shockley.json",
            "--dry-run",
        ],
        repo_root=ROOT,
    )
    printed = json.loads(capsys.readouterr().out)

    assert status == 0
    assert printed["layer_shapes"] == [[128], [32], [20]]
    assert printed["num_iterations"] == 4
    assert printed["runner"]["parameter_source"] == (
        "configs/train/default_double_shockley.json"
    )
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
