from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch
from torchvision import transforms

from model.resistive.network import DeepResistiveEnergy
from repro.train import PAPER_NONLINEARITIES, _build_mnist_transform, load_training_config


ROOT = Path(__file__).resolve().parents[1]


def test_training_configs_cover_all_paper_nonlinearities() -> None:
    paths = [
        ROOT / "configs" / "train" / name
        for name in (
            "digits_single_shockley.json",
            "digits_double_shockley.json",
            "digits_pwl.json",
        )
    ]
    loaded = [load_training_config(path, repo_root=ROOT)[0] for path in paths]
    assert {config.non_linearity for config in loaded} == set(PAPER_NONLINEARITIES)
    for config in loaded:
        assert len(config.layer_shapes) == 3
        assert config.num_iterations == 4
        if config.non_linearity != "experimental":
            assert config.exponential_diode_param
        assert config.adaptive_equilibrium is False
        assert config.scheduler_gamma == 1.0


def test_editable_defaults_are_dataset_specific_and_composed() -> None:
    cases = [
        ("default_single_shockley.json", "digits", "single_diode_exponential", 32),
        ("default_double_shockley.json", "digits", "double_diode_exponential", 32),
        ("default_custom_iv.json", "digits", "experimental", 32),
        (
            "default_mnist_single_shockley.json",
            "mnist",
            "single_diode_exponential",
            100,
        ),
        (
            "default_mnist_double_shockley.json",
            "mnist",
            "double_diode_exponential",
            100,
        ),
        ("default_mnist_custom_iv.json", "mnist", "experimental", 100),
    ]
    seen: set[tuple[str, str]] = set()

    for name, dataset, non_linearity, hidden_width in cases:
        path = ROOT / "configs" / "train" / name
        config, expanded = load_training_config(path, repo_root=ROOT)
        source = json.loads(path.read_text(encoding="utf-8"))

        seen.add((dataset, non_linearity))
        assert config.dataset["name"] == dataset
        assert config.non_linearity == non_linearity
        assert config.layer_shapes[-2] == (hidden_width,)
        assert config.batch_size == 10
        assert "simulator_profile" in source
        assert "simulator_profile" not in expanded
        assert expanded["simulator_profile_source"] == source["simulator_profile"]
        profile_path = ROOT / source["simulator_profile"]
        assert expanded["simulator_profile_sha256"] == hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()
        assert "quadratic_diode_param" not in expanded
        assert "hard_sigmoid_param" not in expanded
        if non_linearity == "experimental":
            assert config.iv_data_path is not None
            assert Path(config.iv_data_path).is_file()

    assert seen == {
        (dataset, non_linearity)
        for dataset in ("digits", "mnist")
        for non_linearity in PAPER_NONLINEARITIES
    }


def test_training_configs_declare_documented_json_schema() -> None:
    schema_path = ROOT / "configs" / "train" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    for field in (
        "z_thresh",
        "exp_clip",
        "use_polish",
        "max_newton_iters",
        "damping",
        "experimental_newton_max_steps",
        "experimental_newton_tol",
        "adaptive_equilibrium",
        "rel_tol",
        "vn_tol",
    ):
        assert field in schema["properties"]
        assert schema["properties"][field]["description"]

    schema_url = schema["$id"]
    for path in sorted((ROOT / "configs" / "train").glob("*.json")):
        if path == schema_path:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(raw)
        assert raw["$schema"] == schema_url
        assert set(schema["required"]).issubset(raw)
        assert "quadratic_diode_param" not in raw
        assert "hard_sigmoid_param" not in raw


def test_simulator_profiles_declare_documented_json_schema() -> None:
    schema_path = ROOT / "configs" / "simulator" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    schema_url = schema["$id"]

    for path in sorted((ROOT / "configs" / "simulator").glob("*.json")):
        if path == schema_path:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(raw)
        assert raw["$schema"] == schema_url


def test_composed_config_reloads_without_profile_file_lookup(tmp_path: Path) -> None:
    source = ROOT / "configs" / "train" / "default_single_shockley.json"
    _, expanded = load_training_config(source, repo_root=ROOT)
    generated = tmp_path / "config.generated.json"
    generated.write_text(json.dumps(expanded), encoding="utf-8")

    reloaded, reloaded_raw = load_training_config(generated, repo_root=ROOT)

    assert reloaded.non_linearity == "single_diode_exponential"
    assert reloaded_raw == expanded


def test_simulator_profile_rejects_inline_setting_collisions(tmp_path: Path) -> None:
    source = ROOT / "configs" / "train" / "default_single_shockley.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["mode"] = "asynchronous"
    candidate = tmp_path / "collision.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate fields.*mode"):
        load_training_config(candidate, repo_root=ROOT)


def test_simulator_profile_requires_existing_repo_file(tmp_path: Path) -> None:
    source = ROOT / "configs" / "train" / "default_single_shockley.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["simulator_profile"] = "configs/simulator/does-not-exist.json"
    candidate = tmp_path / "missing-profile.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="existing JSON file"):
        load_training_config(candidate, repo_root=ROOT)


def test_legacy_unused_training_parameters_are_dropped(tmp_path: Path) -> None:
    source = ROOT / "configs" / "train" / "digits_double_shockley.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["quadratic_diode_param"] = {
        "diode_conductance": 1.0,
        "v_min": -0.5,
        "v_max": 0.5,
    }
    raw["hard_sigmoid_param"] = {
        "g_on": 10.0,
        "g_off": 0.1,
        "v_min": -1.0,
        "v_max": 1.0,
    }
    candidate = tmp_path / "legacy.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")

    _, expanded = load_training_config(candidate, repo_root=ROOT)

    assert "quadratic_diode_param" not in expanded
    assert "hard_sigmoid_param" not in expanded


def test_paper_mnist_config_is_the_reported_drn_xs_shape() -> None:
    config, _ = load_training_config(
        ROOT / "configs" / "train" / "mnist_paper_double_shockley.json",
        repo_root=ROOT,
    )
    assert config.layer_shapes == [(2, 28, 28), (100,), (20,)]
    assert config.non_linearity == "double_diode_exponential"
    assert config.num_iterations == 4
    assert config.num_epochs == 100
    assert config.adaptive_equilibrium is False
    assert config.input_gain == 50.0
    assert config.learning_rates == [0.15, 0.08, 0.05]
    assert config.scheduler_gamma == 0.99
    assert config.use_polish is False
    assert config.exp_clip == 165.0

    raw = json.loads(
        (ROOT / "configs" / "train" / "mnist_paper_double_shockley.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw["dataset"]["normalize_mean"] == 0.1307
    assert raw["dataset"]["normalize_standard_deviation"] == 0.3081
    assert raw["dataset"]["normalize_scale"] == 0.3


def test_paper_mnist_rates_align_with_optimizer_parameter_order() -> None:
    config, _ = load_training_config(
        ROOT / "configs" / "train" / "mnist_paper_double_shockley.json",
        repo_root=ROOT,
    )
    energy = DeepResistiveEnergy(
        layer_shapes=config.layer_shapes,
        weight_gains=config.weight_gains,
        input_gain=config.input_gain,
        non_linearity=config.non_linearity,
        exponential_diode_param=config.exponential_diode_param,
        quadratic_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=config.voltage_amp,
        current_amp=config.current_amp,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
        weight_init_mode=config.weight_init_mode,
    )
    named_rates = [
        (param.name.rsplit("_", 1)[0], rate)
        for param, rate in zip(energy.params(), config.learning_rates)
    ]
    assert named_rates == [
        ("DenseWeight", 0.15),
        ("DenseWeight", 0.08),
        ("Bias", 0.05),
    ]


def test_paper_mnist_normalization_matches_historical_post_scale() -> None:
    _, raw = load_training_config(
        ROOT / "configs" / "train" / "mnist_paper_double_shockley.json",
        repo_root=ROOT,
    )
    transform = _build_mnist_transform(raw["dataset"], transforms)
    white_pixel = np.full((1, 1), 255, dtype=np.uint8)
    actual = transform(white_pixel)
    expected = 0.3 * (1.0 - 0.1307) / 0.3081
    assert torch.allclose(actual, torch.tensor([[[expected]]]), atol=1e-6)


def test_single_shockley_digits_config_has_validated_defaults() -> None:
    config, _ = load_training_config(
        ROOT / "configs" / "train" / "digits_single_shockley.json",
        repo_root=ROOT,
    )
    assert config.layer_shapes == [(128,), (64,), (10,)]
    assert config.learning_rates == [0.005, 0.005, 0.0]
    assert config.input_gain == 10.0
    assert config.batch_size == 10
    assert config.num_iterations == 4
    assert config.adaptive_equilibrium is False


def test_pwl_curve_is_portable_and_present() -> None:
    config, _ = load_training_config(
        ROOT / "configs" / "train" / "digits_pwl.json",
        repo_root=ROOT,
    )
    assert config.iv_data_path is not None
    assert Path(config.iv_data_path).is_file()


@pytest.mark.parametrize("adaptive_value", [True, "false", 0])
def test_training_configs_reject_adaptive_equilibrium(
    tmp_path: Path,
    adaptive_value: object,
) -> None:
    source = ROOT / "configs" / "train" / "digits_double_shockley.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["adaptive_equilibrium"] = adaptive_value
    candidate = tmp_path / "adaptive-training.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Expected config 'adaptive_equilibrium'",
    ):
        load_training_config(candidate, repo_root=ROOT)


@pytest.mark.parametrize("gamma", [0.0, -0.1])
def test_training_configs_reject_nonpositive_scheduler_gamma(
    tmp_path: Path,
    gamma: float,
) -> None:
    source = ROOT / "configs" / "train" / "digits_double_shockley.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scheduler_gamma"] = gamma
    candidate = tmp_path / "scheduler-training.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected config 'scheduler_gamma'"):
        load_training_config(candidate, repo_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("z_thresh", 1.0, "Expected config 'z_thresh'"),
        ("z_thresh", float("nan"), "Expected config 'z_thresh'"),
        ("exp_clip", 0.0, "Expected config 'exp_clip'"),
        ("max_newton_iters", -1, "Expected config 'max_newton_iters'"),
        ("damping", 0.0, "Expected config 'damping'"),
        ("overrelaxation_factor", 0.0, "Expected config 'overrelaxation_factor'"),
        ("experimental_newton_max_steps", 0, "Expected config 'experimental_newton_max_steps'"),
        ("experimental_newton_tol", 0.0, "Expected config 'experimental_newton_tol'"),
    ],
)
def test_training_configs_reject_invalid_solver_settings(
    tmp_path: Path,
    field: str,
    value: object,
    expected_message: str,
) -> None:
    source = ROOT / "configs" / "train" / "digits_double_shockley.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config[field] = value
    candidate = tmp_path / "invalid-solver-setting.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_training_config(candidate, repo_root=ROOT)


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_training_configs_require_boolean_use_polish(
    tmp_path: Path,
    value: object,
) -> None:
    source = ROOT / "configs" / "train" / "digits_double_shockley.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["use_polish"] = value
    candidate = tmp_path / "invalid-polish-setting.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected config 'use_polish' to be a boolean"):
        load_training_config(candidate, repo_root=ROOT)
