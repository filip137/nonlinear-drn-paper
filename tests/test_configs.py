from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.strict_config import (
    ConfigurationReferenceError,
    ConfigurationValidationError,
)
from repro.train import _build_mnist_transform, load_training_config


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "configs" / "train"


def test_every_training_source_is_runtime_validated_and_expands() -> None:
    sources = sorted(path for path in TRAIN_DIR.glob("*.json") if path.name != "schema.json")
    assert len(sources) == 10
    for source in sources:
        typed, expanded = load_training_config(source, repo_root=ROOT)
        assert expanded["schema_version"] == 2
        assert "simulation" in expanded
        assert "execution" in expanded
        assert "simulation_ref" not in expanded
        assert "execution_ref" not in expanded
        assert typed.model["bias"]
        assert isinstance(typed.model["signed_weights"], bool)
        assert typed.training["optimizer"]["momentum"] == 0.0
        assert typed.training["optimizer"]["weight_decay"] == 0.0


def test_runtime_rejects_unknown_typo_and_wrong_scalar_type(tmp_path: Path) -> None:
    raw = json.loads((TRAIN_DIR / "default_double_shockley.json").read_text())
    raw["training"]["epochz"] = raw["training"].pop("epochs")
    typo = tmp_path / "typo.json"
    typo.write_text(json.dumps(raw))
    with pytest.raises(ConfigurationValidationError, match="training"):
        load_training_config(typo, repo_root=ROOT)

    raw = json.loads((TRAIN_DIR / "default_double_shockley.json").read_text())
    raw["training"]["epochs"] = "15"
    wrong_type = tmp_path / "wrong-type.json"
    wrong_type.write_text(json.dumps(raw))
    with pytest.raises(ConfigurationValidationError, match="epochs"):
        load_training_config(wrong_type, repo_root=ROOT)


def test_profile_reference_hash_is_enforced(tmp_path: Path) -> None:
    raw = json.loads((TRAIN_DIR / "default_double_shockley.json").read_text())
    raw["simulation_ref"]["sha256"] = "0" * 64
    candidate = tmp_path / "bad-hash.json"
    candidate.write_text(json.dumps(raw))
    with pytest.raises(ConfigurationReferenceError, match="SHA-256 mismatch"):
        load_training_config(candidate, repo_root=ROOT)


def test_partial_or_overlapping_composition_is_rejected(tmp_path: Path) -> None:
    raw = json.loads((TRAIN_DIR / "default_double_shockley.json").read_text())
    profile = json.loads(
        (ROOT / raw["simulation_ref"]["path"]).read_text(encoding="utf-8")
    )
    raw["simulation"] = profile["simulation"]
    candidate = tmp_path / "collision.json"
    candidate.write_text(json.dumps(raw))
    with pytest.raises(ConfigurationValidationError):
        load_training_config(candidate, repo_root=ROOT)


def test_pwl_curve_is_a_repository_relative_asset_path() -> None:
    config, _ = load_training_config(
        TRAIN_DIR / "digits_pwl.json",
        repo_root=ROOT,
    )
    curve = config.simulation["updater"]["curve"]
    assert curve == "data/assets/experimental_curve_voff_0.8_200_points.npz"
    assert (ROOT / curve).is_file()
    assert config.simulation["updater"]["extrapolation"] == "clamp"


def test_paper_mnist_values_remain_explicit() -> None:
    config, _ = load_training_config(
        TRAIN_DIR / "mnist_paper_double_shockley.json",
        repo_root=ROOT,
    )
    assert config.layer_shapes == [(2, 28, 28), (100,), (20,)]
    assert config.num_iterations == 4
    assert config.num_epochs == 100
    assert config.input_gain == 50.0
    assert config.learning_rates == [0.15, 0.08, 0.05]
    assert config.scheduler_gamma == 0.99
    assert config.dataset["preprocessing"] == {
        "dtype": "float32",
        "method": "normalized_tensor",
        "mean": 0.1307,
        "standard_deviation": 0.3081,
        "scale": 0.3,
    }


def test_mnist_transform_uses_only_explicit_preprocessing() -> None:
    from torchvision import transforms

    config, _ = load_training_config(
        TRAIN_DIR / "mnist_paper_double_shockley.json",
        repo_root=ROOT,
    )
    transform = _build_mnist_transform(config.dataset["preprocessing"], transforms)
    assert len(transform.transforms) == 3
    normalize = transform.transforms[1]
    assert tuple(normalize.mean) == (0.1307,)
    assert tuple(normalize.std) == (0.3081,)
