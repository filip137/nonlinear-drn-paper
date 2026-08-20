from __future__ import annotations

from pathlib import Path

from repro.train import PAPER_NONLINEARITIES, load_training_config


ROOT = Path(__file__).resolve().parents[1]


def test_training_configs_cover_all_paper_nonlinearities() -> None:
    paths = sorted((ROOT / "configs" / "train").glob("digits_*.json"))
    loaded = [load_training_config(path, repo_root=ROOT)[0] for path in paths]
    assert {config.non_linearity for config in loaded} == set(PAPER_NONLINEARITIES)
    for config in loaded:
        assert config.quadratic_diode_param
        assert config.exponential_diode_param
        assert config.hard_sigmoid_param


def test_paper_mnist_config_is_the_reported_drn_xs_shape() -> None:
    config, _ = load_training_config(
        ROOT / "configs" / "train" / "mnist_paper_double_shockley.json",
        repo_root=ROOT,
    )
    assert config.layer_shapes == [(2, 28, 28), (100,), (20,)]
    assert config.non_linearity == "double_diode_exponential"
    assert config.num_iterations == 4
    assert config.num_epochs == 100


def test_pwl_curve_is_portable_and_present() -> None:
    config, _ = load_training_config(
        ROOT / "configs" / "train" / "digits_pwl.json",
        repo_root=ROOT,
    )
    assert config.iv_data_path is not None
    assert Path(config.iv_data_path).is_file()
