from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from model.resistive.minimizer import (
    ConfigurableExponentialSingleDiodeUpdater,
    ExperimentalIVCurveUpdater,
    MinimizerSettings,
    QuadraticMinimizer,
)
from model.resistive.layer import NonlinearResistiveLayer
from repro.iv_data import load_iv_data


ROOT = Path(__file__).resolve().parents[1]
MINIMIZER_PATH = ROOT / "repro" / "vendor" / "model" / "resistive" / "minimizer.py"


def _settings(**overrides: object) -> MinimizerSettings:
    values: dict[str, object] = {
        "rel_tol": 1e-5,
        "vn_tol": 1e-6,
        "use_polish": True,
        "max_newton_iters": 32,
        "z_thresh": 1e10,
        "exp_clip": 165.0,
        "experimental_newton_tol": 1e-5,
        "b_clamp": 1e6,
        "pwl_extrapolation": "clamp",
        "pwl_nonconvergence_policy": "accept_last",
        "lambertw_backend": "torchlambertw",
        "lambertw_asymptotic_terms": 4,
        "single_diode_min_a": 1e-30,
        "single_diode_polish_abs_tol": 1e-6,
        "single_diode_polish_rel_tol": 1e-6,
        "double_diode_polish_residual_tol": 1e-6,
    }
    values.update(overrides)
    return MinimizerSettings(**values)


def _minimizer_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "fn": object(),
        "free_layers": [],
        "num_iterations": 1,
        "mode": "asynchronous",
        "non_linearity": "linear",
        "quadratic_diode_param": {},
        "exponential_diode_param": {},
        "hard_sigmoid_param": {},
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "minimizer_settings": _settings(),
        "iv_data": None,
        "double_diode_updater": None,
        "adaptive_equilibrium": False,
        "overrelaxation_factor": 1.0,
        "single_diode_updater": None,
        "damping": None,
        "experimental_newton_max_steps": None,
    }
    values.update(overrides)
    return values


class _QuadraticFunctionStub:
    def grad_layer_fn(self, layer):
        return lambda: torch.zeros_like(layer.state)

    def a_coef_fn(self, layer):
        return lambda: torch.ones_like(layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(layer.state)


def test_original_resistive_minimizer_module_is_canonical() -> None:
    expected_module = "model.resistive.minimizer"
    assert QuadraticMinimizer.__module__ == expected_module
    assert MinimizerSettings.__module__ == expected_module
    assert ConfigurableExponentialSingleDiodeUpdater.__module__ == expected_module
    assert not (ROOT / "repro" / "vendor" / "labs" / "custom_minimizer.py").exists()

    for path in (ROOT / "repro" / "train.py", ROOT / "repro" / "digits_validate.py"):
        assert "labs.custom_minimizer" not in path.read_text(encoding="utf-8")


def test_vendored_model_does_not_import_reproduction_layer() -> None:
    tree = ast.parse(MINIMIZER_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any(name == "repro" or name.startswith("repro.") for name in imported_modules)


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_minimizer_settings_require_boolean_polish(value: object) -> None:
    with pytest.raises(ValueError, match="Expected use_polish to be a boolean"):
        _settings(use_polish=value)


def test_minimizer_requires_explicit_complete_settings() -> None:
    parameter = inspect.signature(QuadraticMinimizer).parameters["minimizer_settings"]
    assert parameter.default is inspect.Parameter.empty

    with pytest.raises(TypeError, match="explicit minimizer_settings"):
        QuadraticMinimizer(**_minimizer_kwargs(minimizer_settings=None))

    with pytest.raises(TypeError, match=r"missing .* 'b_clamp'"):
        MinimizerSettings(
            rel_tol=1e-5,
            vn_tol=1e-6,
            use_polish=True,
            max_newton_iters=32,
            z_thresh=1e10,
            exp_clip=165.0,
            experimental_newton_tol=1e-5,
        )


def test_minimizer_has_no_hidden_runtime_policy_defaults() -> None:
    signature = inspect.signature(QuadraticMinimizer)
    policy_fields = {
        "hard_sigmoid_param",
        "minimizer_settings",
        "iv_data",
        "double_diode_updater",
        "adaptive_equilibrium",
        "overrelaxation_factor",
        "single_diode_updater",
        "damping",
        "experimental_newton_max_steps",
    }
    assert all(
        signature.parameters[name].default is inspect.Parameter.empty
        for name in policy_fields
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rel_tol", "1e-5", "Expected rel_tol to be a JSON number"),
        ("max_newton_iters", "32", "Expected max_newton_iters to be an integer"),
        (
            "lambertw_asymptotic_terms",
            5,
            "Expected lambertw_asymptotic_terms to be between 1 and 4",
        ),
        (
            "pwl_extrapolation",
            "CLAMP",
            "Expected pwl_extrapolation to be exactly one of",
        ),
    ],
)
def test_minimizer_settings_reject_coercions_and_noncanonical_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings(**{field: value})


def test_minimizer_source_has_no_hidden_numerical_environment_or_aliases() -> None:
    source = MINIMIZER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "DRN_B_CLAMP",
        "LABS_IV_EXTRAPOLATION",
        "NONLINEARITY_ALIASES",
        "UPDATER_ALIASES",
        "DEFAULT_MINIMIZER_SETTINGS",
        "overrelated",
    ):
        assert forbidden not in source


def test_configured_values_override_hostile_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRN_B_CLAMP", "1")
    monkeypatch.setenv("LABS_IV_EXTRAPOLATION", "linear")
    layer = NonlinearResistiveLayer(
        (2,),
        batch_size=1,
        non_linearity="experimental",
    )
    fn = _QuadraticFunctionStub()
    settings = _settings(b_clamp=1e6, pwl_extrapolation="clamp")

    diode = ConfigurableExponentialSingleDiodeUpdater(
        layer,
        fn,
        {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.8},
        settings,
    )
    pwl = ExperimentalIVCurveUpdater(
        layer,
        fn,
        torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
        damping=1.0,
        max_newton_steps=32,
        newton_tol=1e-5,
        extrapolation=settings.pwl_extrapolation,
        nonconvergence_policy=settings.pwl_nonconvergence_policy,
    )

    assert diode._b_clip == 1e6
    assert pwl._extrapolation == "clamp"
    assert pwl._clamp is True


@pytest.mark.parametrize(
    "non_linearity",
    [None, "DoubleDiodeExponential", "double-diode-exponential"],
)
def test_minimizer_rejects_noncanonical_nonlinearity(non_linearity: object) -> None:
    with pytest.raises(ValueError, match="Expected non_linearity to be exactly one of"):
        QuadraticMinimizer(**_minimizer_kwargs(non_linearity=non_linearity))


@pytest.mark.parametrize(
    ("non_linearity", "selector_field", "selector"),
    [
        ("double_diode_exponential", "double_diode_updater", None),
        ("double_diode_exponential", "double_diode_updater", "custom"),
        ("double_diode_exponential", "double_diode_updater", "overrelated"),
        ("single_diode_exponential", "single_diode_updater", None),
        ("single_diode_exponential", "single_diode_updater", "overrelated"),
        ("experimental", "double_diode_updater", None),
        ("experimental", "double_diode_updater", "custom"),
    ],
)
def test_minimizer_rejects_missing_and_alias_updater_selectors(
    non_linearity: str,
    selector_field: str,
    selector: object,
) -> None:
    overrides: dict[str, object] = {
        "non_linearity": non_linearity,
        selector_field: selector,
    }
    if non_linearity == "experimental":
        overrides["iv_data"] = object()
    with pytest.raises(ValueError, match=f"Expected {selector_field} to be exactly one of"):
        QuadraticMinimizer(**_minimizer_kwargs(**overrides))


def test_measured_iv_loader_reports_expected_format(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LABS_IV_CURVE_PATH", raising=False)
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, samples=np.arange(4))

    with pytest.raises(
        ValueError,
        match="Expected experimental IV data to contain 'iv' or both 'i' and 'v' arrays",
    ):
        load_iv_data(invalid)


def test_measured_iv_loader_rejects_non_npz_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LABS_IV_CURVE_PATH", raising=False)
    invalid = tmp_path / "curve.npy"
    np.save(invalid, np.zeros((2, 2)))

    with pytest.raises(ValueError, match=r"use a \.npz file"):
        load_iv_data(invalid)


@pytest.mark.parametrize(
    "contents",
    [
        {
            "iv": np.array(
                [[-2.0, 0.0, 3.0], [-1.0, 0.0, 2.0]], dtype=np.float64
            )
        },
        {
            "i": np.array([-2.0, 0.0, 3.0], dtype=np.float32),
            "v": np.array([-1.0, 0.0, 2.0], dtype=np.float32),
        },
    ],
    ids=("iv-array", "paired-arrays"),
)
def test_measured_iv_loader_accepts_supported_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: dict[str, np.ndarray],
) -> None:
    monkeypatch.delenv("LABS_IV_CURVE_PATH", raising=False)
    curve = tmp_path / "curve.npz"
    np.savez(curve, **contents)

    loaded = load_iv_data(curve)

    np.testing.assert_allclose(
        loaded.numpy(),
        np.array([[-2.0, 0.0, 3.0], [-1.0, 0.0, 2.0]]),
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ({"iv": np.zeros((3, 2))}, r"shape \(2, N\)"),
        (
            {"i": np.zeros((1, 2)), "v": np.array([0.0, 1.0])},
            "one-dimensional",
        ),
        (
            {"i": np.array([0.0, 1.0]), "v": np.array([0.0, 1.0, 2.0])},
            "equal lengths",
        ),
        (
            {"iv": np.array([["0", "1"], ["0", "1"]])},
            "real numeric values",
        ),
        ({"iv": np.array([[0.0], [0.0]])}, "at least two samples"),
        (
            {"iv": np.array([[0.0, np.inf], [0.0, 1.0]])},
            "samples to be finite",
        ),
        (
            {"iv": np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])},
            "voltages to be strictly increasing",
        ),
        (
            {"iv": np.array([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])},
            "currents to be nondecreasing",
        ),
    ],
    ids=(
        "iv-shape",
        "paired-dimensionality",
        "paired-lengths",
        "nonnumeric",
        "too-short",
        "nonfinite",
        "unordered-voltage",
        "decreasing-current",
    ),
)
def test_measured_iv_loader_rejects_invalid_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: dict[str, np.ndarray],
    message: str,
) -> None:
    monkeypatch.delenv("LABS_IV_CURVE_PATH", raising=False)
    curve = tmp_path / "invalid-curve.npz"
    np.savez(curve, **contents)

    with pytest.raises(ValueError, match=message):
        load_iv_data(curve)
