from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from model.resistive.minimizer import (
    ConfigurableExponentialSingleDiodeUpdater,
    MinimizerSettings,
    QuadraticMinimizer,
)
from repro.iv_data import load_iv_data


ROOT = Path(__file__).resolve().parents[1]
MINIMIZER_PATH = ROOT / "repro" / "vendor" / "model" / "resistive" / "minimizer.py"


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
        MinimizerSettings(
            rel_tol=1e-5,
            vn_tol=1e-6,
            use_polish=value,
            max_newton_iters=32,
            z_thresh=1e10,
            exp_clip=165.0,
        )


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
