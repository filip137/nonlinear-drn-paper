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
