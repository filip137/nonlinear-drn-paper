from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


_COMMON_VERSIONS = {
    "matplotlib": "3.10.6",
    "scikit-learn": "1.7.2",
    "scipy": "1.16.0",
    "sympy": "1.13.1",
    "torchlambertw": "0.0.4",
}

_STACKS = {
    (3, 12): {
        "numpy": "2.0.1",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
    },
    (3, 13): {
        "numpy": "2.1.3",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
    },
}

_IMPORT_NAMES = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "sympy": "sympy",
    "torch": "torch",
    "torchlambertw": "torchlambertw",
    "torchvision": "torchvision",
}


def _base_version(version: str) -> str:
    """Ignore CUDA/CPU local tags while retaining the pinned public version."""

    return version.split("+", maxsplit=1)[0]


def expected_versions() -> dict[str, str]:
    python_key = sys.version_info[:2]
    if python_key not in _STACKS:
        raise RuntimeError(
            "Expected Python 3.12 or 3.13. "
            f"Provided Python {platform.python_version()}."
        )
    return {**_COMMON_VERSIONS, **_STACKS[python_key]}


def verify_environment() -> dict[str, str]:
    """Validate the pinned packages and import their compiled extensions."""

    expected = expected_versions()
    installed: dict[str, str] = {}
    problems: list[str] = []

    for distribution, expected_version in sorted(expected.items()):
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{distribution} is missing")
            continue
        installed[distribution] = actual_version
        if _base_version(actual_version) != expected_version:
            problems.append(
                f"{distribution}=={expected_version} is required, found {actual_version}"
            )

    if problems:
        raise RuntimeError(
            "Expected the complete pinned environment from "
            "`python -m pip install -r requirements.txt`. "
            f"Provided: {'; '.join(problems)}."
        )

    import_problems: list[str] = []
    for distribution, module_name in _IMPORT_NAMES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # Binary ABI and matched-pair errors are not ImportError.
            import_problems.append(
                f"{distribution} {installed[distribution]} failed to import "
                f"({type(exc).__name__}: {exc})"
            )

    if import_problems:
        raise RuntimeError(
            "Expected every pinned scientific package to import successfully. "
            f"Provided: {'; '.join(import_problems)}."
        )

    return {
        "python": platform.python_version(),
        "numpy": installed["numpy"],
        "torch": installed["torch"],
        "torchvision": installed["torchvision"],
    }
