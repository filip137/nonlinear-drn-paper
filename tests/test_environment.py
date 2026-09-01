from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.execution import apply_execution_profile
from repro.environment import (
    RequirementsError,
    expected_versions,
    requirement_pins,
    verify_environment,
    verify_project_dependency_sync,
)
from repro.strict_config import load_validated_json


ROOT = Path(__file__).resolve().parents[1]
TORCHLAMBERTW_COMMIT = "d365cac4d9a3e6074a03709cf865b4896e3f3cd4"


class _FakeDistribution:
    def __init__(self, version: str, direct_url: dict | None = None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


def _marker(python_version: str) -> dict[str, str]:
    return {
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
    }


def _git_metadata(commit: str = TORCHLAMBERTW_COMMIT) -> dict:
    return {
        "url": "https://github.com/gmgeorg/torchlambertw.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": commit,
            "requested_revision": commit,
        },
    }


def test_repository_requirement_graph_selects_exact_marker_specific_wheels() -> None:
    cpu_312 = expected_versions(
        "requirements.txt",
        marker_environment=_marker("3.12"),
        repo_root=ROOT,
    )
    cpu_313 = expected_versions(
        "requirements.txt",
        marker_environment=_marker("3.13"),
        repo_root=ROOT,
    )
    cuda_313 = expected_versions(
        "requirements-cuda.txt",
        marker_environment=_marker("3.13"),
        repo_root=ROOT,
    )

    assert cpu_312["numpy"] == "2.0.1"
    assert cpu_312["torch"] == "2.5.1+cpu"
    assert cpu_312["torchvision"] == "0.20.1+cpu"
    assert cpu_313["numpy"] == "2.1.3"
    assert cpu_313["torch"] == "2.6.0+cpu"
    assert cpu_313["torchvision"] == "0.21.0+cpu"
    assert cuda_313["torch"] == "2.6.0+cu124"
    assert cuda_313["torchvision"] == "0.21.0+cu124"
    assert cpu_313["packaging"] == "26.3"


def test_project_metadata_is_a_checked_portable_mirror_of_canonical_locks() -> None:
    verify_project_dependency_sync(repo_root=ROOT)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "requirements*.txt are the canonical environment locks" in pyproject


def test_project_metadata_version_drift_fails_synchronization(tmp_path: Path) -> None:
    for name in ("requirements.txt", "requirements-common.txt"):
        (tmp_path / name).write_text(
            (ROOT / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        metadata.replace('"scipy==1.16.0"', '"scipy==1.15.0"'),
        encoding="utf-8",
    )

    with pytest.raises(
        RequirementsError,
        match=r"scipy metadata '1\.15\.0'.*lock '1\.16\.0'",
    ):
        verify_project_dependency_sync(repo_root=tmp_path)


def test_recursive_requirements_are_relative_and_marker_aware(tmp_path: Path) -> None:
    nested = tmp_path / "locks"
    nested.mkdir()
    (tmp_path / "requirements.txt").write_text(
        "--extra-index-url https://example.invalid/simple\n"
        "-rlocks/scientific.txt  # resolved relative to this file\n"
        'torch==2.5.1+cpu; python_version == "3.12"\n'
        'torch==2.6.0+cpu; python_version == "3.13"\n',
        encoding="utf-8",
    )
    (nested / "scientific.txt").write_text(
        "-r versions/numpy.txt\nscipy==1.16.0\n",
        encoding="utf-8",
    )
    versions = nested / "versions"
    versions.mkdir()
    (versions / "numpy.txt").write_text(
        'numpy==2.0.1; python_version == "3.12"\n'
        'numpy==2.1.3; python_version == "3.13"\n',
        encoding="utf-8",
    )

    pins = requirement_pins(
        "requirements.txt",
        marker_environment=_marker("3.13"),
        repo_root=tmp_path,
    )

    assert {name: pin.version for name, pin in pins.items()} == {
        "numpy": "2.1.3",
        "scipy": "1.16.0",
        "torch": "2.6.0+cpu",
    }
    assert pins["numpy"].source == (versions / "numpy.txt").resolve()


def test_inactive_marker_branches_must_still_be_exactly_pinned(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        'numpy>=2; python_version == "3.12"\n'
        'numpy==2.1.3; python_version == "3.13"\n',
        encoding="utf-8",
    )

    with pytest.raises(RequirementsError, match="exact version pin.*numpy>=2"):
        requirement_pins(
            requirements,
            marker_environment=_marker("3.13"),
            repo_root=tmp_path,
        )


def test_recursive_include_cycles_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("-r second.txt\n", encoding="utf-8")
    second.write_text("-r first.txt\n", encoding="utf-8")

    with pytest.raises(RequirementsError, match="include cycle"):
        requirement_pins(first, repo_root=tmp_path)


def test_conflicting_active_pins_are_rejected(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.0.1\nnumpy==2.1.3\n", encoding="utf-8")

    with pytest.raises(RequirementsError, match="Conflicting active pins for numpy"):
        requirement_pins(requirements, repo_root=tmp_path)


def test_git_direct_reference_requires_full_immutable_commit(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "torchlambertw @ git+https://github.com/gmgeorg/torchlambertw.git@main\n",
        encoding="utf-8",
    )

    with pytest.raises(RequirementsError, match="40-character Git commit"):
        requirement_pins(requirements, repo_root=tmp_path)


def test_direct_url_requirement_is_provenance_not_a_guessed_version() -> None:
    pins = requirement_pins("requirements.txt", repo_root=ROOT)
    pin = pins["torchlambertw"]

    assert pin.version is None
    assert pin.direct_url == (
        "git+https://github.com/gmgeorg/torchlambertw.git@" + TORCHLAMBERTW_COMMIT
    )
    assert "torchlambertw" not in expected_versions("requirements.txt", repo_root=ROOT)


def test_verifier_compares_local_wheel_tags_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.6.0+cpu\n", encoding="utf-8")
    monkeypatch.setattr(
        "repro.environment.importlib.metadata.distribution",
        lambda name: _FakeDistribution("2.6.0+cu124"),
    )

    with pytest.raises(RuntimeError, match=r"torch==2\.6\.0\+cpu.*2\.6\.0\+cu124"):
        verify_environment(requirements_file=requirements, repo_root=tmp_path)


def test_verifier_accepts_matching_direct_url_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "numpy==2.1.3\n"
        "torch==2.6.0+cu124\n"
        "torchvision==0.21.0+cu124\n"
        "torchlambertw @ git+https://github.com/gmgeorg/torchlambertw.git@"
        f"{TORCHLAMBERTW_COMMIT}\n",
        encoding="utf-8",
    )
    distributions = {
        "numpy": _FakeDistribution("2.1.3"),
        "torch": _FakeDistribution("2.6.0+cu124"),
        "torchvision": _FakeDistribution("0.21.0+cu124"),
        "torchlambertw": _FakeDistribution("0.0.4", _git_metadata()),
    }
    monkeypatch.setattr(
        "repro.environment.importlib.metadata.distribution",
        lambda name: distributions[name],
    )
    monkeypatch.setattr("repro.environment.importlib.import_module", lambda name: object())

    result = verify_environment(requirements_file=requirements, repo_root=tmp_path)

    assert result["numpy"] == "2.1.3"
    assert result["torch"] == "2.6.0+cu124"
    assert result["torchvision"] == "0.21.0+cu124"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        _git_metadata("0" * 40),
        {
            **_git_metadata(),
            "url": "https://github.com/example/fork.git",
        },
    ],
)
def test_verifier_rejects_missing_or_wrong_direct_url_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict | None,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "torchlambertw @ git+https://github.com/gmgeorg/torchlambertw.git@"
        f"{TORCHLAMBERTW_COMMIT}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "repro.environment.importlib.metadata.distribution",
        lambda name: _FakeDistribution("0.0.4", metadata),
    )

    with pytest.raises(RuntimeError, match="direct_url|provenance mismatch"):
        verify_environment(requirements_file=requirements, repo_root=tmp_path)


def test_cuda_workspace_policy_cannot_be_changed_after_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_validated_json(
        ROOT / "configs" / "execution" / "reference_cuda.json",
        "execution-v2.schema.json",
        repo_root=ROOT,
    )["execution"]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr("repro.execution.torch.cuda.is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="after CUDA was initialized"):
        apply_execution_profile(profile)
