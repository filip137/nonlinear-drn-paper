"""Verify the installed environment against the checked-in requirement graph."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


_IMPORT_NAMES = {
    "jsonschema": "jsonschema",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "packaging": "packaging",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "sympy": "sympy",
    "threadpoolctl": "threadpoolctl",
    "torch": "torch",
    "torchlambertw": "torchlambertw",
    "torchvision": "torchvision",
}

_IGNORED_REQUIREMENT_OPTIONS = (
    "--extra-index-url",
    "--find-links",
    "--index-url",
    "--no-index",
    "--only-binary",
    "--prefer-binary",
    "--trusted-host",
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_PORTABLE_METADATA_PACKAGES = {"torch", "torchvision"}
_WHEEL_LOCAL_TAG_PATTERN = re.compile(r"^(?:cpu|cu[0-9]+)$")


class RequirementsError(RuntimeError):
    """A checked-in requirements graph is ambiguous or malformed."""


@dataclass(frozen=True)
class RequirementPin:
    """One active exact distribution or immutable Git requirement."""

    name: str
    version: str | None
    direct_url: str | None
    source: Path
    line_number: int

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line_number}"


@dataclass(frozen=True)
class GitProvenance:
    """Expected PEP 610 provenance derived from an immutable Git URL."""

    url: str
    commit_id: str


def requirement_pins(
    requirements_file: str | Path = "requirements.txt",
    *,
    marker_environment: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, RequirementPin]:
    """Resolve recursive requirement files into active, exact pins.

    Includes are resolved relative to the file containing each ``-r`` line.
    Every requirement is checked for exactness even when its marker is inactive,
    so both supported Python stacks remain locked.
    """

    root = _resolved_repo_root(repo_root)
    entrypoint = Path(requirements_file).expanduser()
    if not entrypoint.is_absolute():
        entrypoint = root / entrypoint
    environment = default_environment()
    if marker_environment is not None:
        environment.update(marker_environment)

    active: dict[str, RequirementPin] = {}
    _read_requirements(
        entrypoint.resolve(),
        environment=environment,
        active=active,
        include_stack=(),
    )
    return dict(sorted(active.items()))


def expected_versions(
    requirements_file: str | Path = "requirements.txt",
    *,
    marker_environment: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Return active exact distribution versions from the requirements graph."""

    return {
        name: pin.version
        for name, pin in requirement_pins(
            requirements_file,
            marker_environment=marker_environment,
            repo_root=repo_root,
        ).items()
        if pin.version is not None
    }


def verify_environment(
    *,
    requirements_file: str | Path = "requirements.txt",
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Validate exact pins, immutable direct URLs, and package imports."""

    root = _resolved_repo_root(repo_root)
    if (root / "pyproject.toml").is_file():
        verify_project_dependency_sync(repo_root=root)
    pins = requirement_pins(requirements_file, repo_root=root)
    installed: dict[str, str] = {}
    problems: list[str] = []

    for name, pin in pins.items():
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} is missing")
            continue
        actual_version = distribution.version
        installed[name] = actual_version
        if pin.version is not None and actual_version != pin.version:
            problems.append(
                f"{name}=={pin.version} is required by {pin.location}, "
                f"found {actual_version}"
            )
        if pin.direct_url is not None:
            provenance_problem = _direct_url_problem(distribution, pin)
            if provenance_problem is not None:
                problems.append(provenance_problem)

    if problems:
        raise RuntimeError(
            "Expected the complete exact environment from "
            f"`python -m pip install -r {requirements_file}`. "
            f"Provided: {'; '.join(problems)}."
        )

    summary_distributions = {"numpy", "torch", "torchvision"}
    missing_summary_pins = sorted(summary_distributions - set(pins))
    if missing_summary_pins:
        raise RequirementsError(
            "The selected requirements graph has no active exact pins for "
            f"{missing_summary_pins} under Python {platform.python_version()}."
        )

    import_problems: list[str] = []
    for distribution_name, module_name in sorted(_IMPORT_NAMES.items()):
        if distribution_name not in pins:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # Binary ABI and matched-pair errors are not ImportError.
            import_problems.append(
                f"{distribution_name} {installed[distribution_name]} failed to import "
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


def verify_project_dependency_sync(
    *,
    repo_root: Path | None = None,
    requirements_file: str | Path = "requirements.txt",
    pyproject_file: str | Path = "pyproject.toml",
) -> None:
    """Ensure project metadata is a portable mirror of the canonical CPU lock.

    ``requirements*.txt`` files are the environment authority. Project metadata
    may omit only the ``+cpu``/``+cu...`` local wheel tag for Torch and
    Torchvision because package indexes are selected outside ``pyproject.toml``.
    """

    root = _resolved_repo_root(repo_root)
    pyproject_path = Path(pyproject_file).expanduser()
    if not pyproject_path.is_absolute():
        pyproject_path = root / pyproject_path
    try:
        with pyproject_path.open("rb") as handle:
            metadata = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RequirementsError(f"Could not read project metadata {pyproject_path}: {exc}.") from exc
    project = metadata.get("project")
    dependencies = None if not isinstance(project, dict) else project.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise RequirementsError(
            f"Expected {pyproject_path} project.dependencies to be a string array."
        )

    problems: list[str] = []
    for python_version in ("3.12", "3.13"):
        environment = default_environment()
        environment.update(
            {
                "python_version": python_version,
                "python_full_version": f"{python_version}.0",
            }
        )
        canonical = requirement_pins(
            requirements_file,
            marker_environment=environment,
            repo_root=root,
        )
        portable = _project_dependency_pins(
            dependencies,
            environment=environment,
            source=pyproject_path,
        )
        missing = sorted(set(canonical) - set(portable))
        extra = sorted(set(portable) - set(canonical))
        if missing:
            problems.append(f"Python {python_version} metadata is missing {missing}")
        if extra:
            problems.append(f"Python {python_version} metadata adds {extra}")
        for name in sorted(set(canonical).intersection(portable)):
            locked = canonical[name]
            mirrored = portable[name]
            if not _metadata_pin_matches(name, locked=locked, mirrored=mirrored):
                problems.append(
                    f"Python {python_version} {name} metadata {_pin_value(mirrored)!r} "
                    f"does not mirror lock {_pin_value(locked)!r}"
                )

    if problems:
        unique = list(dict.fromkeys(problems))
        raise RequirementsError(
            "pyproject.toml dependencies drift from canonical requirements.txt: "
            + "; ".join(unique)
            + "."
        )


def _project_dependency_pins(
    dependencies: list[str],
    *,
    environment: dict[str, str],
    source: Path,
) -> dict[str, RequirementPin]:
    active: dict[str, RequirementPin] = {}
    for index, text in enumerate(dependencies, start=1):
        try:
            requirement = Requirement(text)
        except InvalidRequirement as exc:
            raise RequirementsError(
                f"Invalid project dependency {text!r} in {source}: {exc}."
            ) from exc
        pin = _require_exact_pin(requirement, path=source, line_number=index)
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        if pin.name in active:
            raise RequirementsError(
                f"Duplicate active project dependency {pin.name!r} in {source}."
            )
        active[pin.name] = pin
    return active


def _metadata_pin_matches(
    name: str,
    *,
    locked: RequirementPin,
    mirrored: RequirementPin,
) -> bool:
    if locked.direct_url is not None or mirrored.direct_url is not None:
        return (
            locked.direct_url == mirrored.direct_url
            and locked.version == mirrored.version
        )
    assert locked.version is not None
    assert mirrored.version is not None
    if locked.version == mirrored.version:
        return True
    if name not in _PORTABLE_METADATA_PACKAGES or "+" not in locked.version:
        return False
    public, local = locked.version.rsplit("+", maxsplit=1)
    return (
        _WHEEL_LOCAL_TAG_PATTERN.fullmatch(local) is not None
        and mirrored.version == public
    )


def _pin_value(pin: RequirementPin) -> str:
    return pin.direct_url if pin.direct_url is not None else str(pin.version)


def _resolved_repo_root(repo_root: Path | None) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else repo_root.expanduser().resolve()
    )


def _read_requirements(
    path: Path,
    *,
    environment: dict[str, str],
    active: dict[str, RequirementPin],
    include_stack: tuple[Path, ...],
) -> None:
    if path in include_stack:
        cycle = " -> ".join(str(item) for item in (*include_stack, path))
        raise RequirementsError(f"Recursive requirements include cycle: {cycle}.")
    if not path.is_file():
        parent = include_stack[-1] if include_stack else None
        context = "" if parent is None else f" included from {parent}"
        raise RequirementsError(f"Requirements file does not exist: {path}{context}.")

    stack = (*include_stack, path)
    for line_number, line in _logical_lines(path):
        if not line or line.startswith("#"):
            continue
        include = _include_target(line)
        if include is not None:
            included = Path(include).expanduser()
            if not included.is_absolute():
                included = path.parent / included
            _read_requirements(
                included.resolve(),
                environment=environment,
                active=active,
                include_stack=stack,
            )
            continue
        if line.startswith(_IGNORED_REQUIREMENT_OPTIONS):
            continue
        if line.startswith(("-c ", "--constraint ", "-e ", "--editable ")):
            raise RequirementsError(
                f"Unsupported requirements directive at {path}:{line_number}: {line!r}."
            )
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise RequirementsError(
                f"Invalid requirement at {path}:{line_number}: {line!r} ({exc})."
            ) from exc

        pin = _require_exact_pin(requirement, path=path, line_number=line_number)
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        previous = active.get(pin.name)
        if previous is not None and (
            previous.version != pin.version or previous.direct_url != pin.direct_url
        ):
            raise RequirementsError(
                f"Conflicting active pins for {pin.name}: {previous.location} and "
                f"{pin.location}."
            )
        active[pin.name] = pin


def _logical_lines(path: Path) -> list[tuple[int, str]]:
    try:
        physical = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RequirementsError(f"Could not read requirements file {path}: {exc}.") from exc
    result: list[tuple[int, str]] = []
    parts: list[str] = []
    start = 0
    for line_number, raw in enumerate(physical, start=1):
        stripped = raw.strip()
        if not parts:
            start = line_number
        continued = stripped.endswith("\\")
        parts.append(stripped[:-1].rstrip() if continued else stripped)
        if continued:
            continue
        logical = _strip_inline_comment(
            " ".join(part for part in parts if part).strip()
        )
        result.append((start, logical))
        parts = []
    if parts:
        raise RequirementsError(
            f"Unterminated line continuation at {path}:{start}."
        )
    return result


def _include_target(line: str) -> str | None:
    for prefix in ("-r ", "--requirement "):
        if line.startswith(prefix):
            target = line[len(prefix) :].strip()
            if not target:
                raise RequirementsError(f"Empty requirements include directive: {line!r}.")
            return target
    if line.startswith("-r") and len(line) > 2:
        return line[2:].strip()
    if line.startswith("--requirement="):
        return line.split("=", maxsplit=1)[1].strip()
    return None


def _strip_inline_comment(line: str) -> str:
    for index, character in enumerate(line):
        if character == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line


def _require_exact_pin(
    requirement: Requirement,
    *,
    path: Path,
    line_number: int,
) -> RequirementPin:
    name = canonicalize_name(requirement.name)
    if requirement.extras:
        raise RequirementsError(
            f"Requirement extras are not allowed in the reproducibility lock at "
            f"{path}:{line_number}: {requirement}."
        )
    if requirement.url is not None:
        _git_provenance(requirement.url, location=f"{path}:{line_number}")
        return RequirementPin(
            name=name,
            version=None,
            direct_url=requirement.url,
            source=path,
            line_number=line_number,
        )

    specifiers = list(requirement.specifier)
    if (
        len(specifiers) != 1
        or specifiers[0].operator not in {"==", "==="}
        or "*" in specifiers[0].version
    ):
        raise RequirementsError(
            "Expected one exact version pin using == or === at "
            f"{path}:{line_number}, got {str(requirement)!r}."
        )
    return RequirementPin(
        name=name,
        version=specifiers[0].version,
        direct_url=None,
        source=path,
        line_number=line_number,
    )


def _git_provenance(url: str, *, location: str) -> GitProvenance:
    if not url.startswith("git+") or "@" not in url:
        raise RequirementsError(
            f"Expected an immutable git+ URL with a full commit at {location}, got {url!r}."
        )
    repository, commit = url.rsplit("@", maxsplit=1)
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise RequirementsError(
            f"Expected a full 40-character Git commit at {location}, got {commit!r}."
        )
    return GitProvenance(url=repository.removeprefix("git+"), commit_id=commit.lower())


def _direct_url_problem(
    distribution: importlib.metadata.Distribution,
    pin: RequirementPin,
) -> str | None:
    assert pin.direct_url is not None
    expected = _git_provenance(pin.direct_url, location=pin.location)
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return (
            f"{pin.name} must come from {pin.direct_url} ({pin.location}), but its "
            "installation has no PEP 610 direct_url.json provenance"
        )
    try:
        direct = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"{pin.name} has invalid direct_url.json provenance ({exc})"
    if not isinstance(direct, dict):
        return f"{pin.name} has non-object direct_url.json provenance"
    actual_url = direct.get("url")
    vcs = direct.get("vcs_info")
    if not isinstance(vcs, dict):
        return f"{pin.name} direct_url.json has no Git vcs_info"
    actual_commit = str(vcs.get("commit_id", "")).lower()
    requested = vcs.get("requested_revision")
    mismatches: list[str] = []
    if actual_url != expected.url:
        mismatches.append(f"url={actual_url!r}, expected {expected.url!r}")
    if vcs.get("vcs") != "git":
        mismatches.append(f"vcs={vcs.get('vcs')!r}, expected 'git'")
    if actual_commit != expected.commit_id:
        mismatches.append(
            f"commit_id={actual_commit!r}, expected {expected.commit_id!r}"
        )
    if requested is not None and str(requested).lower() != expected.commit_id:
        mismatches.append(
            f"requested_revision={requested!r}, expected {expected.commit_id!r}"
        )
    if mismatches:
        return f"{pin.name} direct URL provenance mismatch: {', '.join(mismatches)}"
    return None


__all__ = [
    "GitProvenance",
    "RequirementPin",
    "RequirementsError",
    "expected_versions",
    "requirement_pins",
    "verify_environment",
    "verify_project_dependency_sync",
]
