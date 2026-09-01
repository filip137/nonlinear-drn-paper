"""Strict orchestration manifest and checksum-index handling."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path, PurePosixPath
import re
from typing import Any

from repro.config import RuntimeConfig, load_runtime_config
from repro.strict_config import (
    ConfigReference,
    file_sha256,
    load_validated_json,
    resolve_reference,
    validate_document,
)


_CHECKSUM_RELATIVE_PATH = "data/checksums.sha256"
_INVENTORY_RELATIVE_PATH = "data/release_inventory.txt"


@dataclass(frozen=True)
class ReproJob:
    job_id: str
    group: str
    base_config: ConfigReference
    assets: dict[str, ConfigReference]
    equilibrium_override: dict[str, Any]
    provenance: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReproJob":
        return cls(
            job_id=payload["job_id"],
            group=payload["group"],
            base_config=ConfigReference.from_mapping(payload["base_config"]),
            assets={
                name: ConfigReference.from_mapping(reference)
                for name, reference in payload["assets"].items()
            },
            equilibrium_override=dict(payload["overrides"]["equilibrium"]),
            provenance=dict(payload["provenance"]),
        )

    @property
    def config(self) -> str:
        return self.base_config.path

    @property
    def weights(self) -> str:
        return self.assets["weights"].path

    @property
    def reference_npz(self) -> str | None:
        reference = self.assets.get("reference_states")
        return None if reference is None else reference.path

    def config_path(self, root: Path) -> Path:
        resolved = resolve_reference(
            self.base_config,
            repo_root=root,
            schema="replay-v2.schema.json",
        )
        return resolved.absolute_path

    def weights_path(self, root: Path) -> Path:
        return _resolve_asset(self.assets["weights"], root=root)

    def reference_path(self, root: Path) -> Path | None:
        reference = self.assets.get("reference_states")
        return None if reference is None else _resolve_asset(reference, root=root)

    def resolved_config(self, root: Path) -> RuntimeConfig:
        return load_runtime_config(
            self.config_path(root),
            pack_root=root,
            equilibrium_override=self.equilibrium_override,
        )

    def output_dir(self, root: Path) -> Path:
        return root / "outputs" / "validation" / Path(*self.job_id.split("/"))

    def comparison_dir(self, root: Path) -> Path:
        return root / "outputs" / "comparisons" / Path(*self.job_id.split("/"))


@dataclass(frozen=True)
class PackManifest:
    jobs: list[ReproJob]
    demo: dict[str, Any]
    comparison: dict[str, Any]
    execution_ref: ConfigReference
    checksums: dict[str, str]
    raw: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "PackManifest":
        payload = load_validated_json(
            root / "data" / "manifest.json",
            "manifest-v2.schema.json",
            repo_root=root,
        )
        jobs = [ReproJob.from_dict(item) for item in payload["jobs"]]
        _validate_manifest_relations(payload, jobs)
        return cls(
            jobs=jobs,
            demo=dict(payload["demo"]),
            comparison=dict(payload["comparison"]),
            execution_ref=ConfigReference.from_mapping(payload["execution_ref"]),
            checksums=_load_checksum_index(root / "data" / "checksums.sha256"),
            raw=payload,
        )

    def jobs_for_group(self, group: str) -> list[ReproJob]:
        return [job for job in self.jobs if job.group == group]

    def demo_job(self) -> ReproJob:
        job_id = self.demo["job_id"]
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        raise LookupError(
            f"Bundled demo job {job_id!r} was not found in data/manifest.json; "
            "restore the repository's versioned data bundle before running the demo."
        )

    def execution_profile(self, root: Path) -> dict[str, Any]:
        resolved = resolve_reference(
            self.execution_ref,
            repo_root=root,
            schema="execution-v2.schema.json",
        )
        return dict(resolved.document["execution"])

    def resolved_job_config(self, root: Path, job: ReproJob) -> RuntimeConfig:
        base = job.resolved_config(root)
        document = copy.deepcopy(base.document)
        document["execution"] = self.execution_profile(root)
        validate_document(document, "replay-v2.schema.json", repo_root=root)
        if (
            document["model"]["state_dtype"]
            != document["execution"]["backend"]["default_dtype"]
        ):
            raise ValueError(
                "Expected replay model.state_dtype to match "
                "execution.backend.default_dtype."
            )
        return RuntimeConfig(document=document)

    def verify_checksums(self, root: Path | None = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[1]
        resolved_root = root.expanduser().resolve()
        inventory = _load_release_inventory(
            resolved_root / _INVENTORY_RELATIVE_PATH
        )
        if not self.checksums:
            raise ValueError(
                "Expected data/checksums.sha256 to contain a nonempty release index."
            )
        indexed = set(self.checksums)
        expected = set(inventory)
        missing = sorted(expected - indexed)
        unexpected = sorted(indexed - expected)
        if missing or unexpected:
            raise ValueError(
                "Expected data/checksums.sha256 to cover the explicit release "
                "inventory exactly: "
                f"missing={missing}, unexpected={unexpected}."
            )
        for rel_path, expected in sorted(self.checksums.items()):
            relative = Path(rel_path)
            if relative.is_absolute():
                raise ValueError(
                    f"Expected a repository-relative checksum path. Provided value: {rel_path!r}."
                )
            unresolved = resolved_root / relative
            if unresolved.is_symlink():
                raise ValueError(
                    "Expected checksummed release paths to be regular files, not "
                    f"symbolic links. Provided value: {rel_path!r}."
                )
            path = unresolved.resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"Expected checksum path to stay inside the repository: {rel_path!r}."
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(
                    f"Expected checksummed file to exist. Provided value: {path}"
                )
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    "Expected SHA256 checksum to match data/checksums.sha256. "
                    f"Provided value: path={path}, expected={expected}, actual={actual}."
                )

    def verify_references(self, root: Path | None = None) -> None:
        """Resolve every scientific reference and every effective job config."""

        if root is None:
            root = Path(__file__).resolve().parents[1]
        self.execution_profile(root)
        verified_assets: set[tuple[str, str]] = set()
        for job in self.jobs:
            self.resolved_job_config(root, job)
            for reference in job.assets.values():
                key = (reference.path, reference.sha256)
                if key not in verified_assets:
                    _resolve_asset(reference, root=root)
                    verified_assets.add(key)


def _resolve_asset(reference: ConfigReference, *, root: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    relative = Path(reference.path)
    if relative.is_absolute():
        raise ValueError(
            f"Expected asset path to be repository-relative. Provided value: {reference.path!r}."
        )
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Expected asset path to stay inside the repository: {reference.path!r}."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected referenced asset to exist. Provided value: {path}."
        )
    actual = file_sha256(path)
    if actual != reference.sha256:
        raise ValueError(
            "Expected referenced asset SHA256 to match the manifest. "
            f"Provided value: path={path}, expected={reference.sha256}, actual={actual}."
        )
    return path


def _load_checksum_index(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line:
            continue
        try:
            digest, relative = raw_line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed checksum index line {line_number} in {path}: {raw_line!r}."
            ) from exc
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
        ):
            raise ValueError(
                f"Malformed checksum index line {line_number} in {path}: {raw_line!r}."
            )
        if relative in checksums:
            raise ValueError(
                f"Duplicate checksum path {relative!r} on line {line_number} in {path}."
            )
        checksums[relative] = digest
    return checksums


def _load_release_inventory(path: Path) -> tuple[str, ...]:
    """Load the sorted, explicit list of files inside the checksum boundary."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FileNotFoundError(
            f"Could not read explicit release inventory {path}: {exc}."
        ) from exc
    if not lines:
        raise ValueError(f"Expected release inventory {path} to be nonempty.")
    for line_number, relative in enumerate(lines, start=1):
        _validate_release_inventory_path(
            relative,
            source=path,
            line_number=line_number,
        )
    if len(set(lines)) != len(lines):
        raise ValueError(f"Expected release inventory {path} paths to be unique.")
    if lines != sorted(lines):
        raise ValueError(f"Expected release inventory {path} paths to be sorted.")
    if _CHECKSUM_RELATIVE_PATH in lines:
        raise ValueError(
            "data/checksums.sha256 cannot checksum itself and must not appear in "
            f"release inventory {path}."
        )
    if _INVENTORY_RELATIVE_PATH not in lines:
        raise ValueError(
            f"Expected release inventory {path} to include its own path "
            f"{_INVENTORY_RELATIVE_PATH!r} so its exact bytes are checksummed."
        )
    return tuple(lines)


def _validate_release_inventory_path(
    relative: str,
    *,
    source: Path,
    line_number: int,
) -> None:
    pure = PurePosixPath(relative)
    if (
        not relative
        or relative != relative.strip()
        or "\\" in relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(
            "Expected a normalized repository-relative path at "
            f"{source}:{line_number}, got {relative!r}."
        )


def _validate_manifest_relations(
    payload: dict[str, Any], jobs: list[ReproJob]
) -> None:
    identifiers = [job.job_id for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Expected every manifest job_id to be unique.")
    for job in jobs:
        parts = job.job_id.split("/")
        if any(
            not part
            or part in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None
            for part in parts
        ):
            raise ValueError(
                "Expected job_id to contain safe nonempty path segments. "
                f"Provided value: {job.job_id!r}."
            )
        if job.group in {"error_vs_iter", "vol_tol"} and job.reference_npz is None:
            raise ValueError(
                f"Expected {job.group} job {job.job_id!r} to declare reference_states."
            )
    demo_id = payload["demo"]["job_id"]
    if identifiers.count(demo_id) != 1:
        raise ValueError(
            "Expected manifest.demo.job_id to identify exactly one job. "
            f"Provided value: {demo_id!r}."
        )


__all__ = ["PackManifest", "ReproJob"]
