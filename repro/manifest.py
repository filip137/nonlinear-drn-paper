from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReproJob:
    job_id: str
    group: str
    family: str
    hidden_layers: int
    hidden_size: int
    config: str
    weights: str
    reference_npz: str | None
    num_iterations: int
    rel_tol: str | None = None
    overrelaxation_factor: str | float | None = None
    variant: str | None = None
    source: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReproJob":
        return cls(**payload)

    def config_path(self, root: Path) -> Path:
        return root / self.config

    def weights_path(self, root: Path) -> Path:
        return root / self.weights

    def reference_path(self, root: Path) -> Path | None:
        return None if self.reference_npz is None else root / self.reference_npz

    def output_dir(self, root: Path) -> Path:
        safe = self.job_id.replace("/", "__")
        return root / "outputs" / "validation" / safe

    def comparison_dir(self, root: Path) -> Path:
        safe = self.job_id.replace("/", "__")
        return root / "outputs" / "comparisons" / safe


@dataclass(frozen=True)
class PackManifest:
    jobs: list[ReproJob]
    checksums: dict[str, str]
    raw: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "PackManifest":
        path = root / "data" / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            jobs=[ReproJob.from_dict(item) for item in payload["jobs"]],
            checksums=dict(payload.get("checksums", {})),
            raw=payload,
        )

    def jobs_for_group(self, group: str) -> list[ReproJob]:
        return [job for job in self.jobs if job.group == group]

    def verify_checksums(self, root: Path | None = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[1]
        for rel_path, expected in sorted(self.checksums.items()):
            path = root / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Expected checksummed file to exist. Provided value: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    "Expected SHA256 checksum to match manifest. "
                    f"Provided value: path={path}, expected={expected}, actual={actual}."
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
