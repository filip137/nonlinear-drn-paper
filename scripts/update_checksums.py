#!/usr/bin/env python3
"""Refresh the versioned-artifact SHA256 index after an intentional update."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
CHECKSUM_PATH = ROOT / "data" / "checksums.sha256"
EXCLUDED_FILES = {MANIFEST_PATH, CHECKSUM_PATH}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versioned_files() -> list[Path]:
    result = subprocess.run(
        ("git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [
        ROOT / relative.decode("utf-8")
        for relative in result.stdout.split(b"\0")
        if relative
    ]
    return [path for path in sorted(paths) if path not in EXCLUDED_FILES]


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checksums = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in versioned_files()
    }
    manifest.update(
        {
            "name": "nonlinear-drn-paper",
            "description": (
                "Standalone training, numerical replay, and figure/table reproduction "
                "artifact for the nonlinear DRN paper."
            ),
            "created_by": "scripts/update_checksums.py",
            "notes": [
                "Training supports the single-Shockley, double-Shockley, and measured/PWL nonlinearities.",
                "Timing figures use curated CSVs because wall-clock timing is machine-dependent.",
                "SPICE reference states are bundled; SPICE netlist regeneration is out of scope.",
            ],
            "checksums": checksums,
        }
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    CHECKSUM_PATH.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    print(f"indexed {len(checksums)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
