#!/usr/bin/env python3
"""Refresh the versioned-artifact SHA256 index after an intentional update."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_PATH = ROOT / "data" / "checksums.sha256"
INVENTORY_PATH = ROOT / "data" / "release_inventory.txt"
CHECKSUM_RELATIVE_PATH = CHECKSUM_PATH.relative_to(ROOT).as_posix()
INVENTORY_RELATIVE_PATH = INVENTORY_PATH.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_release_paths(*, repo_root: Path = ROOT) -> list[str]:
    """Return the exact staged/tracked release inventory.

    Untracked files are deliberately absent: callers must stage every intended
    release file before refreshing the inventory.  A tracked-but-deleted path is
    an error instead of being silently omitted from the integrity boundary.
    """

    root = repo_root.expanduser().resolve()
    result = subprocess.run(
        ("git", "ls-files", "--cached", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    tracked: list[str] = []
    for encoded in filter(None, result.stdout.split(b"\0")):
        try:
            relative = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Expected every tracked release path to be valid UTF-8."
            ) from exc
        _validate_release_path(relative)
        if relative == CHECKSUM_RELATIVE_PATH:
            continue
        tracked.append(relative)

    if len(set(tracked)) != len(tracked):
        raise RuntimeError("Expected Git to return each tracked release path once.")
    tracked.sort()
    if INVENTORY_RELATIVE_PATH not in tracked:
        raise RuntimeError(
            f"Expected {INVENTORY_RELATIVE_PATH} to be tracked before refreshing "
            "release integrity metadata. Stage every intended release file first."
        )
    for relative in tracked:
        path = root / PurePosixPath(relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                "Expected every tracked release inventory path to be an existing "
                f"regular file. Provided value: {relative!r}."
            )
    return tracked


def versioned_files(*, repo_root: Path = ROOT) -> list[Path]:
    """Compatibility view of :func:`tracked_release_paths` as absolute paths."""

    root = repo_root.expanduser().resolve()
    return [root / PurePosixPath(path) for path in tracked_release_paths(repo_root=root)]


def _validate_release_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        not relative
        or relative != relative.strip()
        or "\\" in relative
        or "\n" in relative
        or "\r" in relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(
            "Expected a normalized, repository-relative UTF-8 release path. "
            f"Provided value: {relative!r}."
        )


def main() -> int:
    relative_paths = tracked_release_paths(repo_root=ROOT)
    inventory_text = "".join(f"{relative}\n" for relative in relative_paths)
    inventory_bytes = inventory_text.encode("utf-8")
    checksums: dict[str, str] = {}
    for relative in relative_paths:
        if relative == INVENTORY_RELATIVE_PATH:
            checksums[relative] = hashlib.sha256(inventory_bytes).hexdigest()
        else:
            checksums[relative] = sha256(ROOT / PurePosixPath(relative))

    INVENTORY_PATH.write_text(inventory_text, encoding="utf-8")
    CHECKSUM_PATH.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    print(f"indexed {len(checksums)} tracked release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
