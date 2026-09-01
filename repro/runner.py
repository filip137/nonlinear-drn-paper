"""Thin resolve-then-run interface for training configurations."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from repro.strict_config import pretty_json_text


def build_training_config(
    source: str | Path | Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fully expanded v2 snapshot with no implicit experiment policy."""

    root = _resolve_repo_root(repo_root)
    _ensure_vendor_path(root)
    from repro.train import resolve_training_config

    return resolve_training_config(
        source,
        repo_root=root,
        overrides=overrides,
    ).document


def write_training_config(
    source: str | Path | Mapping[str, Any],
    destination: str | Path,
    *,
    repo_root: Path | None = None,
    overrides: Sequence[str] = (),
) -> Path:
    """Materialize a complete executable config before numerical work."""

    root = _resolve_repo_root(repo_root)
    document = build_training_config(
        source,
        repo_root=root,
        overrides=overrides,
    )
    path = Path(destination).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json_text(document), encoding="utf-8")
    return path


def run_drn(
    config: str | Path,
    *,
    repo_root: Path | None = None,
    output_dir: str | Path | None = None,
    overrides: Sequence[str] = (),
    download: bool = False,
):
    """Train from one source/snapshot; all scientific edits are recorded overrides."""

    root = _resolve_repo_root(repo_root)
    _ensure_vendor_path(root)
    from repro.train import run_training

    output = None if output_dir is None else Path(output_dir)
    return run_training(
        root,
        Path(config),
        output_dir=output,
        overrides=overrides,
        download=download,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
) -> int:
    root = _resolve_repo_root(repo_root)
    parser = argparse.ArgumentParser(
        description="Resolve and train strict nonlinear-DRN v2 configurations."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="JSON_POINTER=JSON_VALUE",
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        help="Resolve and write the complete snapshot without training.",
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)

    if args.write_config is not None:
        path = write_training_config(
            args.config,
            args.write_config,
            repo_root=root,
            overrides=args.override,
        )
        print(f"resolved_config: {path}")
        return 0
    result = run_drn(
        args.config,
        repo_root=root,
        output_dir=args.output,
        overrides=args.override,
        download=args.download,
    )
    print(f"training_output: {result.output_dir}")
    return 0


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(repo_root).expanduser().resolve()


def _ensure_vendor_path(root: Path) -> None:
    vendor = str(root / "repro" / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


__all__ = [
    "build_training_config",
    "run_drn",
    "write_training_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
