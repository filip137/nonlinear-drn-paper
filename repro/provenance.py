"""Build machine-readable receipts for scientific runs.

Receipts describe what actually ran; they never supply configuration values.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_info
from repro.strict_config import document_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(document: Any) -> str:
    return document_sha256(document)


def sha256_arrays(*arrays: np.ndarray) -> str:
    """Hash array identity, shape, dtype, and C-order bytes deterministically."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_run_receipt(
    *,
    repo_root: Path,
    resolved_config: Mapping[str, Any],
    execution: Mapping[str, Any],
    device: torch.device,
    assets: Mapping[str, Path] | None = None,
    source_documents: Sequence[Mapping[str, Any]] = (),
    data_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete, JSON-serializable provenance receipt."""

    root = repo_root.expanduser().resolve()
    asset_receipt: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted((assets or {}).items()):
        path = raw_path.expanduser().resolve()
        asset_receipt[name] = {
            "path": _relative_or_absolute(path, root),
            "sha256": sha256_file(path),
        }

    receipt: dict[str, Any] = {
        "receipt_version": 1,
        "resolved_config_sha256": sha256_json(resolved_config),
        "source_documents": [dict(item) for item in source_documents],
        "assets": asset_receipt,
        "source_tree": _git_receipt(root),
        "platform": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable": sys.executable,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _installed_packages(),
        "package_direct_urls": _installed_direct_urls(),
        "execution": dict(execution),
        "runtime": _torch_receipt(device),
        "native_threadpools": threadpool_info(),
        "data_fingerprint": data_fingerprint,
        "split_fingerprint": split_fingerprint,
    }
    if extra:
        receipt["run"] = dict(extra)
    return receipt


def _git_receipt(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    dirty_digest: str | None = None
    if commit is not None and status is not None:
        digest = hashlib.sha256()
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--", "."),
            cwd=root,
            check=False,
            capture_output=True,
        )
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=root,
            check=False,
            capture_output=True,
        )
        if diff.returncode == 0 and untracked.returncode == 0:
            digest.update(b"tracked-diff\0")
            digest.update(diff.stdout)
            digest.update(b"untracked-files\0")
            for encoded_path in filter(None, untracked.stdout.split(b"\0")):
                path = root / encoded_path.decode("utf-8", errors="strict")
                digest.update(encoded_path)
                digest.update(b"\0")
                if path.is_file():
                    digest.update(path.read_bytes())
                digest.update(b"\0")
            dirty_digest = digest.hexdigest()
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "status_sha256": None if status is None else hashlib.sha256(status.encode()).hexdigest(),
        "dirty_content_sha256": dirty_digest,
    }


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _installed_direct_urls() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        raw = distribution.read_text("direct_url.json")
        if name and raw:
            try:
                records[name.lower()] = json.loads(raw)
            except json.JSONDecodeError:
                records[name.lower()] = {"invalid_direct_url_json": raw}
    return dict(sorted(records.items()))


def _torch_receipt(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda: dict[str, Any] | None = None
    if device.type == "cuda" and cuda_available:
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda = {
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_index": index,
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    return {
        "device": str(device),
        "torch": torch.__version__,
        "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "default_device": str(torch.get_default_device()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda": cuda,
    }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "build_run_receipt",
    "sha256_arrays",
    "sha256_file",
    "sha256_json",
]
