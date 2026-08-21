"""Device selection and actionable CUDA environment diagnostics."""

from __future__ import annotations

import sys
from typing import Any

import torch


def resolve_device(value: str) -> torch.device:
    """Resolve a requested device without silently changing CUDA runs to CPU."""

    if value not in {"cpu", "cuda"}:
        raise ValueError(
            "Expected device to be 'cpu' or 'cuda'. "
            f"Provided value: {value!r}."
        )
    if value == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")

    torch_version = str(torch.__version__)
    cuda_runtime = getattr(torch.version, "cuda", None)
    if cuda_runtime is None:
        cause = "The installed PyTorch build is CPU-only."
    else:
        cause = (
            "The CUDA-enabled PyTorch build could not initialize; possible causes "
            "include a bundled CUDA runtime newer than the NVIDIA driver or a GPU "
            "hidden from the current process."
        )
    raise RuntimeError(
        "Expected CUDA to be available for --device cuda. "
        "Provided environment: "
        f"python={sys.executable!r}, torch={torch_version!r}, "
        f"torch.version.cuda={cuda_runtime!r}, "
        "torch.cuda.is_available()=False. "
        f"{cause} Activate this repository's virtual environment with "
        "`source .venv/bin/activate`, install the pinned compatible stack with "
        "`python -m pip install -r requirements-cuda.txt`, then run "
        "`python scripts/reproduce.py verify --device cuda`. "
        "The requested run was not moved to CPU."
    )


def cuda_summary() -> dict[str, Any]:
    """Return the initialized CUDA runtime and first-device description."""

    device = resolve_device("cuda")
    return {
        "torch": str(torch.__version__),
        "runtime": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(device),
    }
