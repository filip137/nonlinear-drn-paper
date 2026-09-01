"""Apply an explicit, validated execution profile."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from repro.device import resolve_device


_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}
_THREAD_ENV = {
    "omp": "OMP_NUM_THREADS",
}
_THREADPOOL_LIMITERS: list[Any] = []


def apply_execution_profile(execution: Mapping[str, Any]) -> torch.device:
    """Apply every execution choice and return the configured torch device.

    The mapping is expected to have passed the v2 execution schema.  Required
    indexing below is intentional: a missing setting is an error, never a
    request for a process-dependent default.
    """

    validate_execution_relations(execution)
    backend = execution["backend"]
    determinism = execution["determinism"]
    threads = execution["threads"]

    for config_name, environment_name in _THREAD_ENV.items():
        os.environ[environment_name] = str(threads[config_name])
    os.environ["MKL_NUM_THREADS"] = str(threads["blas"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads["blas"])
    # Environment variables cover imports that happen after profile resolution;
    # threadpoolctl also constrains BLAS/OpenMP libraries that an embedding
    # application imported earlier. Keep the limiter objects alive globally.
    _THREADPOOL_LIMITERS.extend(
        [
            threadpool_limits(limits=threads["blas"], user_api="blas"),
            threadpool_limits(limits=threads["omp"], user_api="openmp"),
        ]
    )
    torch.set_num_threads(threads["torch_intraop"])
    _set_interop_threads(threads["torch_interop"])

    dtype_name = backend["default_dtype"]
    try:
        torch.set_default_dtype(_DTYPES[dtype_name])
    except KeyError as exc:  # Defensive if called without schema validation.
        raise ValueError(
            "Expected execution.backend.default_dtype to be 'float32' or "
            f"'float64'. Provided value: {dtype_name!r}."
        ) from exc

    construction_device = backend["construction_device"]
    if construction_device != "cpu":
        raise ValueError(
            "Expected execution.backend.construction_device to be 'cpu' so "
            "model initialization is independent of inherited process state."
        )
    torch.set_default_device(construction_device)

    torch.use_deterministic_algorithms(
        determinism["torch_deterministic_algorithms"],
        warn_only=determinism["warn_only"],
    )
    torch.set_float32_matmul_precision(determinism["float32_matmul_precision"])

    requested = backend["device"]
    if requested == "cpu":
        return resolve_device("cpu")
    if requested != "cuda":
        raise ValueError(
            "Expected execution.backend.device to be 'cpu' or 'cuda'. "
            f"Provided value: {requested!r}."
        )

    cuda = execution["cuda"]
    workspace_config = cuda["cublas_workspace_config"]
    inherited_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and inherited_workspace_config != workspace_config:
        raise RuntimeError(
            "Cannot apply execution.cuda.cublas_workspace_config after CUDA was "
            "initialized with a different process environment: "
            f"current={inherited_workspace_config!r}, requested={workspace_config!r}."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace_config
    torch.backends.cudnn.deterministic = cuda["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = cuda["cudnn_benchmark"]
    # Matrix multiplication is owned solely by float32_matmul_precision above;
    # the separate cuDNN convolution policy cannot override it.
    torch.backends.cudnn.allow_tf32 = cuda["cudnn_allow_tf32"]
    resolve_device("cuda")
    index = backend["index"]
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            "Expected execution.backend.index to identify an available CUDA "
            f"device. Provided index={index}, count={torch.cuda.device_count()}."
        )
    return torch.device("cuda", index)


def seed_from_config(seed: int, execution: Mapping[str, Any]) -> None:
    """Seed exactly the generators declared by a validated reference profile."""

    seeding = execution["seeding"]
    unexpected = {
        key: value
        for key, value in seeding.items()
        if value != "config_seed"
    }
    if unexpected:
        raise ValueError(
            "Expected every execution seeding policy to be 'config_seed'. "
            f"Provided value: {unexpected!r}."
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader_kwargs(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the complete loader execution policy into torch arguments."""

    loader = execution["dataloader"]
    workers = loader["num_workers"]
    persistent = loader["persistent_workers"]
    if persistent and workers == 0:
        raise ValueError(
            "Expected persistent_workers to be false when num_workers is zero."
        )
    return {
        "num_workers": workers,
        "persistent_workers": persistent,
        "pin_memory": loader["pin_memory"],
        "prefetch_factor": loader["prefetch_factor"],
        "timeout": loader["timeout_seconds"],
        "worker_init_fn": None,
        "multiprocessing_context": None,
        "pin_memory_device": "",
    }


def validate_execution_relations(execution: Mapping[str, Any]) -> None:
    """Validate cross-field execution rules before an executable snapshot is accepted."""

    loader = execution["dataloader"]
    if loader["persistent_workers"] and loader["num_workers"] == 0:
        raise ValueError(
            "Expected persistent_workers to be false when num_workers is zero."
        )
    if execution["backend"]["construction_device"] != "cpu":
        raise ValueError("Expected execution.backend.construction_device to be 'cpu'.")


def _set_interop_threads(value: int) -> None:
    try:
        torch.set_num_interop_threads(value)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts. Reusing
        # an already-applied profile is safe; attempting to change it is not.
        if torch.get_num_interop_threads() != value:
            raise RuntimeError(
                "Cannot change torch inter-op threads after parallel work has "
                f"started: current={torch.get_num_interop_threads()}, requested={value}."
            )


__all__ = [
    "apply_execution_profile",
    "dataloader_kwargs",
    "seed_from_config",
    "validate_execution_relations",
]
