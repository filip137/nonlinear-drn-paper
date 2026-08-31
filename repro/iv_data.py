"""Load measured current-voltage data at the reproducibility-app boundary."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def load_iv_data(
    configured_path: str | Path | None,
    *,
    use_environment_override: bool = True,
) -> torch.Tensor:
    """Return measured I-V samples as a 2xN tensor.

    File-system and NumPy concerns intentionally stay outside the vendored model
    package. ``LABS_IV_CURVE_PATH`` remains the highest-priority historical
    override used by the original experiment scripts. Config builders can
    disable that override when validating the exact path they will serialize.
    """

    environment_path = (
        os.environ.get("LABS_IV_CURVE_PATH") if use_environment_override else None
    )
    candidate = environment_path or configured_path
    if candidate is None:
        raise FileNotFoundError(
            "Expected experimental IV curve path via LABS_IV_CURVE_PATH or config "
            "'iv_data_path' (existing .npz file). Provided value: None."
        )
    path = Path(candidate).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "Expected experimental IV curve path via LABS_IV_CURVE_PATH or config "
            f"'iv_data_path' (existing .npz file). Provided value: {path}"
        )
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Expected experimental IV data to use a .npz file. "
            f"Provided value: {path}."
        )

    with np.load(path, allow_pickle=False) as data:
        if "iv" in data:
            iv = np.asarray(data["iv"])
            if iv.ndim != 2 or iv.shape[0] != 2:
                raise ValueError(
                    "Expected experimental IV array 'iv' to have shape (2, N) "
                    f"in current-then-voltage order. Provided shape: {iv.shape!r} "
                    f"in {path}."
                )
        elif "i" in data and "v" in data:
            current = np.asarray(data["i"])
            voltage = np.asarray(data["v"])
            if current.ndim != 1 or voltage.ndim != 1:
                raise ValueError(
                    "Expected experimental IV arrays 'i' and 'v' to be "
                    f"one-dimensional. Provided shapes: i={current.shape!r}, "
                    f"v={voltage.shape!r} in {path}."
                )
            if current.shape[0] != voltage.shape[0]:
                raise ValueError(
                    "Expected experimental IV arrays 'i' and 'v' to have equal "
                    f"lengths. Provided lengths: i={current.shape[0]}, "
                    f"v={voltage.shape[0]} in {path}."
                )
            iv = np.stack([current, voltage], axis=0)
        else:
            raise ValueError(
                "Expected experimental IV data to contain 'iv' or both 'i' and 'v' "
                f"arrays. Provided file: {path}."
            )

    if not np.issubdtype(iv.dtype, np.number) or np.issubdtype(
        iv.dtype, np.complexfloating
    ):
        raise ValueError(
            "Expected experimental IV samples to be real numeric values. "
            f"Provided dtype: {iv.dtype} in {path}."
        )
    if iv.shape[1] < 2:
        raise ValueError(
            "Expected experimental IV data to contain at least two samples. "
            f"Provided sample count: {iv.shape[1]} in {path}."
        )
    if not np.isfinite(iv).all():
        raise ValueError(
            "Expected experimental IV samples to be finite. "
            f"Provided file: {path}."
        )

    current = iv[0]
    voltage = iv[1]
    if not np.all(np.diff(voltage) > 0):
        raise ValueError(
            "Expected experimental IV voltages to be strictly increasing. "
            f"Provided file: {path}."
        )
    if not np.all(np.diff(current) >= 0):
        raise ValueError(
            "Expected experimental IV currents to be nondecreasing. "
            f"Provided file: {path}."
        )
    return torch.as_tensor(iv)
