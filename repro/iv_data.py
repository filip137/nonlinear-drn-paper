"""Load explicitly configured current-voltage data."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch


def load_iv_data(
    configured_path: str | Path | None,
    *,
    expected_sha256: str | None = None,
) -> torch.Tensor:
    """Return measured I-V samples as a 2xN tensor.

    File-system and NumPy concerns intentionally stay outside the vendored model
    package.  The path is deliberately explicit: environment variables must not
    be able to replace a checksummed scientific input behind the configuration's
    back.
    """

    if configured_path is None:
        raise FileNotFoundError(
            "Expected the configured experimental IV curve path to name an "
            "existing .npz file. Provided value: None."
        )
    path = Path(configured_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "Expected the configured experimental IV curve path to name an "
            f"existing .npz file. Provided value: {path}"
        )
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Expected experimental IV data to use a .npz file. "
            f"Provided value: {path}."
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Could not read configured IV data {path}: {exc}.") from exc
    if expected_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                "Expected experimental IV data SHA256 to match its configuration. "
                f"Provided value: path={path}, expected={expected_sha256}, actual={actual}."
            )

    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
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
