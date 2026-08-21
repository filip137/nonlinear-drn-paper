"""Load measured current-voltage data at the reproducibility-app boundary."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def load_iv_data(configured_path: str | Path | None) -> torch.Tensor:
    """Return measured I-V samples as a 2xN tensor.

    File-system and NumPy concerns intentionally stay outside the vendored model
    package. ``LABS_IV_CURVE_PATH`` remains the highest-priority historical
    override used by the original experiment scripts.
    """

    candidate = os.environ.get("LABS_IV_CURVE_PATH") or configured_path
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

    with np.load(path) as data:
        if "iv" in data:
            iv = data["iv"]
        elif "i" in data and "v" in data:
            iv = np.stack([data["i"], data["v"]], axis=0)
        else:
            raise ValueError(
                "Expected experimental IV data to contain 'iv' or both 'i' and 'v' "
                f"arrays. Provided file: {path}."
            )
    return torch.as_tensor(iv)
