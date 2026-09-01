from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from repro.npz_compare import compare_npz


ROOT = Path(__file__).resolve().parents[1]


def _comparison_policy() -> dict[str, object]:
    manifest = json.loads(
        (ROOT / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    return dict(manifest["comparison"])


def _write_npz(path: Path, names: tuple[str, ...]) -> None:
    np.savez(
        path,
        **{
            name: np.arange(6, dtype=np.float32).reshape(2, 3)
            for name in names
        },
    )


def test_compare_npz_requires_complete_matching_layer_keys(tmp_path: Path) -> None:
    current = tmp_path / "current.npz"
    reference = tmp_path / "reference.npz"
    _write_npz(current, ("Layer_1", "Layer_2"))
    _write_npz(reference, ("Layer_1", "Layer_3"))

    with pytest.raises(ValueError, match="partial key overlap"):
        compare_npz(
            current,
            reference,
            tmp_path / "out",
            policy=_comparison_policy(),
        )


def test_compare_npz_allows_unambiguous_full_positional_mapping(tmp_path: Path) -> None:
    current = tmp_path / "current.npz"
    reference = tmp_path / "reference.npz"
    _write_npz(current, ("Layer_10", "Layer_11"))
    _write_npz(reference, ("Layer_1", "Layer_2"))

    result = compare_npz(
        current,
        reference,
        tmp_path / "out",
        policy=_comparison_policy(),
    )

    assert result["layers"] == ["Layer_10->Layer_1", "Layer_11->Layer_2"]


def test_compare_npz_rejects_dropped_or_extra_layers(tmp_path: Path) -> None:
    current = tmp_path / "current.npz"
    reference = tmp_path / "reference.npz"
    _write_npz(current, ("Layer_10", "Layer_11", "Layer_12"))
    _write_npz(reference, ("Layer_1", "Layer_2"))

    with pytest.raises(ValueError, match="complete state-layer coverage"):
        compare_npz(
            current,
            reference,
            tmp_path / "out",
            policy=_comparison_policy(),
        )


@pytest.mark.parametrize("archive", ("current", "reference"))
@pytest.mark.parametrize("nonfinite", (np.nan, np.inf, -np.inf))
def test_compare_npz_rejects_nonfinite_state_arrays(
    tmp_path: Path,
    archive: str,
    nonfinite: float,
) -> None:
    current = tmp_path / "current.npz"
    reference = tmp_path / "reference.npz"
    finite = np.arange(6, dtype=np.float32).reshape(2, 3)
    invalid = finite.copy()
    invalid[0, 0] = nonfinite
    np.savez(current, Layer_1=invalid if archive == "current" else finite)
    np.savez(reference, Layer_1=invalid if archive == "reference" else finite)
    output_dir = tmp_path / "out"

    with pytest.raises(FloatingPointError, match="finite state arrays"):
        compare_npz(
            current,
            reference,
            output_dir,
            policy=_comparison_policy(),
        )

    assert not (
        output_dir / "cross_layer_rel_l1_percentiles_node_weighted.json"
    ).exists()


def test_compare_npz_writes_strict_json_with_full_manifest_policy(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.npz"
    reference = tmp_path / "reference.npz"
    _write_npz(current, ("Layer_1", "Layer_2"))
    _write_npz(reference, ("Layer_1", "Layer_2"))
    output_dir = tmp_path / "out"
    policy = _comparison_policy()

    result = compare_npz(
        current,
        reference,
        output_dir,
        policy=policy,
    )
    summary = output_dir / "cross_layer_rel_l1_percentiles_node_weighted.json"

    def reject_nonstandard_constant(token: str) -> None:
        raise AssertionError(f"Non-standard JSON constant: {token}")

    stored = json.loads(
        summary.read_text(encoding="utf-8"),
        parse_constant=reject_nonstandard_constant,
    )
    assert stored == result
    assert stored["comparison_policy"] == policy
    assert set(stored["comparison_policy"]) == {
        "accumulation_dtype",
        "percentile_method",
        "percentiles",
        "policy_version",
        "reference_data_fingerprint",
        "reference_split_fingerprint",
        "relative_error_epsilon",
    }
    json.dumps(stored, allow_nan=False)
