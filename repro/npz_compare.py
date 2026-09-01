from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from repro.manifest import PackManifest, ReproJob
from repro.provenance import sha256_file, sha256_json
from repro.strict_config import pretty_json_text


def compare_job(
    pack_root: Path, manifest: PackManifest, job: ReproJob
) -> dict | None:
    reference = job.reference_path(pack_root)
    if reference is None:
        return None
    cd_npz = job.output_dir(pack_root) / "validation_states.npz"
    if not cd_npz.exists():
        raise FileNotFoundError(f"Expected validation_states.npz before comparison. Provided value: {cd_npz}")
    validation = _validated_run_identity(pack_root, manifest, job, cd_npz)
    out_dir = job.comparison_dir(pack_root)
    return compare_npz(
        cd_npz,
        reference,
        out_dir,
        policy=manifest.comparison,
        pack_root=pack_root,
        provenance=validation,
    )


def compare_npz(
    cd_npz: Path,
    reference_npz: Path,
    output_dir: Path,
    *,
    policy: Mapping[str, object],
    pack_root: Path | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict:
    percentiles = tuple(policy["percentiles"])
    epsilon = float(policy["relative_error_epsilon"])
    percentile_method = str(policy["percentile_method"])
    if policy["accumulation_dtype"] != "float64":
        raise ValueError("Expected comparison accumulation_dtype to be 'float64'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cd_npz, allow_pickle=False) as cd_data, np.load(reference_npz, allow_pickle=False) as ref_data:
        pairs = _comparison_layer_pairs(cd_data.files, ref_data.files, cd_npz, reference_npz)
        total_nodes = 0
        total_mae = None
        total_ref = None
        per_layer = {}
        layer_labels = []
        for cd_layer, ref_layer in pairs:
            label = cd_layer if cd_layer == ref_layer else f"{cd_layer}->{ref_layer}"
            layer_labels.append(label)
            cd = _flatten(cd_data[cd_layer]).astype(np.float64, copy=False)
            ref = _flatten(ref_data[ref_layer]).astype(np.float64, copy=False)
            if cd.shape != ref.shape:
                raise ValueError(f"Expected matching shapes for {label}. Provided value: {cd.shape} vs {ref.shape}.")
            if not np.isfinite(cd).all() or not np.isfinite(ref).all():
                raise FloatingPointError(
                    f"Expected finite state arrays for comparison layer {label}."
                )
            node_count = cd.shape[1]
            total_nodes += node_count
            rel_l1 = np.mean(np.abs(cd - ref), axis=1, dtype=np.float64) / (
                np.mean(np.abs(ref), axis=1, dtype=np.float64) + epsilon
            )
            per_layer[label] = {
                f"p{p}": float(np.percentile(rel_l1, p, method=percentile_method))
                for p in percentiles
            }
            mae = np.mean(np.abs(cd - ref), axis=1, dtype=np.float64)
            ref_abs = np.mean(np.abs(ref), axis=1, dtype=np.float64)
            total_mae = mae * node_count if total_mae is None else total_mae + mae * node_count
            total_ref = ref_abs * node_count if total_ref is None else total_ref + ref_abs * node_count

    node_weighted = total_mae / (total_ref + epsilon)
    if not np.isfinite(node_weighted).all():
        raise FloatingPointError("Expected finite node-weighted comparison values.")
    payload = {
        "cd_npz": _display_path(cd_npz, pack_root),
        "reference_npz": _display_path(reference_npz, pack_root),
        "layers": layer_labels,
        "total_nodes": int(total_nodes),
        "comparison_policy": dict(policy),
        "node_weighted_rel_l1_percentiles": {
            f"p{p}": float(np.percentile(node_weighted, p, method=percentile_method))
            for p in percentiles
        },
        "per_layer_rel_l1_percentiles": per_layer,
    }
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    path = output_dir / "cross_layer_rel_l1_percentiles_node_weighted.json"
    path.write_text(pretty_json_text(payload), encoding="utf-8")
    return payload


def load_comparison_summary(
    pack_root: Path, manifest: PackManifest, job: ReproJob
) -> dict:
    path = job.comparison_dir(pack_root) / "cross_layer_rel_l1_percentiles_node_weighted.json"
    if not path.is_file():
        raise FileNotFoundError(f"Expected comparison summary for {job.job_id!r}: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _validated_run_identity(
        pack_root,
        manifest,
        job,
        job.output_dir(pack_root) / "validation_states.npz",
    )
    if payload.get("provenance") != expected:
        raise ValueError(
            f"Comparison summary provenance is stale for job {job.job_id!r}."
        )
    if payload.get("comparison_policy") != manifest.comparison:
        raise ValueError(
            f"Comparison summary policy is stale for job {job.job_id!r}."
        )
    return payload


def _validated_run_identity(
    pack_root: Path,
    manifest: PackManifest,
    job: ReproJob,
    states_path: Path,
) -> dict[str, object]:
    run_dir = job.output_dir(pack_root)
    metadata_path = run_dir / "validation_metadata.json"
    receipt_path = run_dir / "run_receipt.json"
    if not metadata_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(
            f"Expected v2 validation metadata and receipt before comparing {job.job_id!r}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_config = sha256_json(manifest.resolved_job_config(pack_root, job).document)
    actual_states = sha256_file(states_path)
    checks = {
        "metadata job_id": (metadata.get("job_id"), job.job_id),
        "receipt job_id": (receipt.get("run", {}).get("job_id"), job.job_id),
        "metadata config": (metadata.get("resolved_config_sha256"), expected_config),
        "receipt config": (receipt.get("resolved_config_sha256"), expected_config),
        "validation states": (metadata.get("validation_states_sha256"), actual_states),
        "data fingerprint": (
            receipt.get("data_fingerprint"),
            manifest.comparison["reference_data_fingerprint"],
        ),
        "split fingerprint": (
            receipt.get("split_fingerprint"),
            manifest.comparison["reference_split_fingerprint"],
        ),
        "weights": (
            receipt.get("assets", {}).get("weights", {}).get("sha256"),
            job.assets["weights"].sha256,
        ),
    }
    if job.reference_npz is not None:
        checks["reference states"] = (
            receipt.get("assets", {}).get("reference_states", {}).get("sha256"),
            job.assets["reference_states"].sha256,
        )
    failures = [name for name, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise ValueError(
            f"Validation artifacts are stale or inconsistent for {job.job_id!r}: "
            + ", ".join(failures)
            + "."
        )
    return {
        "job_id": job.job_id,
        "resolved_config_sha256": expected_config,
        "validation_states_sha256": actual_states,
        "reference_states_sha256": job.assets["reference_states"].sha256,
        "data_fingerprint": receipt["data_fingerprint"],
        "split_fingerprint": receipt["split_fingerprint"],
    }


def _comparison_layer_pairs(cd_files: list[str], ref_files: list[str], cd_npz: Path, reference_npz: Path) -> list[tuple[str, str]]:
    cd_layers = _state_layers(cd_files)
    ref_layers = _state_layers(ref_files)
    common = set(cd_layers) & set(ref_layers)
    if set(cd_layers) == set(ref_layers) and cd_layers:
        return [(layer, layer) for layer in cd_layers]
    if common:
        raise ValueError(
            "Expected NPZ state-layer keys to match completely; partial key "
            f"overlap would compare an ambiguous subset. Provided value: "
            f"cd={cd_layers!r}, reference={ref_layers!r}."
        )
    if len(cd_layers) == len(ref_layers):
        return list(zip(cd_layers, ref_layers))
    raise ValueError(
        "Expected complete state-layer coverage in both NPZ files. "
        f"Provided value: {cd_npz} has {cd_layers!r}; "
        f"{reference_npz} has {ref_layers!r}."
    )


def _state_layers(names: list[str]) -> list[str]:
    return sorted(
        [
            name for name in names
            if name.startswith("Layer_") and "Node_Order" not in name
        ],
        key=_layer_sort_key,
    )


def _flatten(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def _layer_sort_key(name: str):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (0, int(digits)) if digits else (1, name)


def _display_path(path: Path, pack_root: Path | None) -> str:
    if pack_root is None:
        return str(path)
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)
