from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repro.manifest import ReproJob


PERCENTILES = (50, 60, 70, 80, 90, 95, 99)


def compare_job(pack_root: Path, job: ReproJob) -> dict | None:
    reference = job.reference_path(pack_root)
    if reference is None:
        return None
    cd_npz = job.output_dir(pack_root) / "validation_states.npz"
    if not cd_npz.exists():
        raise FileNotFoundError(f"Expected validation_states.npz before comparison. Provided value: {cd_npz}")
    out_dir = job.comparison_dir(pack_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    return compare_npz(cd_npz, reference, out_dir, pack_root=pack_root)


def compare_npz(cd_npz: Path, reference_npz: Path, output_dir: Path, *, pack_root: Path | None = None) -> dict:
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
            cd = _flatten(cd_data[cd_layer])
            ref = _flatten(ref_data[ref_layer])
            if cd.shape != ref.shape:
                raise ValueError(f"Expected matching shapes for {label}. Provided value: {cd.shape} vs {ref.shape}.")
            node_count = cd.shape[1]
            total_nodes += node_count
            rel_l1 = np.mean(np.abs(cd - ref), axis=1) / (np.mean(np.abs(ref), axis=1) + 1e-12)
            per_layer[label] = {f"p{p}": float(np.percentile(rel_l1, p)) for p in PERCENTILES}
            mae = np.mean(np.abs(cd - ref), axis=1)
            ref_abs = np.mean(np.abs(ref), axis=1)
            total_mae = mae * node_count if total_mae is None else total_mae + mae * node_count
            total_ref = ref_abs * node_count if total_ref is None else total_ref + ref_abs * node_count

    node_weighted = total_mae / (total_ref + 1e-12)
    payload = {
        "cd_npz": _display_path(cd_npz, pack_root),
        "reference_npz": _display_path(reference_npz, pack_root),
        "layers": layer_labels,
        "total_nodes": int(total_nodes),
        "node_weighted_rel_l1_percentiles": {f"p{p}": float(np.percentile(node_weighted, p)) for p in PERCENTILES},
        "per_layer_rel_l1_percentiles": per_layer,
    }
    path = output_dir / "cross_layer_rel_l1_percentiles_node_weighted.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _comparison_layer_pairs(cd_files: list[str], ref_files: list[str], cd_npz: Path, reference_npz: Path) -> list[tuple[str, str]]:
    common = sorted(set(cd_files) & set(ref_files), key=_layer_sort_key)
    if common:
        return [(layer, layer) for layer in common]

    cd_layers = _state_layers(cd_files)
    ref_layers = _state_layers(ref_files)
    if len(cd_layers) == len(ref_layers):
        return list(zip(cd_layers, ref_layers))
    if len(cd_layers) > len(ref_layers):
        # SPICE/reference exports often omit the clamped input layer.  Align the
        # trailing validation layers so free/output states compare by position.
        return list(zip(cd_layers[-len(ref_layers):], ref_layers))
    raise ValueError(f"Expected comparable layer keys in NPZ files. Provided value: {cd_npz}, {reference_npz}.")


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
