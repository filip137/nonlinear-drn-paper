"""Deterministic validation of bundled Digits checkpoints."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network
from model.resistive.network import DeepResistiveEnergy
from repro.config import RuntimeConfig, parse_layer_shapes
from repro.execution import (
    apply_execution_profile,
    dataloader_kwargs,
    seed_from_config,
)
from repro.manifest import PackManifest, ReproJob
from repro.minimizer_factory import build_minimizer, simulation_assets
from repro.provenance import build_run_receipt, sha256_arrays, sha256_file
from repro.strict_config import pretty_json_text, validate_document


@dataclass(frozen=True)
class ValidationResult:
    run_dir: Path
    states_npz: Path
    metadata_json: Path
    receipt_json: Path
    accuracy: float


@dataclass(frozen=True)
class DemoResult:
    job_id: str
    config: str
    weights: str
    architecture: tuple[int, ...]
    non_linearity: str
    num_iterations: int
    batch_size: int
    execution_profile: str
    device: str
    correct: int
    total: int
    accuracy: float
    receipt: dict[str, Any]


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def run_validation(
    pack_root: Path,
    manifest: PackManifest,
    job: ReproJob,
) -> ValidationResult:
    """Run one manifest job using its resolved config and execution profile."""

    cfg = manifest.resolved_job_config(pack_root, job)
    execution = cfg.document["execution"]
    device = apply_execution_profile(execution)
    seed_from_config(cfg.seed, execution)

    run_dir = job.output_dir(pack_root)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Expected a new or empty validation output directory so stale "
            f"artifacts cannot be reused. Provided value: {run_dir}."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _, test_loader, data_fingerprint, split_fingerprint = _digits_loaders(
        cfg.data,
        execution,
        device,
    )
    weights_path = job.weights_path(pack_root)
    energy_fn, network, free_layers, cost_fn, _, layer_shapes = _build_energy_stack(
        cfg=cfg,
        weights_path=weights_path,
        device=device,
    )
    minimizer = build_minimizer(
        fn=energy_fn,
        free_layers=free_layers,
        simulation=cfg.simulation,
        equilibrium=cfg.equilibrium,
        repo_root=pack_root,
    )

    inputs_batches = []
    labels_batches = []
    indices_batches = []
    # Stable per-network depth keys match the reference export. Global vendor
    # Layer counters depend on process history and must never enter artifacts.
    state_layers = list(network.free_layers())
    states = {
        f"Layer_{depth}": []
        for depth, _layer in enumerate(state_layers, start=1)
    }
    total = 0
    correct = 0
    for inputs, labels, indices in test_loader:
        network.set_input(inputs, reset=True)
        minimizer.compute_equilibrium()
        _validate_equilibrium_result(minimizer, free_layers, cfg.equilibrium)
        inputs_batches.append(inputs.detach().cpu())
        labels_batches.append(labels.detach().cpu())
        indices_batches.append(indices.detach().cpu())
        for depth, layer in enumerate(state_layers, start=1):
            states[f"Layer_{depth}"].append(layer.state.detach().cpu())
        cost_fn.set_target(labels)
        errors = cost_fn.error_fn()
        total += int(errors.numel())
        correct += int(errors.numel()) - int(errors.sum().item())

    inputs_npz = run_dir / "validation_inputs.npz"
    states_npz = run_dir / "validation_states.npz"
    np.savez(
        inputs_npz,
        inputs=torch.cat(inputs_batches, dim=0).numpy(),
        labels=torch.cat(labels_batches, dim=0).numpy(),
        indices=torch.cat(indices_batches, dim=0).numpy(),
    )
    np.savez(
        states_npz,
        **{name: torch.cat(values, dim=0).numpy() for name, values in states.items()},
    )
    accuracy = correct / total if total else 0.0

    resolved_path = run_dir / "config.resolved.json"
    resolved_path.write_text(pretty_json_text(cfg.document), encoding="utf-8")
    assets = {
        "weights": weights_path,
        **simulation_assets(cfg.simulation, repo_root=pack_root),
    }
    reference = job.reference_path(pack_root)
    if reference is not None:
        assets["reference_states"] = reference
    receipt = build_run_receipt(
        repo_root=pack_root,
        resolved_config=cfg.document,
        execution=execution,
        device=device,
        assets=assets,
        source_documents=[
            {"owner": "replay", **job.base_config.as_dict()},
            {"owner": "execution", **manifest.execution_ref.as_dict()},
        ],
        data_fingerprint=data_fingerprint,
        split_fingerprint=split_fingerprint,
        extra={
            "job_id": job.job_id,
            "validation_correct": correct,
            "validation_total": total,
            "validation_accuracy": accuracy,
        },
    )
    receipt_json = run_dir / "run_receipt.json"
    receipt_json.write_text(pretty_json_text(receipt), encoding="utf-8")
    metadata = {
        "job_id": job.job_id,
        "group": job.group,
        "config": job.config,
        "weights": job.weights,
        "reference_npz": job.reference_npz,
        "dims": cfg.dims,
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "nonlinearity": cfg.non_linearity,
        "device": str(device),
        "batch_size": cfg.batch_size,
        "equilibrium": cfg.equilibrium,
        "validation_accuracy": accuracy,
        "validation_correct": correct,
        "validation_total": total,
        "validation_inputs": _pack_rel(inputs_npz, pack_root),
        "validation_inputs_sha256": sha256_file(inputs_npz),
        "validation_states": _pack_rel(states_npz, pack_root),
        "validation_states_sha256": sha256_file(states_npz),
        "resolved_config_sha256": receipt["resolved_config_sha256"],
    }
    metadata_json = run_dir / "validation_metadata.json"
    metadata_json.write_text(pretty_json_text(metadata), encoding="utf-8")
    return ValidationResult(
        run_dir=run_dir,
        states_npz=states_npz,
        metadata_json=metadata_json,
        receipt_json=receipt_json,
        accuracy=accuracy,
    )


def run_demo(pack_root: Path, manifest: PackManifest) -> DemoResult:
    """Evaluate the manifest-selected checkpoint without writing artifacts."""

    job = manifest.demo_job()
    cfg = manifest.resolved_job_config(pack_root, job)
    document = copy.deepcopy(cfg.document)
    document["data"]["loader"]["batch_size"] = manifest.demo["batch_size"]
    validate_document(document, "replay-v2.schema.json", repo_root=pack_root)
    cfg = RuntimeConfig(document=document)

    execution = cfg.document["execution"]
    device = apply_execution_profile(execution)
    seed_from_config(cfg.seed, execution)
    _, test_loader, data_fingerprint, split_fingerprint = _digits_loaders(
        cfg.data,
        execution,
        device,
    )
    weights_path = job.weights_path(pack_root)
    energy_fn, network, free_layers, cost_fn, _, _ = _build_energy_stack(
        cfg=cfg,
        weights_path=weights_path,
        device=device,
    )
    minimizer = build_minimizer(
        fn=energy_fn,
        free_layers=free_layers,
        simulation=cfg.simulation,
        equilibrium=cfg.equilibrium,
        repo_root=pack_root,
    )

    total = 0
    correct = 0
    with torch.inference_mode():
        for inputs, labels, _ in test_loader:
            network.set_input(inputs, reset=True)
            minimizer.compute_equilibrium()
            _validate_equilibrium_result(minimizer, free_layers, cfg.equilibrium)
            cost_fn.set_target(labels)
            errors = cost_fn.error_fn()
            total += int(errors.numel())
            correct += int(errors.numel()) - int(errors.sum().item())
    accuracy = correct / total if total else 0.0
    receipt = build_run_receipt(
        repo_root=pack_root,
        resolved_config=cfg.document,
        execution=execution,
        device=device,
        assets={
            "weights": weights_path,
            **simulation_assets(cfg.simulation, repo_root=pack_root),
        },
        source_documents=[
            {"owner": "replay", **job.base_config.as_dict()},
            {"owner": "execution", **manifest.execution_ref.as_dict()},
        ],
        data_fingerprint=data_fingerprint,
        split_fingerprint=split_fingerprint,
        extra={
            "job_id": job.job_id,
            "demo_batch_size": cfg.batch_size,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        },
    )
    return DemoResult(
        job_id=job.job_id,
        config=job.config,
        weights=job.weights,
        architecture=tuple(cfg.dims),
        non_linearity=cfg.non_linearity,
        num_iterations=cfg.num_iterations,
        batch_size=cfg.batch_size,
        execution_profile=execution["name"],
        device=str(device),
        correct=correct,
        total=total,
        accuracy=accuracy,
        receipt=receipt,
    )


def _pack_rel(path: Path, pack_root: Path) -> str:
    try:
        return path.relative_to(pack_root).as_posix()
    except ValueError:
        return str(path)


def _digits_loaders(
    data: dict[str, Any],
    execution: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, str, str]:
    if data["source"] != {
        "name": "digits",
        "loader": "sklearn.datasets.load_digits",
    }:
        raise ValueError(
            "Expected replay data.source to identify sklearn Digits exactly. "
            f"Provided value: {data['source']!r}."
        )
    values, targets = load_digits(
        n_class=10,
        return_X_y=True,
        as_frame=False,
    )
    values = np.asarray(values)
    targets = np.asarray(targets)
    original_indices = np.arange(values.shape[0], dtype=np.int64)
    data_fingerprint = sha256_arrays(values, targets)

    subset = data["subset"]
    if subset["method"] != "random_max_points":
        raise ValueError(
            "Expected replay data.subset.method to be 'random_max_points'. "
            f"Provided value: {subset['method']!r}."
        )
    if subset["max_points"] < values.shape[0]:
        selected = np.random.RandomState(data["seed"]).permutation(values.shape[0])[
            : subset["max_points"]
        ]
        values = values[selected]
        targets = targets[selected]
        original_indices = original_indices[selected]

    preprocessing = data["preprocessing"]
    if (
        preprocessing["input_dtype"] != "float32"
        or preprocessing["target_dtype"] != "int64"
    ):
        raise ValueError(
            "Expected replay preprocessing dtypes to be float32 inputs and int64 "
            f"targets. Provided value: {preprocessing!r}."
        )
    inputs = (
        torch.tensor(values, dtype=torch.float32, device=device)
        * preprocessing["input_scale"]
        + preprocessing["input_offset"]
    )
    labels = torch.tensor(targets, dtype=torch.long, device=device)
    dataset = TensorDataset(inputs, labels)
    split = data["split"]
    if split["method"] != "random_fraction":
        raise ValueError(
            "Expected replay data.split.method to be 'random_fraction'. "
            f"Provided value: {split['method']!r}."
        )
    if split["rounding"] != "floor":
        raise ValueError("Expected replay data.split.rounding to be 'floor'.")
    train_size = int(split["train_fraction"] * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(data["seed"])
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )
    split_fingerprint = sha256_arrays(
        original_indices,
        np.asarray(train_dataset.indices, dtype=np.int64),
        np.asarray(test_dataset.indices, dtype=np.int64),
    )
    loader_policy = data["loader"]
    common = dataloader_kwargs(execution)
    return (
        DataLoader(
            train_dataset,
            batch_size=loader_policy["batch_size"],
            shuffle=loader_policy["shuffle"],
            drop_last=loader_policy["drop_last"],
            **common,
        ),
        DataLoader(
            IndexedDataset(test_dataset),
            batch_size=loader_policy["batch_size"],
            shuffle=loader_policy["shuffle"],
            drop_last=loader_policy["drop_last"],
            **common,
        ),
        data_fingerprint,
        split_fingerprint,
    )


def _build_energy_stack(
    *,
    cfg: RuntimeConfig,
    weights_path: Path,
    device: torch.device,
):
    _, _, _, layer_shapes = parse_layer_shapes(cfg.dims)
    model = cfg.model
    if model["state_dtype"] != cfg.document["execution"]["backend"]["default_dtype"]:
        raise ValueError(
            "Expected model.state_dtype to match execution.backend.default_dtype."
        )
    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=cfg.weight_gains,
        input_gain=cfg.input_gain,
        non_linearity=cfg.non_linearity,
        exponential_diode_param=cfg.exponential_diode_param,
        quadratic_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=cfg.voltage_amp,
        current_amp=cfg.current_amp,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        weight_init_mode=model["weight_initialization"],
        bias_scale_mode=cfg.bias_scale_mode,
        bias_interaction_type=cfg.bias_interaction_type,
        bias_enabled=model["bias"]["enabled"],
        bias_initial_value=model["bias"]["initialization"]["value"],
        bias_minimum=model["bias"]["bounds"]["minimum"],
        bias_maximum=model["bias"]["bounds"]["maximum"],
        signed_weights=cfg.signed_weights,
        conv_pipeline=[],
        learn_input_gain=model["amplification_learning"]["input_gain"],
        learn_voltage_amp=model["amplification_learning"]["voltage_factor"],
        learn_current_amp=model["amplification_learning"]["current_factor"],
    )
    energy_fn.set_device(device)
    energy_fn.load(weights_path)
    network = Network(energy_fn, input_mode=_input_mode(model))
    output_layer = energy_fn.layers()[-1]
    output = model["output"]
    if output["encoding"] == "single_ended":
        if output_layer.shape[0] != output["classes"]:
            raise ValueError("Configured single-ended output width does not match the model.")
        cost_fn = SquaredError(output_layer)
    elif output["encoding"] == "differential_pair":
        if output_layer.shape[0] != 2 * output["classes"]:
            raise ValueError("Configured differential output width does not match the model.")
        cost_fn = SquaredErrorPairedOutputs(output_layer, output["classes"])
    else:  # Defensive when called without schema validation.
        raise ValueError(f"Unsupported output encoding: {output['encoding']!r}.")
    return (
        energy_fn,
        network,
        network.free_layers(),
        cost_fn,
        output_layer,
        layer_shapes,
    )


def _input_mode(model: dict[str, Any]) -> str:
    if model["input_encoding"] != "signed_pair":
        raise ValueError(
            "Expected replay model.input_encoding to be 'signed_pair'. "
            f"Provided value: {model['input_encoding']!r}."
        )
    return "train"


def _validate_equilibrium_result(minimizer, free_layers, equilibrium) -> None:
    for depth, layer in enumerate(free_layers, start=1):
        if not torch.isfinite(layer.state).all():
            raise FloatingPointError(
                f"Expected finite replay equilibrium state at free layer {depth}."
            )
    if equilibrium["method"] != "voltage_change":
        return
    allowed = (
        equilibrium["relative_tolerance"] * minimizer.final_reference_scale
        + equilibrium["absolute_tolerance"]
    )
    if minimizer.final_max_voltage_change > allowed:
        raise RuntimeError(
            "Replay equilibrium exhausted max_sweeps without satisfying its "
            "voltage-change tolerance: "
            f"delta={minimizer.final_max_voltage_change}, allowed={allowed}."
        )


__all__ = ["DemoResult", "ValidationResult", "run_demo", "run_validation"]
