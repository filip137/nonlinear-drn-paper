"""Strict, deterministic equilibrium-propagation training."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network
from model.resistive.network import DeepResistiveEnergy
from repro.execution import (
    apply_execution_profile,
    dataloader_kwargs,
    seed_from_config,
    validate_execution_relations,
)
from repro.minimizer_factory import build_minimizer, simulation_assets
from repro.provenance import build_run_receipt, sha256_arrays, sha256_file
from repro.strict_config import (
    CompositionResult,
    ConfigurationOverrideError,
    JsonPointerOverride,
    apply_json_pointer_overrides,
    pretty_json_text,
    resolve_config_source,
    validate_document,
)
from training.sgd import AugmentedFunction, EquilibriumProp


_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class TrainingConfig:
    """Typed view over a fully expanded, schema-validated training document."""

    document: dict[str, Any]

    @property
    def description(self) -> str:
        return self.document["description"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.document["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.document["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.document["training"]

    @property
    def simulation(self) -> dict[str, Any]:
        return self.document["simulation"]

    @property
    def equilibrium(self) -> dict[str, Any]:
        return self.document["equilibrium"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.document["execution"]

    @property
    def layer_shapes(self) -> list[tuple[int, ...]]:
        return [tuple(shape) for shape in self.model["layer_shapes"]]

    @property
    def non_linearity(self) -> str:
        return self.simulation["nonlinearity"]

    @property
    def weight_gains(self) -> list[float]:
        return list(self.model["weight_gains"])

    @property
    def weight_min(self) -> float:
        return self.model["weight_bounds"]["minimum"]

    @property
    def weight_max(self) -> float:
        return self.model["weight_bounds"]["maximum"]

    @property
    def weight_init_mode(self) -> str:
        return self.model["weight_initialization"]

    @property
    def input_gain(self) -> float:
        return self.model["input_gain"]

    @property
    def voltage_amp(self) -> float:
        return self.simulation["amplification"]["voltage_factor"]

    @property
    def current_amp(self) -> float:
        return self.simulation["amplification"]["current_factor"]

    @property
    def exponential_diode_param(self) -> dict[str, float]:
        if self.non_linearity == "experimental":
            return {}
        physical = self.simulation["physical"]
        return {
            "I_s": physical["saturation_current"],
            "V_t": physical["thermal_voltage"],
            "V_off": physical["offset_voltage"],
        }

    @property
    def batch_size(self) -> int:
        return self.training["loader"]["batch_size"]

    @property
    def num_epochs(self) -> int:
        return self.training["epochs"]

    @property
    def num_iterations(self) -> int:
        return self.equilibrium["sweeps"]

    @property
    def learning_rates(self) -> list[float]:
        return list(self.training["optimizer"]["learning_rates"])

    @property
    def scheduler_gamma(self) -> float:
        return self.training["scheduler"]["gamma"]

    @property
    def nudging(self) -> float:
        return self.training["equilibrium_propagation"]["nudging"]

    @property
    def ep_variant(self) -> str:
        return self.training["equilibrium_propagation"]["variant"]

    @property
    def seed(self) -> int:
        return self.dataset["seed"]


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    final_checkpoint: Path | None
    best_checkpoint: Path
    history_path: Path
    receipt_path: Path
    history: dict[str, list[float]]


class _LiveProgress:
    """Render running metrics in place on terminals and periodically in logs."""

    def __init__(
        self,
        phase: str,
        epoch: int,
        num_epochs: int,
        total_batches: int,
        *,
        stream: TextIO | None = None,
        interactive: bool | None = None,
    ) -> None:
        self.phase = phase
        self.epoch = int(epoch)
        self.num_epochs = int(num_epochs)
        self.total_batches = int(total_batches)
        self.stream = sys.stdout if stream is None else stream
        self.interactive = (
            bool(self.stream.isatty()) if interactive is None else bool(interactive)
        )
        self.log_interval = max(1, math.ceil(max(self.total_batches, 1) / 10))
        self.started_at = time.perf_counter()
        self.last_width = 0
        self.started = False

    def start(self) -> None:
        self.started_at = time.perf_counter()
        self.started = True
        self._write(
            f"{self.phase} epoch={self.epoch}/{self.num_epochs} "
            f"batches={self.total_batches} starting"
        )

    def update(
        self, batch: int, *, running_loss: float, running_accuracy: float
    ) -> None:
        batch = int(batch)
        if (
            not self.interactive
            and batch != 1
            and batch != self.total_batches
            and batch % self.log_interval != 0
        ):
            return
        elapsed = time.perf_counter() - self.started_at
        self._write(
            f"{self.phase} epoch={self.epoch}/{self.num_epochs} "
            f"batch={batch}/{self.total_batches} loss={running_loss:.6g} "
            f"accuracy={running_accuracy:.2%} elapsed={elapsed:.1f}s"
        )

    def close(self) -> None:
        if self.interactive and self.started:
            print(file=self.stream, flush=True)
        self.started = False

    def _write(self, message: str) -> None:
        if self.interactive:
            rendered = message.ljust(self.last_width)
            print(f"\r{rendered}", end="", file=self.stream, flush=True)
            self.last_width = max(self.last_width, len(message))
        else:
            print(message, file=self.stream, flush=True)


def load_training_config(
    path: Path | Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[TrainingConfig, dict[str, Any]]:
    """Resolve a v2 source or reload an already-expanded immutable snapshot."""

    source_record = _capture_source_record(path, repo_root=repo_root)
    result = resolve_config_source(
        path,
        schema="training-v2.schema.json",
        reference_fields={
            "simulation_ref": "simulation",
            "execution_ref": "execution",
        },
        reference_schemas={
            "simulation": "simulator-v2.schema.json",
            "execution": "execution-v2.schema.json",
        },
        repo_root=repo_root,
    )
    _verify_and_stamp_source(result, source_record, repo_root=repo_root)
    config = TrainingConfig(document=result.document)
    _validate_relations(config)
    simulation_assets(config.simulation, repo_root=repo_root)
    return config, result.document


def resolve_training_config(
    path: Path | Mapping[str, Any],
    *,
    repo_root: Path,
    overrides: Sequence[str] = (),
) -> CompositionResult:
    """Resolve references, apply explicit overrides, and validate the snapshot."""

    source_record = _capture_source_record(path, repo_root=repo_root)
    result = resolve_config_source(
        path,
        schema="training-v2.schema.json",
        reference_fields={
            "simulation_ref": "simulation",
            "execution_ref": "execution",
        },
        reference_schemas={
            "simulation": "simulator-v2.schema.json",
            "execution": "execution-v2.schema.json",
        },
        repo_root=repo_root,
    )
    _verify_and_stamp_source(result, source_record, repo_root=repo_root)
    if not overrides:
        config = TrainingConfig(result.document)
        _validate_relations(config)
        simulation_assets(config.simulation, repo_root=repo_root)
        return result

    parsed = [_parse_override(value) for value in overrides]
    permitted_owners = {
        "data",
        "model",
        "training",
        "equilibrium",
        "simulation",
        "execution",
    }
    forbidden = [
        entry.pointer
        for entry in parsed
        if not entry.pointer.startswith("/")
        or entry.pointer.split("/", maxsplit=2)[1] not in permitted_owners
    ]
    if forbidden:
        raise ConfigurationOverrideError(
            "Overrides may replace existing values only within data, model, "
            "training, equilibrium, simulation, or execution; generated identity "
            f"and provenance fields are immutable. Provided pointers: {forbidden}."
        )
    document = apply_json_pointer_overrides(result.document, parsed)
    provenance = document["provenance"]
    if "generation_overrides" in provenance:
        raise ConfigurationOverrideError(
            "An expanded snapshot already records generation_overrides; create a "
            "new source config instead of stacking untraceable edits."
        )
    provenance["generation_overrides"] = [
        {"pointer": entry.pointer, "value": entry.value} for entry in parsed
    ]
    validate_document(document, "training-v2.schema.json", repo_root=repo_root)
    config = TrainingConfig(document)
    _validate_relations(config)
    simulation_assets(config.simulation, repo_root=repo_root)
    return CompositionResult(document=document, references=result.references)


def run_training(
    repo_root: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
    overrides: Sequence[str] = (),
    download: bool = False,
) -> TrainingResult:
    """Resolve then run one training config; numerical kwargs are not accepted."""

    root = repo_root.expanduser().resolve()
    source_path = config_path.expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    source_sha256 = sha256_file(source_path)
    resolved = resolve_training_config(
        source_path,
        repo_root=root,
        overrides=overrides,
    )
    cfg = TrainingConfig(resolved.document)
    if sha256_file(source_path) != source_sha256:
        raise RuntimeError(
            "Training source changed after resolution; refusing to run with "
            "ambiguous source provenance."
        )

    if output_dir is None:
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        output_dir = root / "outputs" / "training" / f"{source_path.stem}_{stamp}"
    elif not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Expected a new or empty training output directory so stale artifacts "
            f"cannot be mixed with this run. Provided value: {output_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # The exact executable snapshot exists before importing data or constructing
    # the numerical model. A failed run can therefore be replayed unambiguously.
    resolved_path = output_dir / "config.resolved.json"
    resolved_path.write_text(pretty_json_text(cfg.document), encoding="utf-8")

    device = apply_execution_profile(cfg.execution)
    seed_from_config(cfg.seed, cfg.execution)
    (
        train_loader,
        evaluation_loader,
        data_fingerprint,
        split_fingerprint,
    ) = _build_loaders(cfg, repo_root=root, download=download)
    energy_fn = _build_energy(cfg, device)
    network = Network(energy_fn, input_mode=_input_mode(cfg.model))
    free_layers = network.free_layers()
    cost_fn = _build_cost(cfg, energy_fn.layers()[-1])
    ep = cfg.training["equilibrium_propagation"]
    augmented_fn = AugmentedFunction(
        energy_fn,
        cost_fn,
        nudging_mode=ep["nudging_mode"],
        current_scale=None,
    )
    inference_minimizer = build_minimizer(
        fn=energy_fn,
        free_layers=free_layers,
        simulation=cfg.simulation,
        equilibrium=cfg.equilibrium,
        repo_root=root,
    )
    training_minimizer = build_minimizer(
        fn=augmented_fn,
        free_layers=free_layers,
        simulation=cfg.simulation,
        equilibrium=cfg.equilibrium,
        repo_root=root,
    )

    params = energy_fn.params()
    if len(cfg.learning_rates) != len(params):
        raise ValueError(
            "Expected training.optimizer.learning_rates to contain one value per "
            f"trainable parameter ({len(params)}). Provided value: {cfg.learning_rates!r}."
        )
    estimator = EquilibriumProp(
        params,
        free_layers,
        augmented_fn,
        cost_fn,
        training_minimizer,
        variant=ep["variant"],
        nudging=ep["nudging"],
        use_alternative_formula=ep["gradient_formula"] == "alternative",
    )
    optimizer_config = cfg.training["optimizer"]
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter.state], "lr": learning_rate}
            for parameter, learning_rate in zip(params, cfg.learning_rates)
        ],
        lr=0.0,
        momentum=optimizer_config["momentum"],
        dampening=optimizer_config["dampening"],
        weight_decay=optimizer_config["weight_decay"],
        nesterov=optimizer_config["nesterov"],
        maximize=optimizer_config["maximize"],
        foreach=optimizer_config["foreach"],
        differentiable=optimizer_config["differentiable"],
        fused=optimizer_config["fused"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.training["scheduler"]["gamma"],
        last_epoch=cfg.training["scheduler"]["initial_epoch"],
    )

    limits = cfg.training["batch_limits"]
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    best_accuracy = -math.inf
    best_checkpoint = output_dir / "model_best.pt"
    started = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        train_progress = _LiveProgress(
            "train",
            epoch + 1,
            cfg.num_epochs,
            _effective_batch_count(train_loader, limits["train"]),
        )
        train_loss, train_accuracy = _train_epoch(
            network,
            cost_fn,
            params,
            optimizer,
            estimator,
            inference_minimizer,
            train_loader,
            zero_grad_set_to_none=optimizer_config["zero_grad_set_to_none"],
            max_batches=limits["train"],
            progress=train_progress,
        )
        scheduler.step()
        evaluation_progress = _LiveProgress(
            "eval",
            epoch + 1,
            cfg.num_epochs,
            _effective_batch_count(evaluation_loader, limits["evaluation"]),
        )
        test_loss, test_accuracy = _evaluate(
            network,
            cost_fn,
            inference_minimizer,
            evaluation_loader,
            max_batches=limits["evaluation"],
            progress=evaluation_progress,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            energy_fn.save(best_checkpoint)
        print(
            f"epoch={epoch + 1}/{cfg.num_epochs} "
            f"train_loss={train_loss:.6g} train_accuracy={train_accuracy:.4f} "
            f"test_loss={test_loss:.6g} test_accuracy={test_accuracy:.4f}"
        )

    duration_seconds = time.perf_counter() - started
    checkpoint_policy = cfg.training["checkpoint"]
    final_checkpoint: Path | None = None
    if checkpoint_policy["save_final"]:
        final_checkpoint = output_dir / "model.pt"
        energy_fn.save(final_checkpoint)
    history_path = output_dir / "history.json"
    history_path.write_text(pretty_json_text(history), encoding="utf-8")

    assets = {
        "best_checkpoint": best_checkpoint,
        **simulation_assets(cfg.simulation, repo_root=root),
    }
    if final_checkpoint is not None:
        assets["final_checkpoint"] = final_checkpoint
    receipt = build_run_receipt(
        repo_root=root,
        resolved_config=cfg.document,
        execution=cfg.execution,
        device=device,
        assets=assets,
        source_documents=[
            {
                "owner": "training",
                "path": _relative_or_absolute(source_path, root),
                "sha256": source_sha256,
            },
            *[
                dict(record)
                for record in cfg.document["provenance"].get(
                    "config_sources", []
                )
                if not (
                    record["path"] == _relative_or_absolute(source_path, root)
                    and record["sha256"] == source_sha256
                )
            ],
        ],
        data_fingerprint=data_fingerprint,
        split_fingerprint=split_fingerprint,
        extra={
            "source_config": _relative_or_absolute(source_path, root),
            "duration_seconds": duration_seconds,
            "best_test_accuracy": best_accuracy,
            "final_test_accuracy": history["test_accuracy"][-1],
            "final_train_accuracy": history["train_accuracy"][-1],
        },
    )
    receipt_path = output_dir / "run_receipt.json"
    receipt_path.write_text(pretty_json_text(receipt), encoding="utf-8")
    metadata = {
        "resolved_config_sha256": receipt["resolved_config_sha256"],
        "receipt": receipt_path.name,
        "best_checkpoint": best_checkpoint.name,
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "final_checkpoint": None if final_checkpoint is None else final_checkpoint.name,
        "final_checkpoint_sha256": (
            None if final_checkpoint is None else sha256_file(final_checkpoint)
        ),
        "best_test_accuracy": best_accuracy,
        "final_test_accuracy": history["test_accuracy"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
    }
    (output_dir / "run_metadata.json").write_text(
        pretty_json_text(metadata),
        encoding="utf-8",
    )
    return TrainingResult(
        output_dir=output_dir,
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        history_path=history_path,
        receipt_path=receipt_path,
        history=history,
    )


def run_smoke_suite(repo_root: Path) -> list[TrainingResult]:
    """Run explicit generated one-batch snapshots for all three families."""

    names = (
        "digits_single_shockley.json",
        "digits_double_shockley.json",
        "digits_pwl.json",
    )
    overrides = (
        "/training/epochs=1",
        "/equilibrium/sweeps=2",
        "/training/batch_limits/train=1",
        "/training/batch_limits/evaluation=1",
    )
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
    return [
        run_training(
            repo_root,
            repo_root / "configs" / "train" / name,
            output_dir=repo_root
            / "outputs"
            / "training"
            / "smoke"
            / run_id
            / Path(name).stem,
            overrides=overrides,
        )
        for name in names
    ]


def _build_energy(cfg: TrainingConfig, device: torch.device) -> DeepResistiveEnergy:
    model = cfg.model
    if model["state_dtype"] != cfg.execution["backend"]["default_dtype"]:
        raise ValueError(
            "Expected model.state_dtype to match execution.backend.default_dtype."
        )
    energy_fn = DeepResistiveEnergy(
        layer_shapes=cfg.layer_shapes,
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
        weight_init_mode=cfg.weight_init_mode,
        bias_scale_mode=model["bias"]["scale_mode"],
        bias_interaction_type=model["bias"]["interaction"],
        bias_enabled=model["bias"]["enabled"],
        bias_initial_value=model["bias"]["initialization"]["value"],
        bias_minimum=model["bias"]["bounds"]["minimum"],
        bias_maximum=model["bias"]["bounds"]["maximum"],
        signed_weights=model["signed_weights"],
        conv_pipeline=[],
        learn_input_gain=model["amplification_learning"]["input_gain"],
        learn_voltage_amp=model["amplification_learning"]["voltage_factor"],
        learn_current_amp=model["amplification_learning"]["current_factor"],
    )
    energy_fn.set_device(device)
    return energy_fn


def _build_cost(cfg: TrainingConfig, output_layer):
    output = cfg.model["output"]
    if output["classes"] != 10:
        raise ValueError(
            f"Expected {cfg.dataset['source']} model.output.classes to be 10; "
            f"got {output['classes']}."
        )
    width = int(output_layer.shape[0])
    if output["encoding"] == "single_ended":
        if width != output["classes"]:
            raise ValueError("Configured single-ended output width does not match.")
        return SquaredError(output_layer)
    if output["encoding"] == "differential_pair":
        if width != 2 * output["classes"]:
            raise ValueError("Configured differential output width does not match.")
        return SquaredErrorPairedOutputs(output_layer, output["classes"])
    raise ValueError(f"Unsupported model.output.encoding: {output['encoding']!r}.")


def _build_loaders(
    cfg: TrainingConfig,
    *,
    repo_root: Path,
    download: bool,
) -> tuple[DataLoader, DataLoader, str, str]:
    if cfg.dataset["source"] == "sklearn_digits":
        return _digits_loaders(cfg)
    if cfg.dataset["source"] == "torchvision_mnist":
        return _mnist_loaders(cfg, repo_root=repo_root, download=download)
    raise ValueError(f"Unsupported data.source: {cfg.dataset['source']!r}.")


def _digits_loaders(
    cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader, str, str]:
    values, targets = load_digits(
        n_class=10,
        return_X_y=True,
        as_frame=False,
    )
    values = np.asarray(values)
    targets = np.asarray(targets)
    original_indices = np.arange(values.shape[0], dtype=np.int64)
    data_fingerprint = sha256_arrays(values, targets)

    subset = cfg.dataset["subset"]
    if subset["method"] == "seeded_random":
        if subset["count"] > len(values):
            raise ValueError(
                "Expected Digits subset.count not to exceed the loaded dataset "
                f"length {len(values)}. Provided value: {subset['count']}."
            )
        selected = np.random.RandomState(cfg.seed).permutation(len(values))[: subset["count"]]
        values = values[selected]
        targets = targets[selected]
        original_indices = original_indices[selected]
    elif subset["method"] != "all":
        raise ValueError(f"Unsupported Digits subset method: {subset['method']!r}.")

    preprocessing = cfg.dataset["preprocessing"]
    input_dtype = _TORCH_DTYPES[cfg.dataset["preprocessing"]["dtype"]]
    inputs = torch.tensor(values, dtype=input_dtype)
    inputs = (
        inputs / preprocessing["divisor"] * preprocessing["multiplier"]
        + preprocessing["offset"]
    )
    labels = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    split = cfg.dataset["split"]
    if split["rounding"] != "floor":
        raise ValueError("Expected data.split.rounding to be 'floor'.")
    train_size = math.floor(split["train_fraction"] * len(dataset))
    test_size = len(dataset) - train_size
    if train_size < 1 or test_size < 1:
        raise ValueError(
            "Expected the configured Digits split to produce at least one train "
            f"and evaluation example. Produced train={train_size}, evaluation={test_size}."
        )
    generator = torch.Generator().manual_seed(cfg.seed)
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
    return (
        _loader(cfg, train_dataset, shuffle=cfg.training["loader"]["train_shuffle"], generator=generator),
        _loader(cfg, test_dataset, shuffle=cfg.training["loader"]["evaluation_shuffle"]),
        data_fingerprint,
        split_fingerprint,
    )


def _mnist_loaders(
    cfg: TrainingConfig,
    *,
    repo_root: Path,
    download: bool,
) -> tuple[DataLoader, DataLoader, str, str]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - optional binary installation
        raise RuntimeError(
            "Expected torchvision to be importable for torchvision_mnist. "
            f"Provided environment raised: {exc!r}."
        ) from exc
    relative_root = Path(cfg.dataset["path"])
    if relative_root.is_absolute():
        raise ValueError("Expected MNIST data.path to be repository-relative.")
    dataset_root = (repo_root / relative_root).resolve()
    try:
        dataset_root.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("Expected MNIST data.path to stay inside the repository.") from exc
    transform = _build_mnist_transform(cfg.dataset["preprocessing"], transforms)
    try:
        train_dataset = datasets.MNIST(
            root=dataset_root,
            train=True,
            download=download,
            transform=transform,
        )
        evaluation_dataset = datasets.MNIST(
            root=dataset_root,
            train=False,
            download=download,
            transform=transform,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Expected MNIST under data.path or --download. "
            f"Provided value: path={dataset_root}, download={download}."
        ) from exc

    data_fingerprint = sha256_arrays(
        np.asarray(train_dataset.data),
        np.asarray(train_dataset.targets),
        np.asarray(evaluation_dataset.data),
        np.asarray(evaluation_dataset.targets),
    )
    subset = cfg.dataset["subset"]
    train_indices = np.arange(len(train_dataset), dtype=np.int64)
    evaluation_indices = np.arange(len(evaluation_dataset), dtype=np.int64)
    if subset["method"] == "prefix":
        if subset["train_count"] > len(train_dataset) or subset["evaluation_count"] > len(evaluation_dataset):
            raise ValueError(
                "Expected MNIST prefix counts not to exceed the official dataset "
                f"sizes. Provided train={subset['train_count']}/{len(train_dataset)}, "
                f"evaluation={subset['evaluation_count']}/{len(evaluation_dataset)}."
            )
        train_indices = train_indices[: subset["train_count"]]
        evaluation_indices = evaluation_indices[: subset["evaluation_count"]]
        train_dataset = Subset(train_dataset, train_indices.tolist())
        evaluation_dataset = Subset(
            evaluation_dataset, evaluation_indices.tolist()
        )
    elif subset["method"] != "all":
        raise ValueError(f"Unsupported MNIST subset method: {subset['method']!r}.")
    split_fingerprint = sha256_arrays(train_indices, evaluation_indices)
    generator = torch.Generator().manual_seed(cfg.seed)
    return (
        _loader(
            cfg,
            train_dataset,
            shuffle=cfg.training["loader"]["train_shuffle"],
            generator=generator,
        ),
        _loader(
            cfg,
            evaluation_dataset,
            shuffle=cfg.training["loader"]["evaluation_shuffle"],
        ),
        data_fingerprint,
        split_fingerprint,
    )


def _loader(
    cfg: TrainingConfig,
    dataset,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=cfg.training["loader"]["drop_last"],
        generator=generator,
        **dataloader_kwargs(cfg.execution),
    )


def _build_mnist_transform(preprocessing: dict[str, Any], transforms):
    dtype = _TORCH_DTYPES[preprocessing["dtype"]]
    steps: list[Any] = [transforms.ToTensor()]
    if dtype != torch.float32:
        steps.append(
            transforms.Lambda(lambda tensor, dtype=dtype: tensor.to(dtype=dtype))
        )
    if preprocessing["method"] == "normalized_tensor":
        steps.append(
            transforms.Normalize(
                (preprocessing["mean"],),
                (preprocessing["standard_deviation"],),
            )
        )
        if preprocessing["scale"] != 1.0:
            steps.append(
                transforms.Lambda(
                    lambda tensor, scale=preprocessing["scale"]: tensor * scale
                )
            )
    elif preprocessing["method"] != "to_tensor":
        raise ValueError(
            f"Unsupported MNIST preprocessing method: {preprocessing['method']!r}."
        )
    return transforms.Compose(steps)


def _train_epoch(
    network,
    cost_fn,
    params,
    optimizer,
    estimator,
    inference_minimizer,
    loader,
    *,
    zero_grad_set_to_none: bool,
    max_batches: int | None,
    progress: _LiveProgress,
) -> tuple[float, float]:
    loss_sum = 0.0
    correct = 0
    seen = 0
    progress.start()
    try:
        for batch_index, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(network._function._device)
            labels = labels.to(network._function._device)
            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
            network.set_input(inputs, reset=True)
            _validate_input_state(network, inputs)
            inference_minimizer.compute_equilibrium()
            _validate_finite_layers(network.free_layers(), context="free inference")
            cost_fn.set_target(labels)
            batch_loss = float(cost_fn.eval().mean().item())
            if not math.isfinite(batch_loss):
                raise FloatingPointError("Expected a finite training loss.")
            errors = cost_fn.error_fn()
            batch_size = int(labels.numel())
            loss_sum += batch_loss * batch_size
            correct += batch_size - int(errors.sum().item())
            seen += batch_size
            gradients = estimator.compute_gradient()
            _validate_finite_layers(network.free_layers(), context="nudged equilibrium")
            if len(gradients) < len(params):
                raise RuntimeError(
                    f"Expected at least {len(params)} gradients, got {len(gradients)}."
                )
            for parameter, gradient in zip(params, gradients):
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "Expected every equilibrium-propagation gradient to be "
                        f"finite. Provided parameter={parameter.name!r}."
                    )
                parameter.state.grad = gradient.detach()
            optimizer.step()
            for parameter in params:
                parameter.clamp_()
            progress.update(
                batch_index + 1,
                running_loss=loss_sum / seen,
                running_accuracy=correct / seen,
            )
    finally:
        progress.close()
    if seen == 0:
        raise ValueError("Expected at least one training batch.")
    return loss_sum / seen, correct / seen


def _evaluate(
    network,
    cost_fn,
    minimizer,
    loader,
    *,
    max_batches: int | None,
    progress: _LiveProgress,
) -> tuple[float, float]:
    loss_sum = 0.0
    correct = 0
    seen = 0
    progress.start()
    try:
        for batch_index, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(network._function._device)
            labels = labels.to(network._function._device)
            network.set_input(inputs, reset=True)
            _validate_input_state(network, inputs)
            minimizer.compute_equilibrium()
            _validate_finite_layers(network.free_layers(), context="evaluation equilibrium")
            cost_fn.set_target(labels)
            batch_size = int(labels.numel())
            batch_loss = float(cost_fn.eval().mean().item())
            if not math.isfinite(batch_loss):
                raise FloatingPointError("Expected a finite evaluation loss.")
            loss_sum += batch_loss * batch_size
            correct += batch_size - int(cost_fn.error_fn().sum().item())
            seen += batch_size
            progress.update(
                batch_index + 1,
                running_loss=loss_sum / seen,
                running_accuracy=correct / seen,
            )
    finally:
        progress.close()
    if seen == 0:
        raise ValueError("Expected at least one evaluation batch.")
    return loss_sum / seen, correct / seen


def _effective_batch_count(loader, maximum: int | None) -> int:
    return len(loader) if maximum is None else min(len(loader), maximum)


def _validate_input_state(network, raw_inputs: torch.Tensor) -> None:
    expected = tuple(network.layers()[0].shape)
    provided = tuple(network.layers()[0].state.shape[1:])
    if provided != expected:
        raise ValueError(
            "Expected doubled input tensor shape to match model.layer_shapes[0]. "
            f"Provided raw={tuple(raw_inputs.shape)}, doubled={provided}, expected={expected}."
        )


def _validate_finite_layers(layers, *, context: str) -> None:
    for depth, layer in enumerate(layers, start=1):
        if not torch.isfinite(layer.state).all():
            raise FloatingPointError(
                f"Expected finite {context} state at free layer {depth}."
            )


def _validate_relations(cfg: TrainingConfig) -> None:
    validate_execution_relations(cfg.execution)
    if cfg.model["topology"] != "dense":
        raise ValueError("Expected model.topology to be 'dense'.")
    expected_input = (
        (128,)
        if cfg.dataset["source"] == "sklearn_digits"
        else (2, 28, 28)
    )
    if cfg.layer_shapes[0] != expected_input:
        raise ValueError(
            f"Expected {cfg.dataset['source']} input layer shape {expected_input}, "
            f"got {cfg.layer_shapes[0]}."
        )
    if any(len(shape) != 1 for shape in cfg.layer_shapes[1:]):
        raise ValueError(
            "Expected every dense hidden/output layer shape to be one-dimensional."
        )
    expected_gains = len(cfg.layer_shapes) - 1
    if len(cfg.weight_gains) != expected_gains:
        raise ValueError(
            f"Expected one model.weight_gain per dense matrix ({expected_gains})."
        )
    if cfg.weight_min > cfg.weight_max:
        raise ValueError("Expected model weight minimum not to exceed maximum.")
    data_dtype = cfg.dataset["preprocessing"]["dtype"]
    if data_dtype != cfg.model["state_dtype"]:
        raise ValueError(
            "Expected data.preprocessing.dtype to match model.state_dtype. "
            f"Provided values: {data_dtype!r} and {cfg.model['state_dtype']!r}."
        )
    if cfg.non_linearity == "single_diode_exponential" and any(
        shape[0] % 2 for shape in cfg.layer_shapes[1:-1]
    ):
        raise ValueError(
            "Expected every single-Shockley hidden width to be even."
        )
    edges = len(cfg.layer_shapes) - 1
    hidden_biases = len(cfg.layer_shapes) - 2 if cfg.model["bias"]["enabled"] else 0
    parameter_count = edges * (2 if cfg.model["signed_weights"] else 1) + hidden_biases
    if len(cfg.learning_rates) != parameter_count:
        raise ValueError(
            "Expected training.optimizer.learning_rates to contain one value per "
            f"trainable parameter ({parameter_count}); got {len(cfg.learning_rates)}."
        )
    execution_dtype = cfg.execution["backend"]["default_dtype"]
    if execution_dtype != cfg.model["state_dtype"]:
        raise ValueError(
            "Expected execution.backend.default_dtype to match model.state_dtype. "
            f"Provided values: {execution_dtype!r} and "
            f"{cfg.model['state_dtype']!r}."
        )
    if cfg.equilibrium["method"] != "fixed_sweeps":
        raise ValueError(
            "Expected training equilibrium.method to be 'fixed_sweeps'; adaptive "
            "phase lengths change the EP estimator protocol."
        )
    batch_size = cfg.training["loader"]["batch_size"]
    drop_last = cfg.training["loader"]["drop_last"]
    subset = cfg.dataset["subset"]
    if cfg.dataset["source"] == "sklearn_digits":
        total = 1797 if subset["method"] == "all" else subset["count"]
        train_examples = math.floor(cfg.dataset["split"]["train_fraction"] * total)
        evaluation_examples = total - train_examples
    else:
        train_examples = 60000 if subset["method"] == "all" else subset["train_count"]
        evaluation_examples = (
            10000 if subset["method"] == "all" else subset["evaluation_count"]
        )
    if train_examples < 1 or evaluation_examples < 1:
        raise ValueError(
            "Expected the configured data policy to select at least one training "
            "and evaluation example."
        )
    if drop_last and (
        train_examples < batch_size or evaluation_examples < batch_size
    ):
        raise ValueError(
            "Expected training.loader.drop_last=true to leave at least one full "
            "training and evaluation batch."
        )
    output = cfg.model["output"]
    if output["classes"] != 10:
        raise ValueError(
            f"Expected {cfg.dataset['source']} model.output.classes to be 10; "
            f"got {output['classes']}."
        )
    width = cfg.layer_shapes[-1][0]
    expected_width = (
        output["classes"]
        if output["encoding"] == "single_ended"
        else 2 * output["classes"]
    )
    if width != expected_width:
        raise ValueError(
            f"Expected output width {expected_width} for {output!r}, got {width}."
        )
    if cfg.equilibrium["initial_state"] != "zeros":
        raise ValueError("Expected training equilibrium.initial_state to be 'zeros'.")


def _input_mode(model: Mapping[str, Any]) -> str:
    if model["input_encoding"] != "signed_pair":
        raise ValueError(
            "Expected model.input_encoding to be 'signed_pair'. "
            f"Provided value: {model['input_encoding']!r}."
        )
    return "train"


def _parse_override(value: str) -> JsonPointerOverride:
    if "=" not in value:
        raise ConfigurationOverrideError(
            "Expected --override JSON_POINTER=JSON_VALUE. "
            f"Provided value: {value!r}."
        )
    pointer, encoded = value.split("=", maxsplit=1)
    try:
        parsed = json.loads(
            encoded,
            parse_constant=lambda constant: (_raise_non_finite(constant)),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ConfigurationOverrideError(
            f"Invalid JSON value in override {value!r}: {exc}."
        ) from exc
    return JsonPointerOverride(pointer=pointer, value=parsed)


def _raise_non_finite(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is not permitted")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _capture_source_record(
    source: Path | Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, str] | None:
    if isinstance(source, Mapping):
        return None
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    return {
        "owner": "training_source",
        "path": _relative_or_absolute(path, repo_root.resolve()),
        "sha256": sha256_file(path),
    }


def _verify_and_stamp_source(
    result: CompositionResult,
    source_record: dict[str, str] | None,
    *,
    repo_root: Path,
) -> None:
    if source_record is None:
        return
    source_path = Path(source_record["path"])
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    if sha256_file(source_path.resolve()) != source_record["sha256"]:
        raise RuntimeError(
            "Training source changed while it was being resolved; retry from "
            "stable source bytes."
        )
    sources = result.document["provenance"].setdefault("config_sources", [])
    if not any(item["owner"] == "training_source" for item in sources):
        sources.insert(0, source_record)
    validate_document(result.document, "training-v2.schema.json", repo_root=repo_root)


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "load_training_config",
    "resolve_training_config",
    "run_smoke_suite",
    "run_training",
]
