from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network
from model.resistive.minimizer import MinimizerSettings, QuadraticMinimizer
from model.resistive.network import DeepResistiveEnergy
from repro.device import resolve_device
from repro.iv_data import load_iv_data
from training.sgd import AugmentedFunction, EquilibriumProp


PAPER_NONLINEARITIES = (
    "single_diode_exponential",
    "double_diode_exponential",
    "experimental",
)

_SIMULATOR_PROFILE_FIELDS = frozenset(
    {
        "non_linearity",
        "voltage_amp",
        "current_amp",
        "exponential_diode_param",
        "iv_data_path",
        "minimizer_impl",
        "mode",
        "single_diode_updater",
        "double_diode_updater",
        "overrelaxation_factor",
        "adaptive_equilibrium",
        "rel_tol",
        "vn_tol",
        "use_polish",
        "max_newton_iters",
        "z_thresh",
        "exp_clip",
        "damping",
        "experimental_newton_max_steps",
        "experimental_newton_tol",
    }
)
_SIMULATOR_PROFILE_METADATA_FIELDS = frozenset({"$schema", "description", "source"})
_LEGACY_UNUSED_TRAINING_FIELDS = frozenset(
    {"quadratic_diode_param", "hard_sigmoid_param"}
)


@dataclass(frozen=True)
class TrainingConfig:
    description: str
    dataset: dict[str, Any]
    layer_shapes: list[tuple[int, ...]]
    non_linearity: str
    weight_gains: list[float]
    weight_min: float
    weight_max: float
    weight_init_mode: str
    input_gain: float
    voltage_amp: float
    current_amp: float
    exponential_diode_param: dict[str, Any]
    batch_size: int
    num_epochs: int
    num_iterations: int
    learning_rates: list[float]
    scheduler_gamma: float
    nudging: float
    ep_variant: str
    minimizer_impl: str
    mode: str
    single_diode_updater: str
    double_diode_updater: str
    overrelaxation_factor: float
    adaptive_equilibrium: bool
    rel_tol: float
    vn_tol: float
    use_polish: bool
    max_newton_iters: int
    z_thresh: float
    exp_clip: float
    damping: float
    experimental_newton_max_steps: int
    experimental_newton_tol: float
    seed: int
    iv_data_path: str | None


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    final_checkpoint: Path
    best_checkpoint: Path
    history_path: Path
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

    def update(self, batch: int, *, running_loss: float, running_accuracy: float) -> None:
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


def load_training_config(path: Path, *, repo_root: Path) -> tuple[TrainingConfig, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "Expected the training configuration to contain a JSON object. "
            f"Provided value in {path}: {type(raw).__name__}."
        )
    raw = _compose_simulator_profile(raw, repo_root=repo_root)
    for legacy_field in _LEGACY_UNUSED_TRAINING_FIELDS:
        raw.pop(legacy_field, None)

    dataset = _required_dict(raw, "dataset")
    dataset_name = str(dataset.get("name", "")).strip().lower()
    if dataset_name not in {"digits", "mnist"}:
        raise ValueError(
            "Expected config 'dataset.name' to be 'digits' or 'mnist'. "
            f"Provided value: {dataset.get('name')!r}."
        )

    non_linearity = _required_str(raw, "non_linearity")
    if non_linearity not in PAPER_NONLINEARITIES:
        raise ValueError(
            f"Expected config 'non_linearity' to be one of {PAPER_NONLINEARITIES}. "
            f"Provided value: {non_linearity!r}."
        )

    if non_linearity in {"single_diode_exponential", "double_diode_exponential"}:
        exponential = _required_dict(raw, "exponential_diode_param")
        _require_keys(exponential, ("I_s", "V_t", "V_off"), "exponential_diode_param")
    else:
        exponential = {}

    shapes_raw = _required_list(raw, "layer_shapes")
    if len(shapes_raw) < 2 or any(not isinstance(shape, list) or not shape for shape in shapes_raw):
        raise ValueError(
            "Expected config 'layer_shapes' to contain at least two non-empty integer lists. "
            f"Provided value: {shapes_raw!r}."
        )
    layer_shapes = [tuple(int(value) for value in shape) for shape in shapes_raw]
    if any(any(value <= 0 for value in shape) for shape in layer_shapes):
        raise ValueError(
            "Expected every layer shape dimension to be positive. "
            f"Provided value: {shapes_raw!r}."
        )

    weight_gains = [float(value) for value in _required_list(raw, "weight_gains")]
    expected_gains = len(layer_shapes) - 1
    if len(weight_gains) != expected_gains:
        raise ValueError(
            f"Expected config 'weight_gains' to contain {expected_gains} values. "
            f"Provided value: {weight_gains!r}."
        )

    iv_data_path = raw.get("iv_data_path")
    if iv_data_path is not None:
        candidate = Path(str(iv_data_path)).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        iv_data_path = str(candidate.resolve())
    if non_linearity == "experimental" and iv_data_path is None:
        raise ValueError(
            "Expected config 'iv_data_path' for the measured/PWL nonlinearity. "
            "Provided value: None."
        )

    adaptive_equilibrium = _required(raw, "adaptive_equilibrium")
    if not isinstance(adaptive_equilibrium, bool):
        raise ValueError(
            "Expected config 'adaptive_equilibrium' to be the boolean false during training. "
            f"Provided value: {adaptive_equilibrium!r}."
        )
    use_polish = _required(raw, "use_polish")
    if not isinstance(use_polish, bool):
        raise ValueError(
            "Expected config 'use_polish' to be a boolean. "
            f"Provided value: {use_polish!r}."
        )

    config = TrainingConfig(
        description=str(raw.get("description", "")),
        dataset=dataset,
        layer_shapes=layer_shapes,
        non_linearity=non_linearity,
        weight_gains=weight_gains,
        weight_min=float(_required(raw, "weight_min")),
        weight_max=float(_required(raw, "weight_max")),
        weight_init_mode=_required_str(raw, "weight_init_mode"),
        input_gain=float(_required(raw, "input_gain")),
        voltage_amp=float(_required(raw, "voltage_amp")),
        current_amp=float(_required(raw, "current_amp")),
        exponential_diode_param=exponential,
        batch_size=int(_required(raw, "batch_size")),
        num_epochs=int(_required(raw, "num_epochs")),
        num_iterations=int(_required(raw, "num_iterations")),
        learning_rates=[float(value) for value in _required_list(raw, "learning_rates")],
        scheduler_gamma=float(_required(raw, "scheduler_gamma")),
        nudging=float(_required(raw, "nudging")),
        ep_variant=_required_str(raw, "ep_variant"),
        minimizer_impl=_required_str(raw, "minimizer_impl"),
        mode=_required_str(raw, "mode"),
        single_diode_updater=_required_str(raw, "single_diode_updater"),
        double_diode_updater=_required_str(raw, "double_diode_updater"),
        overrelaxation_factor=float(_required(raw, "overrelaxation_factor")),
        adaptive_equilibrium=adaptive_equilibrium,
        rel_tol=float(_required(raw, "rel_tol")),
        vn_tol=float(_required(raw, "vn_tol")),
        use_polish=use_polish,
        max_newton_iters=int(_required(raw, "max_newton_iters")),
        z_thresh=float(_required(raw, "z_thresh")),
        exp_clip=float(_required(raw, "exp_clip")),
        damping=float(_required(raw, "damping")),
        experimental_newton_max_steps=int(_required(raw, "experimental_newton_max_steps")),
        experimental_newton_tol=float(_required(raw, "experimental_newton_tol")),
        seed=int(_required(raw, "seed")),
        iv_data_path=iv_data_path,
    )
    _validate_scalar_fields(config)
    return config, raw


def run_training(
    repo_root: Path,
    config_path: Path,
    *,
    device: str = "cpu",
    output_dir: Path | None = None,
    epochs: int | None = None,
    num_iterations: int | None = None,
    max_batches: int | None = None,
    max_eval_batches: int | None = None,
    download: bool = False,
) -> TrainingResult:
    config_path = config_path.expanduser().resolve()
    cfg, raw = load_training_config(config_path, repo_root=repo_root)
    resolved_epochs = cfg.num_epochs if epochs is None else int(epochs)
    resolved_iterations = cfg.num_iterations if num_iterations is None else int(num_iterations)
    if resolved_epochs <= 0:
        raise ValueError(f"Expected epochs to be positive. Provided value: {resolved_epochs!r}.")
    if resolved_iterations <= 0:
        raise ValueError(
            f"Expected num_iterations to be positive. Provided value: {resolved_iterations!r}."
        )

    torch_device = resolve_device(device)
    _set_seed(cfg.seed)
    if output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = repo_root / "outputs" / "training" / f"{config_path.stem}_{stamp}"
    elif not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = _build_loaders(
        cfg,
        repo_root=repo_root,
        download=download,
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
    )
    energy_fn.set_device(torch_device)
    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    output_width = int(output_layer.shape[0])
    if output_width == 10:
        cost_fn = SquaredError(output_layer)
    elif output_width == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(
            "Expected the output layer width to be 10 or 20 for Digits/MNIST. "
            f"Provided value: {output_layer.shape!r}."
        )

    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    inference_minimizer = _build_minimizer(cfg, energy_fn, free_layers, resolved_iterations)
    training_minimizer = _build_minimizer(cfg, augmented_fn, free_layers, resolved_iterations)
    params = energy_fn.params()
    if len(cfg.learning_rates) != len(params):
        raise ValueError(
            f"Expected config 'learning_rates' to contain {len(params)} values, one per trainable parameter. "
            f"Provided value: {cfg.learning_rates!r}."
        )
    estimator = EquilibriumProp(
        params,
        free_layers,
        augmented_fn,
        cost_fn,
        training_minimizer,
        variant=cfg.ep_variant,
        nudging=cfg.nudging,
    )
    optimizer = torch.optim.SGD(
        [
            {"params": [param.state], "lr": learning_rate}
            for param, learning_rate in zip(params, cfg.learning_rates)
        ],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.scheduler_gamma,
    )

    resolved = dict(raw)
    resolved["config_path"] = _relative_or_absolute(config_path, repo_root)
    resolved["device"] = str(torch_device)
    resolved["num_epochs"] = resolved_epochs
    resolved["num_iterations"] = resolved_iterations
    resolved["max_batches"] = max_batches
    resolved["max_eval_batches"] = max_eval_batches
    resolved["download"] = bool(download)
    resolved["iv_data_path_resolved"] = cfg.iv_data_path
    _write_json(output_dir / "config.resolved.json", resolved)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    best_accuracy = -math.inf
    best_checkpoint = output_dir / "model_best.pt"
    start = time.perf_counter()

    for epoch in range(resolved_epochs):
        train_progress = _LiveProgress(
            "train",
            epoch + 1,
            resolved_epochs,
            _effective_batch_count(train_loader, max_batches),
        )
        train_loss, train_accuracy = _train_epoch(
            network,
            cost_fn,
            params,
            optimizer,
            estimator,
            inference_minimizer,
            train_loader,
            max_batches=max_batches,
            progress=train_progress,
        )
        # Match the paper loop: epoch 1 uses the configured rates, then rates
        # are decayed before epoch 2 begins.
        scheduler.step()
        eval_progress = _LiveProgress(
            "eval",
            epoch + 1,
            resolved_epochs,
            _effective_batch_count(test_loader, max_eval_batches),
        )
        test_loss, test_accuracy = _evaluate(
            network,
            cost_fn,
            inference_minimizer,
            test_loader,
            max_batches=max_eval_batches,
            progress=eval_progress,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            energy_fn.save(best_checkpoint)
        print(
            f"epoch={epoch + 1}/{resolved_epochs} "
            f"train_loss={train_loss:.6g} train_accuracy={train_accuracy:.4f} "
            f"test_loss={test_loss:.6g} test_accuracy={test_accuracy:.4f}"
        )

    duration_seconds = time.perf_counter() - start
    final_checkpoint = output_dir / "model.pt"
    energy_fn.save(final_checkpoint)
    history_path = output_dir / "history.json"
    _write_json(history_path, history)
    metadata = {
        "non_linearity": cfg.non_linearity,
        "dataset": cfg.dataset.get("name"),
        "layer_shapes": [list(shape) for shape in cfg.layer_shapes],
        "learning_rates": cfg.learning_rates,
        "scheduler_gamma": cfg.scheduler_gamma,
        "final_learning_rates": [group["lr"] for group in optimizer.param_groups],
        "scheduler_step_timing": "after-training-epoch",
        "input_gain": cfg.input_gain,
        "adaptive_equilibrium": cfg.adaptive_equilibrium,
        "seed": cfg.seed,
        "device": str(torch_device),
        "epochs": resolved_epochs,
        "num_iterations": resolved_iterations,
        "max_batches": max_batches,
        "max_eval_batches": max_eval_batches,
        "duration_seconds": duration_seconds,
        "best_test_accuracy": best_accuracy,
        "final_test_accuracy": history["test_accuracy"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_checkpoint": final_checkpoint.name,
        "final_checkpoint_sha256": _sha256(final_checkpoint),
        "best_checkpoint": best_checkpoint.name,
        "best_checkpoint_sha256": _sha256(best_checkpoint),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    return TrainingResult(
        output_dir=output_dir,
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        history_path=history_path,
        history=history,
    )


def run_smoke_suite(repo_root: Path, *, device: str = "cpu") -> list[TrainingResult]:
    config_names = (
        "digits_single_shockley.json",
        "digits_double_shockley.json",
        "digits_pwl.json",
    )
    results = []
    for config_name in config_names:
        output = repo_root / "outputs" / "training" / "smoke" / Path(config_name).stem
        result = run_training(
            repo_root,
            repo_root / "configs" / "train" / config_name,
            device=device,
            output_dir=output,
            epochs=1,
            num_iterations=4,
            max_batches=1,
            max_eval_batches=1,
        )
        results.append(result)
    return results


def _build_loaders(
    cfg: TrainingConfig,
    *,
    repo_root: Path,
    download: bool,
) -> tuple[DataLoader, DataLoader]:
    dataset_name = str(cfg.dataset["name"]).lower()
    if dataset_name == "digits":
        digits = load_digits()
        x = digits.data
        y = digits.target
        num_samples = cfg.dataset.get("num_samples")
        if num_samples is not None:
            count = int(num_samples)
            if count <= 1:
                raise ValueError(
                    "Expected config 'dataset.num_samples' to be greater than one or null. "
                    f"Provided value: {num_samples!r}."
                )
            if count < x.shape[0]:
                indices = np.random.RandomState(cfg.seed).permutation(x.shape[0])[:count]
                x = x[indices]
                y = y[indices]
        x_tensor = torch.tensor(x, dtype=torch.float32) / 16.0 * 2.0 - 1.0
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
        train_fraction = float(cfg.dataset.get("train_fraction", 0.8))
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(
                "Expected config 'dataset.train_fraction' to lie strictly between zero and one. "
                f"Provided value: {train_fraction!r}."
            )
        train_size = int(train_fraction * len(dataset))
        test_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(cfg.seed)
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator,
        )
        return (
            DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator),
            DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False),
        )

    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on optional binary installation
        raise RuntimeError(
            "Expected torchvision to be importable for dataset.name='mnist'. "
            f"Provided environment raised: {exc!r}."
        ) from exc
    root_value = cfg.dataset.get("root", "data/external/mnist")
    dataset_root = Path(str(root_value)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    transform = _build_mnist_transform(cfg.dataset, transforms)
    try:
        train_dataset = datasets.MNIST(
            root=dataset_root,
            train=True,
            download=download,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=dataset_root,
            train=False,
            download=download,
            transform=transform,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Expected MNIST under the configured dataset root or --download to be supplied. "
            f"Provided value: root={dataset_root}, download={download}."
        ) from exc
    train_limit = cfg.dataset.get("train_samples")
    test_limit = cfg.dataset.get("test_samples")
    if train_limit is not None:
        train_dataset = Subset(train_dataset, range(min(int(train_limit), len(train_dataset))))
    if test_limit is not None:
        test_dataset = Subset(test_dataset, range(min(int(test_limit), len(test_dataset))))
    generator = torch.Generator().manual_seed(cfg.seed)
    return (
        DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator),
        DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False),
    )


def _build_mnist_transform(dataset: dict[str, Any], transforms):
    transform_steps: list[Any] = [transforms.ToTensor()]
    if bool(dataset.get("normalize", True)):
        normalize_mean = float(dataset.get("normalize_mean", 0.1307))
        normalize_standard_deviation = float(
            dataset.get("normalize_standard_deviation", 0.3081)
        )
        # In the historical runner, ``normalize_std`` was a multiplier applied
        # after standard MNIST normalization. Keep it as a legacy alias while
        # using an unambiguous field name in this repository.
        normalize_scale = float(
            dataset.get("normalize_scale", dataset.get("normalize_std", 1.0))
        )
        if normalize_standard_deviation <= 0.0:
            raise ValueError(
                "Expected config 'dataset.normalize_standard_deviation' to be positive. "
                f"Provided value: {normalize_standard_deviation!r}."
            )
        if normalize_scale <= 0.0:
            raise ValueError(
                "Expected config 'dataset.normalize_scale' to be positive. "
                f"Provided value: {normalize_scale!r}."
            )
        transform_steps.append(
            transforms.Normalize(
                (normalize_mean,),
                (normalize_standard_deviation,),
            )
        )
        if not math.isclose(normalize_scale, 1.0):
            transform_steps.append(
                transforms.Lambda(lambda tensor, scale=normalize_scale: tensor * scale)
            )
    return transforms.Compose(transform_steps)


def _build_minimizer(cfg: TrainingConfig, fn, free_layers, num_iterations: int):
    if cfg.minimizer_impl != "custom":
        raise ValueError(
            "Expected config 'minimizer_impl' to be 'custom'. "
            f"Provided value: {cfg.minimizer_impl!r}."
        )
    settings = MinimizerSettings(
        rel_tol=cfg.rel_tol,
        vn_tol=cfg.vn_tol,
        use_polish=cfg.use_polish,
        max_newton_iters=cfg.max_newton_iters,
        z_thresh=cfg.z_thresh,
        exp_clip=cfg.exp_clip,
        experimental_newton_tol=cfg.experimental_newton_tol,
    )
    iv_data = load_iv_data(cfg.iv_data_path) if cfg.non_linearity == "experimental" else None
    return QuadraticMinimizer(
        fn=fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode=cfg.mode,
        non_linearity=cfg.non_linearity,
        quadratic_diode_param={},
        exponential_diode_param=cfg.exponential_diode_param,
        voltage_amp=cfg.voltage_amp,
        current_amp=cfg.current_amp,
        iv_data=iv_data,
        double_diode_updater=cfg.double_diode_updater,
        adaptive_equilibrium=cfg.adaptive_equilibrium,
        overrelaxation_factor=cfg.overrelaxation_factor,
        single_diode_updater=cfg.single_diode_updater,
        damping=cfg.damping,
        experimental_newton_max_steps=cfg.experimental_newton_max_steps,
        minimizer_settings=settings,
    )


def _train_epoch(
    network,
    cost_fn,
    params,
    optimizer,
    estimator,
    inference_minimizer,
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
            optimizer.zero_grad(set_to_none=True)
            network.set_input(inputs, reset=True)
            _validate_input_state(network, inputs)
            inference_minimizer.compute_equilibrium()
            cost_fn.set_target(labels)
            batch_loss = float(cost_fn.eval().mean().item())
            errors = cost_fn.error_fn()
            batch_size = int(labels.numel())
            loss_sum += batch_loss * batch_size
            correct += batch_size - int(errors.sum().item())
            seen += batch_size

            gradients = estimator.compute_gradient()
            if len(gradients) < len(params):
                raise RuntimeError(
                    f"Expected at least {len(params)} parameter gradients. "
                    f"Provided value: {len(gradients)}."
                )
            for param, gradient in zip(params, gradients):
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "Expected every equilibrium-propagation gradient to be finite. "
                        f"Provided value: parameter={param.name!r}."
                    )
                param.state.grad = gradient.detach()
            optimizer.step()
            for param in params:
                param.clamp_()
            progress.update(
                batch_index + 1,
                running_loss=loss_sum / seen,
                running_accuracy=correct / seen,
            )
    finally:
        progress.close()

    if seen == 0:
        raise ValueError(
            "Expected at least one training batch after applying max_batches. "
            f"Provided value: max_batches={max_batches!r}."
        )
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
            cost_fn.set_target(labels)
            batch_size = int(labels.numel())
            loss_sum += float(cost_fn.eval().mean().item()) * batch_size
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
        raise ValueError(
            "Expected at least one evaluation batch after applying max_eval_batches. "
            f"Provided value: max_eval_batches={max_batches!r}."
        )
    return loss_sum / seen, correct / seen


def _effective_batch_count(loader, max_batches: int | None) -> int:
    available = len(loader)
    if max_batches is None:
        return available
    return min(available, max(0, int(max_batches)))


def _validate_input_state(network, raw_inputs: torch.Tensor) -> None:
    expected = tuple(network.layers()[0].shape)
    provided = tuple(network.layers()[0].state.shape[1:])
    if provided != expected:
        raise ValueError(
            "Expected doubled input tensor shape to match config 'layer_shapes[0]'. "
            f"Provided value: raw_shape={tuple(raw_inputs.shape)}, doubled_shape={provided}, expected={expected}."
        )


def _compose_simulator_profile(
    training: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Expand a repo-relative simulator profile into a training configuration."""

    if "simulator_profile" not in training:
        return dict(training)

    reference = training["simulator_profile"]
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(
            "Expected config field 'simulator_profile' to be a non-empty, "
            f"repo-relative path. Provided value: {reference!r}."
        )

    root = repo_root.expanduser().resolve()
    relative_path = Path(reference.strip())
    if relative_path.is_absolute():
        raise ValueError(
            "Expected config field 'simulator_profile' to be relative to the repository root. "
            f"Provided value: {reference!r}."
        )
    profile_path = (root / relative_path).resolve()
    try:
        profile_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Expected config field 'simulator_profile' to resolve inside the repository root. "
            f"Provided value: {reference!r}."
        ) from exc

    if not profile_path.is_file():
        raise ValueError(
            "Expected config field 'simulator_profile' to name an existing JSON file. "
            f"Provided value: {reference!r} (resolved to {profile_path})."
        )
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Expected config field 'simulator_profile' to name a readable JSON file. "
            f"Provided value: {reference!r} ({exc})."
        ) from exc
    if not isinstance(profile, dict):
        raise ValueError(
            "Expected the simulator profile to contain a JSON object. "
            f"Provided value in {profile_path}: {type(profile).__name__}."
        )

    allowed_profile_fields = (
        _SIMULATOR_PROFILE_FIELDS
        | _SIMULATOR_PROFILE_METADATA_FIELDS
        | _LEGACY_UNUSED_TRAINING_FIELDS
    )
    unknown_fields = sorted(set(profile).difference(allowed_profile_fields))
    if unknown_fields:
        raise ValueError(
            "Expected simulator profile fields to be recognized simulator settings or metadata. "
            f"Provided unknown fields in {profile_path}: {unknown_fields}."
        )

    simulator_values = {
        key: value for key, value in profile.items() if key in _SIMULATOR_PROFILE_FIELDS
    }
    collisions = sorted(set(training).intersection(simulator_values))
    if collisions:
        raise ValueError(
            "Expected simulator settings to come either from 'simulator_profile' or inline, "
            "not both. "
            f"Provided duplicate fields: {collisions}."
        )

    expanded = dict(training)
    expanded.pop("simulator_profile")
    expanded.pop("simulator_profile_source", None)
    expanded.pop("simulator_profile_sha256", None)
    expanded.update(simulator_values)
    expanded["simulator_profile_source"] = _relative_or_absolute(profile_path, root)
    expanded["simulator_profile_sha256"] = _sha256(profile_path)
    return expanded


def _validate_scalar_fields(cfg: TrainingConfig) -> None:
    positive_fields = {
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.num_epochs,
        "num_iterations": cfg.num_iterations,
        "nudging": cfg.nudging,
        "scheduler_gamma": cfg.scheduler_gamma,
        "rel_tol": cfg.rel_tol,
        "vn_tol": cfg.vn_tol,
        "exp_clip": cfg.exp_clip,
        "damping": cfg.damping,
        "overrelaxation_factor": cfg.overrelaxation_factor,
        "experimental_newton_max_steps": cfg.experimental_newton_max_steps,
        "experimental_newton_tol": cfg.experimental_newton_tol,
    }
    for name, value in positive_fields.items():
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(
                f"Expected config '{name}' to be finite and positive. "
                f"Provided value: {value!r}."
            )
    if not math.isfinite(cfg.z_thresh) or cfg.z_thresh <= 1.0:
        raise ValueError(
            "Expected config 'z_thresh' to be finite and greater than 1 for the "
            "large-z Lambert-W expansion. "
            f"Provided value: {cfg.z_thresh!r}."
        )
    if cfg.max_newton_iters < 0:
        raise ValueError(
            "Expected config 'max_newton_iters' to be non-negative. "
            f"Provided value: {cfg.max_newton_iters!r}."
        )
    if cfg.adaptive_equilibrium:
        raise ValueError(
            "Expected config 'adaptive_equilibrium' to be false during training. "
            f"Provided value: {cfg.adaptive_equilibrium!r}."
        )
    if cfg.weight_min > cfg.weight_max:
        raise ValueError(
            "Expected config 'weight_min' to be no greater than 'weight_max'. "
            f"Provided value: weight_min={cfg.weight_min}, weight_max={cfg.weight_max}."
        )
    if cfg.ep_variant not in {"positive", "negative", "centered"}:
        raise ValueError(
            "Expected config 'ep_variant' to be 'positive', 'negative', or 'centered'. "
            f"Provided value: {cfg.ep_variant!r}."
        )
    if cfg.mode not in {"asynchronous", "synchronous", "forward", "backward"}:
        raise ValueError(
            "Expected config 'mode' to be a supported minimizer mode. "
            f"Provided value: {cfg.mode!r}."
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _required(data: dict[str, Any], name: str) -> Any:
    if name not in data or data[name] is None:
        raise ValueError(
            f"Expected config field '{name}' to be present. Provided value: {data.get(name)!r}."
        )
    return data[name]


def _required_str(data: dict[str, Any], name: str) -> str:
    value = _required(data, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Expected config field '{name}' to be a non-empty string. Provided value: {value!r}."
        )
    return value.strip()


def _required_dict(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = _required(data, name)
    if not isinstance(value, dict):
        raise ValueError(f"Expected config field '{name}' to be an object. Provided value: {value!r}.")
    return dict(value)


def _required_list(data: dict[str, Any], name: str) -> list[Any]:
    value = _required(data, name)
    if not isinstance(value, list):
        raise ValueError(f"Expected config field '{name}' to be a list. Provided value: {value!r}.")
    return list(value)


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(
            f"Expected config '{label}' to include keys {keys}. "
            f"Provided value missing: {missing}."
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
