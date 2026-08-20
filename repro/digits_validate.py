from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset, random_split

from labs.custom_minimizer import CustomQuadraticMinimizer, MinimizerSettings
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network
from model.resistive.network import DeepResistiveEnergy
from repro.config import RuntimeConfig, load_runtime_config, parse_layer_shapes
from repro.manifest import ReproJob


DEFAULT_NUM_POINTS = 2000
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


@dataclass(frozen=True)
class ValidationResult:
    run_dir: Path
    states_npz: Path
    metadata_json: Path
    accuracy: float


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def run_validation(pack_root: Path, job: ReproJob, *, device: str = "cpu") -> ValidationResult:
    cfg = load_runtime_config(job.config_path(pack_root), pack_root=pack_root, num_iterations=job.num_iterations)
    _set_seed(cfg.seed)
    torch_device = _resolve_device(device)
    run_dir = job.output_dir(pack_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = _digits_loaders(cfg.batch_size, torch_device, cfg.seed)
    del train_loader

    energy_fn, network, free_layers, cost_fn, _, layer_shapes = _build_energy_stack(
        cfg=cfg,
        weights_path=job.weights_path(pack_root),
        device=torch_device,
    )
    minimizer = _build_minimizer(cfg, energy_fn, free_layers)

    inputs_batches = []
    labels_batches = []
    indices_batches = []
    states = {layer.name: [] for layer in network.layers()}
    total = 0
    correct = 0

    for x, y, idx in test_loader:
        network.set_input(x, reset=True)
        minimizer.compute_equilibrium()

        inputs_batches.append(x.detach().cpu())
        labels_batches.append(y.detach().cpu())
        indices_batches.append(idx.detach().cpu())
        for layer in network.layers():
            states[layer.name].append(layer.state.detach().cpu())

        cost_fn.set_target(y)
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
    np.savez(states_npz, **{name: torch.cat(values, dim=0).numpy() for name, values in states.items()})

    accuracy = correct / total if total else 0.0
    metadata = {
        "job_id": job.job_id,
        "group": job.group,
        "family": job.family,
        "hidden_layers": job.hidden_layers,
        "hidden_size": job.hidden_size,
        "config": job.config,
        "weights": job.weights,
        "reference_npz": job.reference_npz,
        "dims": cfg.dims,
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "device": str(torch_device),
        "batch_size": cfg.batch_size,
        "num_iterations": cfg.num_iterations,
        "rel_tol": cfg.rel_tol,
        "vn_tol": cfg.vn_tol,
        "exp_clip": cfg.exp_clip,
        "max_newton_iters": cfg.max_newton_iters,
        "overrelaxation_factor": cfg.overrelaxation_factor,
        "adaptive_equilibrium": cfg.adaptive_equilibrium,
        "validation_accuracy": accuracy,
        "validation_correct": correct,
        "validation_total": total,
        "validation_inputs": _pack_rel(inputs_npz, pack_root),
        "validation_states": _pack_rel(states_npz, pack_root),
    }
    metadata_json = run_dir / "validation_metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ValidationResult(run_dir=run_dir, states_npz=states_npz, metadata_json=metadata_json, accuracy=accuracy)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _pack_rel(path: Path, pack_root: Path) -> str:
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA to be available for --device cuda. Provided value: cuda.")
    return torch.device(value)


def _digits_loaders(batch_size: int, device: torch.device, seed: int | None):
    digits = load_digits()
    x = digits.data
    y = digits.target
    if DEFAULT_NUM_POINTS < x.shape[0]:
        rng = np.random.RandomState(seed or 0)
        idx = rng.permutation(x.shape[0])[:DEFAULT_NUM_POINTS]
        x = x[idx]
        y = y[idx]
    x_t = torch.tensor(x, dtype=torch.float32, device=device) / 16.0 * 2.0 - 1.0
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    dataset = TensorDataset(x_t, y_t)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed or 0))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(IndexedDataset(test_ds), batch_size=batch_size, shuffle=False),
    )


def _build_energy_stack(*, cfg: RuntimeConfig, weights_path: Path, device: torch.device):
    input_dim, hidden_dims, output_dim, layer_shapes = parse_layer_shapes(cfg.dims)
    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=cfg.weight_gains,
        input_gain=cfg.input_gain,
        non_linearity=cfg.non_linearity,
        exponential_diode_param=cfg.exponential_diode_param,
        quadratic_diode_param=cfg.quadratic_diode_param,
        hard_sigmoid_param=cfg.hard_sigmoid_param,
        voltage_amp=cfg.voltage_amp,
        current_amp=cfg.current_amp,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        bias_scale_mode=cfg.bias_scale_mode,
        bias_interaction_type=cfg.bias_interaction_type,
        signed_weights=cfg.signed_weights,
    )
    energy_fn.set_device(device)
    energy_fn.load(weights_path)
    network = Network(energy_fn)
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Expected output layer width 10 or 20 for digits. Provided value: {output_layer.shape!r}.")
    return energy_fn, network, network.free_layers(), cost_fn, output_layer, layer_shapes


def _build_minimizer(cfg: RuntimeConfig, energy_fn, free_layers):
    settings = MinimizerSettings(
        rel_tol=cfg.rel_tol,
        vn_tol=cfg.vn_tol,
        use_polish=cfg.use_polish,
        max_newton_iters=cfg.max_newton_iters,
        z_thresh=cfg.z_thresh,
        exp_clip=cfg.exp_clip,
        experimental_newton_tol=cfg.experimental_newton_tol,
    )
    if cfg.minimizer_impl != "custom":
        raise ValueError(f"Expected minimizer_impl to be 'custom'. Provided value: {cfg.minimizer_impl!r}.")
    return CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=cfg.num_iterations,
        mode="asynchronous",
        non_linearity=cfg.non_linearity,
        quadratic_diode_param=cfg.quadratic_diode_param,
        exponential_diode_param=cfg.exponential_diode_param,
        voltage_amp=energy_fn.voltage_amp,
        current_amp=energy_fn.current_amp,
        iv_data=None,
        iv_data_path=cfg.iv_data_path,
        double_diode_updater=cfg.double_diode_updater,
        adaptive_equilibrium=cfg.adaptive_equilibrium,
        overrelaxation_factor=cfg.overrelaxation_factor,
        single_diode_updater=cfg.single_diode_updater,
        damping=cfg.damping,
        experimental_newton_max_steps=cfg.experimental_newton_max_steps,
        minimizer_settings=settings,
    )

def _layer_sort_key(name: str):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (0, int(digits)) if digits else (1, name)
