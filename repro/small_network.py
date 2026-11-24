"""Simulate a complete, validated hand-specified network configuration."""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.execution import (
    apply_execution_profile,
    seed_from_config,
    validate_execution_relations,
)
from repro.provenance import build_run_receipt, sha256_arrays, sha256_file
from repro.strict_config import load_validated_json, validate_document


PACK_ROOT = Path(__file__).resolve().parents[1]
_MAX_CARTESIAN_SWEEP_CASES = 1_000_000
_DTYPES = {"float32": np.float32, "float64": np.float64}


@dataclass(frozen=True)
class SmallNetworkResult:
    """Settled physical voltages, convergence information, and provenance."""

    hidden_voltages: tuple[np.ndarray, ...]
    output_voltages: np.ndarray
    converged: bool | None
    sweeps: int
    final_max_voltage_change: float
    receipt: dict[str, Any]
    final_max_current_residual: float | None = None


def make_input_voltage_sweep(
    input_size: int,
    *,
    voltage_min: float,
    voltage_max: float,
    num_points: int,
    dtype: str,
) -> np.ndarray:
    """Create input rows for a config; the last physical input changes fastest."""

    size = _validate_positive_integer(input_size, "input_size")
    points = _validate_positive_integer(num_points, "num_points")
    if points < 2:
        raise ValueError(
            "Expected num_points to be at least 2 for an input-voltage sweep. "
            f"Provided value: {num_points!r}."
        )
    minimum = _validate_finite_float(voltage_min, "voltage_min")
    maximum = _validate_finite_float(voltage_max, "voltage_max")
    if maximum <= minimum:
        raise ValueError(
            "Expected voltage_max to be greater than voltage_min. "
            f"Provided values: voltage_min={minimum}, voltage_max={maximum}."
        )
    num_cases = points**size
    if num_cases > _MAX_CARTESIAN_SWEEP_CASES:
        raise ValueError(
            "Input-voltage sweep is too large: "
            f"{points}^{size}={num_cases} cases exceeds the "
            f"{_MAX_CARTESIAN_SWEEP_CASES} case limit. Reduce num_points or "
            "supply explicit network.input_voltages."
        )
    try:
        numpy_dtype = _DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(
            "Expected dtype to be exactly 'float32' or 'float64'. "
            f"Provided value: {dtype!r}."
        ) from exc
    axis = np.linspace(minimum, maximum, points, dtype=numpy_dtype)
    grids = np.meshgrid(*([axis] * size), indexing="ij")
    return np.stack(grids, axis=-1).reshape(num_cases, size)


def load_small_network_config(
    config: Mapping[str, Any] | str | Path,
    *,
    repo_root: Path = PACK_ROOT,
) -> dict[str, Any]:
    """Load or copy one fully expanded small-network v2 document."""

    if isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.is_absolute():
            path = repo_root.expanduser().resolve() / path
        document = load_validated_json(
            path.resolve(),
            "small-network-v2.schema.json",
            repo_root=repo_root,
        )
        from repro.minimizer_factory import simulation_assets

        simulation_assets(document["simulation"], repo_root=repo_root)
        validate_execution_relations(document["execution"])
        _validate_small_network_relations(document)
        return document
    if not isinstance(config, Mapping):
        raise TypeError(
            "Expected config to be a mapping or JSON path. "
            f"Provided value: {type(config).__name__}."
        )
    document = copy.deepcopy(dict(config))
    validate_document(
        document,
        "small-network-v2.schema.json",
        repo_root=repo_root,
    )
    from repro.minimizer_factory import simulation_assets

    simulation_assets(document["simulation"], repo_root=repo_root)
    validate_execution_relations(document["execution"])
    _validate_small_network_relations(document)
    return document


def simulate_small_network(
    config: Mapping[str, Any] | str | Path,
    *,
    repo_root: Path = PACK_ROOT,
    solver: str = "config",
    vol_tol: float | None = None,
    cur_tol: float | None = None,
) -> SmallNetworkResult:
    """Settle a network with its configured minimizer, FastCD, or TorchCD."""

    for name, value in (("vol_tol", vol_tol), ("cur_tol", cur_tol)):
        if value is not None and _validate_finite_float(value, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    root = repo_root.expanduser().resolve()
    source_path: Path | None = None
    source_sha256: str | None = None
    if isinstance(config, (str, Path)):
        source_path = Path(config).expanduser()
        if not source_path.is_absolute():
            source_path = root / source_path
        source_path = source_path.resolve()
        source_sha256 = sha256_file(source_path)
    document = load_small_network_config(config, repo_root=root)
    source_documents: list[dict[str, str]] = []
    if source_path is not None and source_sha256 is not None:
        if sha256_file(source_path) != source_sha256:
            raise RuntimeError(
                "Small-network source changed while it was being loaded; retry "
                "from stable source bytes."
            )
        source_documents.append(
            {
                "owner": "small_network",
                "path": _relative_or_absolute(source_path, root),
                "sha256": source_sha256,
            }
        )
    network_config = document["network"]
    simulation = document["simulation"]
    equilibrium = document["equilibrium"]
    execution = document["execution"]

    state_dtype = network_config["state_dtype"]
    if state_dtype != execution["backend"]["default_dtype"]:
        raise ValueError(
            "Expected network.state_dtype and execution.backend.default_dtype "
            f"to match. Provided values: {state_dtype!r} and "
            f"{execution['backend']['default_dtype']!r}."
        )
    device = apply_execution_profile(execution)
    seed_from_config(network_config["seed"], execution)
    numpy_dtype = _DTYPES[state_dtype] if solver == "config" else np.float64

    family = simulation["nonlinearity"]
    sizes = _validate_layer_sizes(network_config["layer_sizes"], family)
    matrices = _validate_conductances(
        network_config["conductances"],
        sizes,
        dtype=numpy_dtype,
    )
    inputs = _validate_input_voltages(
        network_config["input_voltages"],
        sizes[0],
        dtype=numpy_dtype,
    )

    from repro.minimizer_factory import simulation_assets

    current_circuit = None
    if cur_tol is not None:
        if family == "experimental":
            from repro.iv_data import load_iv_data
            from repro.solvers import LayeredDRN, TableLaw

            curve = simulation_assets(simulation, repo_root=root)["iv_curve"]
            current, voltage = load_iv_data(curve).numpy()
            law = TableLaw(voltage, current, np.zeros(sum(sizes[1:-1]), dtype=int))
            gains = simulation["amplification"]
            current_circuit = LayeredDRN(matrices, law, gains["voltage_factor"], gains["current_factor"])
        else:
            from repro.solvers.torch_cd import shockley_circuit

            current_circuit = shockley_circuit(matrices, simulation)
    if solver == "config":
        free_states, converged, sweeps, final_delta = _solve_configured(
            sizes, matrices, inputs, network_config, simulation, equilibrium, device, root,
            vol_tol, cur_tol, current_circuit,
        )
    else:
        free_states, sweeps, final_delta = _solve_selected_cd(
            solver, matrices, inputs * network_config["input_gain"],
            simulation, equilibrium, device, vol_tol, cur_tol, current_circuit,
        )
        converged = True
    residual = None
    if current_circuit is not None:
        forcing = current_circuit.forcing(inputs * network_config["input_gain"])
        residual = float(np.abs(current_circuit.residual(np.concatenate(free_states, axis=1), forcing)).max())
    receipt = build_run_receipt(
        repo_root=root,
        resolved_config=document,
        execution=execution,
        device=device,
        assets=simulation_assets(simulation, repo_root=root),
        source_documents=source_documents,
        data_fingerprint=sha256_arrays(inputs, *matrices),
        extra={
            "solver": solver,
            "vol_tol": vol_tol,
            "cur_tol": cur_tol,
            "final_max_current_residual": residual,
            "equilibrium_sweeps": sweeps,
            "final_max_voltage_change": final_delta,
            "converged": converged,
        },
    )
    return SmallNetworkResult(
        hidden_voltages=free_states[:-1],
        output_voltages=free_states[-1],
        converged=converged,
        sweeps=sweeps,
        final_max_voltage_change=final_delta,
        receipt=receipt,
        final_max_current_residual=residual,
    )


def _solve_configured(sizes, matrices, inputs, network_config, simulation, equilibrium, device, root,
                      vol_tol, cur_tol, current_circuit):
    energy_fn, network = _build_network(
        sizes=sizes,
        matrices=matrices,
        inputs=inputs,
        input_gain=network_config["input_gain"],
        input_encoding=network_config["input_encoding"],
        bias_method=network_config["bias"]["method"],
        simulation=simulation,
        device=device,
    )
    from repro.minimizer_factory import build_minimizer

    custom_stop = vol_tol is not None or cur_tol is not None
    budget = equilibrium.get("max_sweeps", equilibrium.get("sweeps"))
    settings = equilibrium
    if custom_stop:
        settings = dict(method="fixed_sweeps", sweeps=1, update_order=equilibrium["update_order"])
    minimizer = build_minimizer(
        fn=energy_fn,
        free_layers=network.free_layers(),
        simulation=simulation,
        equilibrium=settings,
        repo_root=root,
    )
    if custom_stop:
        forcing = current_circuit.forcing(inputs * network_config["input_gain"]) if cur_tol is not None else None
        for sweeps in range(1, budget + 1):
            minimizer.compute_equilibrium(resume=sweeps > 1)
            final_delta = minimizer.final_max_voltage_change
            if vol_tol is not None and final_delta > vol_tol:
                continue
            if cur_tol is not None:
                state = np.concatenate([layer.state.detach().cpu().numpy() for layer in network.free_layers()], axis=1)
                if not np.all(np.abs(current_circuit.residual(state, forcing)) <= cur_tol):
                    continue
            break
        else:
            raise RuntimeError(f"Configured minimizer exhausted {budget} sweeps before meeting the requested tolerances.")
    else:
        minimizer.compute_equilibrium()
    _raise_for_non_finite_states(network.free_layers())

    if not custom_stop:
        sweeps = minimizer.iterations_performed
    final_delta = minimizer.final_max_voltage_change
    if custom_stop:
        converged = True
    elif equilibrium["method"] == "voltage_change":
        threshold = (
            equilibrium["relative_tolerance"] * minimizer.final_reference_scale
            + equilibrium["absolute_tolerance"]
        )
        converged = final_delta <= threshold
        if not converged:
            raise RuntimeError(
                "Small-network equilibrium exhausted max_sweeps without satisfying "
                f"the configured voltage-change criterion: delta={final_delta}, "
                f"threshold={threshold}."
            )
    else:
        converged = None

    free_states = tuple(
        layer.state.detach().cpu().numpy().copy() for layer in network.free_layers()
    )
    return free_states, converged, sweeps, final_delta


def _solve_selected_cd(solver, matrices, inputs, simulation, equilibrium, device, vol_tol, cur_tol, current_circuit):
    from repro.solvers import AdaptiveRelaxation, FastCD, TorchCD

    laws = {"single_diode_exponential": "single", "double_diode_exponential": "double"}
    law = laws.get(simulation["nonlinearity"])
    if solver not in ("fast_cd", "torch_cd"):
        raise ValueError(f"Unknown small-network solver: {solver!r}")
    if law is None or device.type != "cpu":
        raise ValueError(f"{solver} requires a CPU Shockley config.")
    weights = [matrix.astype(np.float64, copy=False) for matrix in matrices]
    custom_stop = vol_tol is not None or cur_tol is not None
    if equilibrium["method"] == "fixed_sweeps" and not custom_stop:
        raise ValueError(f"{solver} requires a voltage-change sweep budget or an explicit tolerance.")
    max_sweeps = equilibrium.get("max_sweeps", equilibrium.get("sweeps"))
    if solver == "fast_cd":
        gains = simulation["amplification"]
        cd = FastCD(weights, law, gains["voltage_factor"], gains["current_factor"],
                    physical=simulation["physical"])
    else:
        cd = TorchCD(weights, simulation)

    states = []
    max_count = 0
    max_delta = 0.0
    for index, row in enumerate(inputs):
        if solver == "fast_cd":
            forcing = current_circuit.forcing(row) if cur_tol is not None else None

            def stop(state, change):
                if vol_tol is not None and change > vol_tol:
                    return False
                return cur_tol is None or np.all(np.abs(current_circuit.residual(state, forcing)) <= cur_tol)

            count = cd.solve(row @ weights[0], AdaptiveRelaxation(),
                             rtol=equilibrium.get("relative_tolerance", 0.0),
                             atol=equilibrium.get("absolute_tolerance", 0.0), max_sweeps=max_sweeps,
                             stop=stop if custom_stop else None)
            state, delta = cd.state, cd.final_max_voltage_change
            converged, reason = cd.converged, cd.reason
        else:
            result = cd.solve(row, tolerance=cur_tol if custom_stop else 1e-9,
                              vol_tol=vol_tol, max_sweeps=max_sweeps)
            state, count = result.voltages, result.iterations
            delta = cd.minimizer.final_max_voltage_change if count else 0.0
            converged, reason = result.converged, result.reason
        if not converged:
            raise RuntimeError(f"{solver} failed for input row {index} after {count} sweeps: {reason}")
        states.append(state.copy())
        max_count = max(max_count, count)
        max_delta = max(max_delta, delta)

    boundaries = np.cumsum([matrix.shape[1] for matrix in weights[:-1]])
    return tuple(np.split(np.stack(states), boundaries, axis=1)), max_count, max_delta


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _validate_layer_sizes(
    layer_sizes: Sequence[int], family: str
) -> tuple[int, ...]:
    if isinstance(layer_sizes, (str, bytes)):
        raise ValueError("Expected network.layer_sizes to be an integer array.")
    raw_sizes = tuple(layer_sizes)
    if len(raw_sizes) < 3:
        raise ValueError(
            "Expected network.layer_sizes to contain input, hidden, and output "
            f"node counts. Provided value: {raw_sizes!r}."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
        for value in raw_sizes
    ):
        raise ValueError(
            "Expected every network layer size to be a positive integer. "
            f"Provided value: {raw_sizes!r}."
        )
    sizes = tuple(int(value) for value in raw_sizes)
    if family == "single_diode_exponential" and any(
        size % 2 for size in sizes[1:-1]
    ):
        raise ValueError(
            "Expected every single-Shockley hidden layer size to be even so "
            "forward- and reverse-oriented nodes can be paired. "
            f"Provided value: {sizes[1:-1]!r}."
        )
    return sizes


def _validate_conductances(
    conductances: Sequence[Any],
    sizes: tuple[int, ...],
    *,
    dtype: type[np.floating[Any]],
) -> tuple[np.ndarray, ...]:
    if isinstance(conductances, (str, bytes)):
        raise ValueError("Expected network.conductances to be an array of matrices.")
    raw_matrices = tuple(conductances)
    expected_count = len(sizes) - 1
    if len(raw_matrices) != expected_count:
        raise ValueError(
            f"Expected {expected_count} conductance matrices for layer sizes "
            f"{sizes!r}. Provided count: {len(raw_matrices)}."
        )
    matrices = []
    for index, (value, pre_size, post_size) in enumerate(
        zip(raw_matrices, sizes[:-1], sizes[1:])
    ):
        matrix = _as_finite_real_array(value, f"conductances[{index}]", dtype=dtype)
        expected_shape = (pre_size, post_size)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Expected conductances[{index}] to have shape {expected_shape}. "
                f"Provided shape: {matrix.shape}."
            )
        if np.any(matrix < 0.0):
            raise ValueError(f"Expected conductances[{index}] to be non-negative.")
        matrices.append(matrix)
    for layer_index in range(1, len(sizes)):
        incident = matrices[layer_index - 1].sum(axis=0, dtype=np.float64)
        if layer_index < len(sizes) - 1:
            incident += matrices[layer_index].sum(axis=1, dtype=np.float64)
        isolated = np.flatnonzero(incident <= 0.0)
        if isolated.size:
            raise ValueError(
                "Expected every free node to have positive incident conductance. "
                f"Layer {layer_index} has isolated node indices {isolated.tolist()}."
            )
    return tuple(matrices)


def _validate_input_voltages(
    value: Any,
    input_size: int,
    *,
    dtype: type[np.floating[Any]],
) -> np.ndarray:
    inputs = _as_finite_real_array(value, "input_voltages", dtype=dtype)
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != input_size:
        raise ValueError(
            f"Expected network.input_voltages to have shape (batch, {input_size}). "
            f"Provided shape: {inputs.shape}."
        )
    return np.ascontiguousarray(inputs)


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(
            f"Expected {name} to be a positive integer. Provided value: {value!r}."
        )
    return int(value)


def _validate_finite_float(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"Expected {name} to be a finite real number. Provided value: {value!r}."
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(
            f"Expected {name} to be a finite real number. Provided value: {value!r}."
        )
    return parsed


def _validate_small_network_relations(document: Mapping[str, Any]) -> None:
    network = document["network"]
    execution = document["execution"]
    state_dtype = network["state_dtype"]
    if state_dtype != execution["backend"]["default_dtype"]:
        raise ValueError(
            "Expected network.state_dtype and execution.backend.default_dtype "
            "to match."
        )
    if document["equilibrium"]["initial_state"] != "zeros":
        raise ValueError("Expected equilibrium.initial_state to be 'zeros'.")
    sizes = _validate_layer_sizes(
        network["layer_sizes"], document["simulation"]["nonlinearity"]
    )
    dtype = _DTYPES[state_dtype]
    _validate_conductances(network["conductances"], sizes, dtype=dtype)
    _validate_input_voltages(network["input_voltages"], sizes[0], dtype=dtype)


def _as_finite_real_array(
    value: Any,
    name: str,
    *,
    dtype: type[np.floating[Any]],
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(
            f"Expected {name} to contain real numeric values. "
            f"Provided dtype: {array.dtype}."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(dtype, copy=True)
    if not np.isfinite(converted).all():
        raise ValueError(f"Expected {name} to contain only finite values.")
    return converted


def _build_network(
    *,
    sizes: tuple[int, ...],
    matrices: tuple[np.ndarray, ...],
    inputs: np.ndarray,
    input_gain: float,
    input_encoding: str,
    bias_method: str,
    simulation: Mapping[str, Any],
    device: torch.device,
):
    _ensure_vendor_path(PACK_ROOT)
    from model.function.network import Network
    from model.resistive.network import DeepResistiveEnergy
    from model.variable.parameter import Bias, DenseWeight

    if bias_method != "none":
        raise ValueError(
            "Expected network.bias.method to be 'none'. "
            f"Provided value: {bias_method!r}."
        )
    if input_encoding != "direct":
        raise ValueError(
            "Expected network.input_encoding to be 'direct'. "
            f"Provided value: {input_encoding!r}."
        )
    family = simulation["nonlinearity"]
    physical = simulation["physical"]
    exponential = {}
    if family in {"single_diode_exponential", "double_diode_exponential"}:
        exponential = {
            "I_s": physical["saturation_current"],
            "V_t": physical["thermal_voltage"],
            "V_off": physical["offset_voltage"],
        }
    maximum = max(1.0, *(float(matrix.max()) for matrix in matrices))
    energy_fn = DeepResistiveEnergy(
        layer_shapes=[(size,) for size in sizes],
        weight_gains=[1.0] * len(matrices),
        input_gain=input_gain,
        non_linearity=family,
        exponential_diode_param=exponential,
        quadratic_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=simulation["amplification"]["voltage_factor"],
        current_amp=simulation["amplification"]["current_factor"],
        weight_min=0.0,
        weight_max=maximum,
        weight_init_mode="kaiming_uniform",
        bias_scale_mode="legacy",
        bias_interaction_type="linear",
        bias_enabled=False,
        bias_initial_value=None,
        bias_minimum=None,
        bias_maximum=None,
        signed_weights=False,
    )
    energy_fn.set_device(device)
    dense_weights = [
        parameter for parameter in energy_fn.params() if isinstance(parameter, DenseWeight)
    ]
    if len(dense_weights) != len(matrices):
        raise RuntimeError(
            "Constructed network did not expose the configured number of dense "
            f"conductance matrices: expected {len(matrices)}, got {len(dense_weights)}."
        )
    for parameter, matrix in zip(dense_weights, matrices):
        parameter.state = torch.from_numpy(matrix.copy()).to(device)
    for parameter in energy_fn.params():
        if isinstance(parameter, Bias):
            parameter.state = torch.zeros_like(parameter.state)
    network = Network(energy_fn, input_mode="direct")
    network.layers()[0].state = (
        input_gain * torch.from_numpy(inputs.copy()).to(device)
    )
    for layer in network.free_layers():
        layer.init_state(inputs.shape[0], device)
    return energy_fn, network


def _raise_for_non_finite_states(free_layers: Sequence[Any]) -> None:
    for index, layer in enumerate(free_layers, start=1):
        if not torch.isfinite(layer.state).all():
            raise FloatingPointError(
                "Small-network equilibrium produced non-finite voltages in "
                f"layer {index}."
            )


def _ensure_vendor_path(root: Path) -> None:
    vendor = str(root / "repro" / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


__all__ = [
    "SmallNetworkResult",
    "load_small_network_config",
    "make_input_voltage_sweep",
    "simulate_small_network",
]
