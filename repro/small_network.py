"""Simulate hand-specified dense nonlinear resistive networks."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from repro.iv_data import load_iv_data
from repro.runner import _canonical_non_linearity, _ensure_vendor_path


PACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IV_DATA_PATH = "data/assets/experimental_curve_voff_0.8_200_points.npz"
DEFAULT_SHOCKLEY_PARAMETERS: Mapping[str, float] = MappingProxyType(
    {
        "I_s": 1e-6,
        "V_t": 0.05,
        "V_off": 0.8,
    }
)

_SIMULATOR_PROFILES = {
    "single_diode_exponential": "configs/simulator/default_single_shockley.json",
    "double_diode_exponential": "configs/simulator/default_double_shockley.json",
    "experimental": "configs/simulator/default_pwl.json",
}
_SHOCKLEY_KEYS = frozenset(DEFAULT_SHOCKLEY_PARAMETERS)


@dataclass(frozen=True)
class SmallNetworkResult:
    """Settled physical voltages and convergence information."""

    hidden_voltages: tuple[np.ndarray, ...]
    output_voltages: np.ndarray
    converged: bool
    sweeps: int
    final_max_voltage_change: float


def simulate_small_network(
    *,
    layer_sizes: Sequence[int],
    conductances: Sequence[Any],
    input_voltages: Any,
    non_linearity: str,
    shockley_parameters: Mapping[str, Real] = DEFAULT_SHOCKLEY_PARAMETERS,
    iv_data_path: str | Path = DEFAULT_IV_DATA_PATH,
    adaptive_equilibrium: bool = True,
    max_sweeps: int = 128,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-6,
) -> SmallNetworkResult:
    """Settle a dense physical network for one or more input-voltage sets.

    ``layer_sizes`` counts physical nodes, and each conductance matrix connects
    one adjacent pair of layers with shape ``(pre_nodes, post_nodes)``. Input
    voltages are applied directly: this helper does not create positive and
    negative input channels.

    Hidden layers use the selected grounded nonlinearity and the output layer
    is linear. For the single-Shockley model, the first half of each hidden
    layer is forward-oriented and the second half is reverse-oriented.
    """

    family = _canonical_non_linearity(non_linearity)
    sizes = _validate_layer_sizes(layer_sizes, family)
    matrices = _validate_conductances(conductances, sizes)
    inputs = _validate_input_voltages(input_voltages, sizes[0])
    shockley = _validate_shockley_parameters(shockley_parameters)
    adaptive = _validate_adaptive_equilibrium(adaptive_equilibrium)
    sweep_limit = _validate_positive_integer(max_sweeps, "max_sweeps")
    rel_tol = _validate_positive_float(relative_tolerance, "relative_tolerance")
    abs_tol = _validate_positive_float(absolute_tolerance, "absolute_tolerance")

    profile = _load_simulator_profile(family)
    energy_fn, network = _build_network(
        sizes=sizes,
        matrices=matrices,
        inputs=inputs,
        family=family,
        shockley=shockley,
        profile=profile,
    )
    minimizer = _build_minimizer(
        energy_fn=energy_fn,
        free_layers=network.free_layers(),
        family=family,
        shockley=shockley,
        profile=profile,
        iv_data_path=iv_data_path,
        relative_tolerance=rel_tol,
        absolute_tolerance=abs_tol,
    )

    converged = False
    final_delta = math.inf
    sweeps = 0
    for sweeps in range(1, sweep_limit + 1):
        previous = [layer.state.detach().clone() for layer in network.free_layers()]
        minimizer.compute_equilibrium()
        _raise_for_non_finite_states(network.free_layers(), sweep=sweeps)

        final_delta = max(
            float((layer.state - old_state).abs().max().item())
            for layer, old_state in zip(network.free_layers(), previous)
        )
        previous_scale = max(
            float(old_state.abs().max().item()) for old_state in previous
        )
        converged = final_delta <= rel_tol * previous_scale + abs_tol
        if adaptive and converged:
            break

    if not converged:
        warnings.warn(
            "Small-network equilibrium did not satisfy the voltage-change "
            f"criterion after {sweeps} sweeps; returning the latest voltages.",
            RuntimeWarning,
            stacklevel=2,
        )

    free_states = [
        layer.state.detach().cpu().numpy().copy() for layer in network.free_layers()
    ]
    return SmallNetworkResult(
        hidden_voltages=tuple(free_states[:-1]),
        output_voltages=free_states[-1],
        converged=converged,
        sweeps=sweeps,
        final_max_voltage_change=final_delta,
    )


def _validate_layer_sizes(
    layer_sizes: Sequence[int], family: str
) -> tuple[int, ...]:
    if isinstance(layer_sizes, (str, bytes)):
        raise ValueError(
            "Expected layer_sizes to contain input, hidden, and output node counts. "
            f"Provided value: {layer_sizes!r}."
        )
    try:
        raw_sizes = tuple(layer_sizes)
    except TypeError as exc:
        raise ValueError(
            "Expected layer_sizes to contain input, hidden, and output node counts. "
            f"Provided value: {layer_sizes!r}."
        ) from exc
    if len(raw_sizes) < 3:
        raise ValueError(
            "Expected layer_sizes to contain at least input, hidden, and output "
            f"node counts. Provided value: {raw_sizes!r}."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
        for value in raw_sizes
    ):
        raise ValueError(
            "Expected every layer size to be a positive integer. "
            f"Provided value: {raw_sizes!r}."
        )
    sizes = tuple(int(value) for value in raw_sizes)
    hidden_sizes = sizes[1:-1]
    if family == "single_diode_exponential" and any(
        size % 2 for size in hidden_sizes
    ):
        raise ValueError(
            "Expected every single-Shockley hidden layer size to be even so "
            "forward- and reverse-oriented nodes can be paired. "
            f"Provided value: {hidden_sizes!r}."
        )
    return sizes


def _validate_conductances(
    conductances: Sequence[Any], sizes: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    if isinstance(conductances, (str, bytes)):
        raise ValueError("Expected conductances to be a sequence of matrices.")
    try:
        raw_matrices = tuple(conductances)
    except TypeError as exc:
        raise ValueError("Expected conductances to be a sequence of matrices.") from exc
    expected_count = len(sizes) - 1
    if len(raw_matrices) != expected_count:
        raise ValueError(
            f"Expected {expected_count} conductance matrices for layer_sizes "
            f"{sizes!r}. Provided count: {len(raw_matrices)}."
        )

    matrices = []
    for index, (value, pre_size, post_size) in enumerate(
        zip(raw_matrices, sizes[:-1], sizes[1:])
    ):
        matrix = _as_finite_real_array(value, f"conductances[{index}]")
        expected_shape = (pre_size, post_size)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Expected conductances[{index}] to have shape {expected_shape}. "
                f"Provided shape: {matrix.shape}."
            )
        if np.any(matrix < 0.0):
            raise ValueError(
                f"Expected conductances[{index}] to be non-negative."
            )
        matrices.append(matrix)

    for layer_index in range(1, len(sizes)):
        incident = matrices[layer_index - 1].sum(axis=0, dtype=np.float64)
        if layer_index < len(sizes) - 1:
            incident = incident + matrices[layer_index].sum(
                axis=1, dtype=np.float64
            )
        isolated = np.flatnonzero(incident <= 0.0)
        if isolated.size:
            raise ValueError(
                f"Expected every free node to have positive incident conductance. "
                f"Layer {layer_index} has isolated node indices {isolated.tolist()}."
            )
    return tuple(matrices)


def _validate_input_voltages(value: Any, input_size: int) -> np.ndarray:
    inputs = _as_finite_real_array(value, "input_voltages")
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    elif inputs.ndim != 2:
        raise ValueError(
            "Expected input_voltages to be one vector or a two-dimensional batch. "
            f"Provided shape: {inputs.shape}."
        )
    if inputs.shape[0] < 1 or inputs.shape[1] != input_size:
        raise ValueError(
            f"Expected input_voltages to have shape ({input_size},) or "
            f"(batch, {input_size}). Provided shape: {inputs.shape}."
        )
    return np.ascontiguousarray(inputs)


def _validate_shockley_parameters(
    value: Mapping[str, Real],
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected shockley_parameters to be a mapping containing I_s, V_t, "
            f"and V_off. Provided value: {value!r}."
        )
    keys = frozenset(value)
    if keys != _SHOCKLEY_KEYS:
        missing = sorted(_SHOCKLEY_KEYS - keys)
        unknown = sorted(keys - _SHOCKLEY_KEYS)
        raise ValueError(
            "Expected shockley_parameters to contain exactly I_s, V_t, and V_off. "
            f"Missing keys: {missing}; unknown keys: {unknown}."
        )
    parsed = {
        name: _validate_finite_float(raw_value, f"shockley_parameters[{name!r}]")
        for name, raw_value in value.items()
    }
    if parsed["I_s"] <= 0.0 or parsed["V_t"] <= 0.0:
        raise ValueError(
            "Expected shockley_parameters I_s and V_t to be positive. "
            f"Provided value: {parsed!r}."
        )
    return parsed


def _validate_adaptive_equilibrium(value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            "Expected adaptive_equilibrium to be a boolean. "
            f"Provided value: {value!r}."
        )
    return value


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(
            f"Expected {name} to be a positive integer. Provided value: {value!r}."
        )
    return int(value)


def _validate_positive_float(value: Real, name: str) -> float:
    parsed = _validate_finite_float(value, name)
    if parsed <= 0.0:
        raise ValueError(
            f"Expected {name} to be finite and positive. Provided value: {value!r}."
        )
    return parsed


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


def _as_finite_real_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected {name} to contain real numeric values."
        ) from exc
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(
            f"Expected {name} to contain real numeric values. "
            f"Provided dtype: {array.dtype}."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(np.float32, copy=True)
    if not np.isfinite(converted).all():
        raise ValueError(f"Expected {name} to contain only finite values.")
    return converted


def _load_simulator_profile(family: str) -> dict[str, Any]:
    path = PACK_ROOT / _SIMULATOR_PROFILES[family]
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("non_linearity") != family:
        raise RuntimeError(
            f"Bundled simulator profile {path} does not describe {family!r}."
        )
    return profile


def _build_network(
    *,
    sizes: tuple[int, ...],
    matrices: tuple[np.ndarray, ...],
    inputs: np.ndarray,
    family: str,
    shockley: dict[str, float],
    profile: dict[str, Any],
):
    _ensure_vendor_path(PACK_ROOT)
    from model.function.network import Network
    from model.resistive.network import DeepResistiveEnergy
    from model.variable.parameter import Bias, DenseWeight

    max_conductance = max(1.0, *(float(matrix.max()) for matrix in matrices))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        energy_fn = DeepResistiveEnergy(
            layer_shapes=[(size,) for size in sizes],
            weight_gains=[1.0] * len(matrices),
            input_gain=1.0,
            non_linearity=family,
            exponential_diode_param=(
                shockley
                if family
                in {"single_diode_exponential", "double_diode_exponential"}
                else {}
            ),
            quadratic_diode_param={},
            hard_sigmoid_param={},
            voltage_amp=float(profile["voltage_amp"]),
            current_amp=float(profile["current_amp"]),
            weight_min=0.0,
            weight_max=max_conductance,
        )
    energy_fn.set_device(torch.device("cpu"))

    dense_weights = [
        parameter
        for parameter in energy_fn.params()
        if isinstance(parameter, DenseWeight)
    ]
    if len(dense_weights) != len(matrices):
        raise RuntimeError(
            "Constructed network did not expose the expected number of dense "
            f"conductance matrices: expected {len(matrices)}, got {len(dense_weights)}."
        )
    for parameter, matrix in zip(dense_weights, matrices):
        parameter.state = torch.from_numpy(matrix.copy())
    for parameter in energy_fn.params():
        if isinstance(parameter, Bias):
            parameter.state = torch.zeros_like(parameter.state)

    network = Network(energy_fn)
    network.layers()[0].state = torch.from_numpy(inputs.copy())
    device = torch.device("cpu")
    for layer in network.free_layers():
        layer.init_state(inputs.shape[0], device)
    return energy_fn, network


def _build_minimizer(
    *,
    energy_fn,
    free_layers,
    family: str,
    shockley: dict[str, float],
    profile: dict[str, Any],
    iv_data_path: str | Path,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    _ensure_vendor_path(PACK_ROOT)
    from model.resistive.minimizer import MinimizerSettings, QuadraticMinimizer

    iv_data = None
    if family == "experimental":
        if not isinstance(iv_data_path, (str, Path)):
            raise ValueError(
                "Expected iv_data_path to be a string or Path for the PWL "
                f"nonlinearity. Provided value: {iv_data_path!r}."
            )
        resolved_iv_path = Path(iv_data_path).expanduser()
        if not resolved_iv_path.is_absolute():
            resolved_iv_path = PACK_ROOT / resolved_iv_path
        iv_data = load_iv_data(
            resolved_iv_path,
            use_environment_override=False,
        )

    settings = MinimizerSettings(
        rel_tol=relative_tolerance,
        vn_tol=absolute_tolerance,
        use_polish=bool(profile["use_polish"]),
        max_newton_iters=int(profile["max_newton_iters"]),
        z_thresh=float(profile["z_thresh"]),
        exp_clip=float(profile["exp_clip"]),
        experimental_newton_tol=float(profile["experimental_newton_tol"]),
    )
    exponential_parameters = (
        shockley
        if family in {"single_diode_exponential", "double_diode_exponential"}
        else {}
    )
    return QuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=1,
        mode="asynchronous",
        non_linearity=family,
        quadratic_diode_param={},
        exponential_diode_param=exponential_parameters,
        voltage_amp=energy_fn.voltage_amp,
        current_amp=energy_fn.current_amp,
        iv_data=iv_data,
        double_diode_updater=profile["double_diode_updater"],
        adaptive_equilibrium=False,
        overrelaxation_factor=float(profile["overrelaxation_factor"]),
        single_diode_updater=profile["single_diode_updater"],
        damping=float(profile["damping"]),
        experimental_newton_max_steps=int(profile["experimental_newton_max_steps"]),
        minimizer_settings=settings,
    )


def _raise_for_non_finite_states(free_layers, *, sweep: int) -> None:
    for index, layer in enumerate(free_layers, start=1):
        if not torch.isfinite(layer.state).all():
            raise FloatingPointError(
                "Small-network equilibrium produced non-finite voltages in "
                f"layer {index} during sweep {sweep}."
            )


__all__ = [
    "DEFAULT_SHOCKLEY_PARAMETERS",
    "SmallNetworkResult",
    "simulate_small_network",
]
