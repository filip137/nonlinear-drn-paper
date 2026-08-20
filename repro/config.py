from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    dims: list[int]
    non_linearity: str
    weight_gains: list[float]
    weight_min: float
    weight_max: float
    input_gain: float
    voltage_amp: float
    current_amp: float
    batch_size: int
    num_iterations: int
    seed: int | None
    bias_scale_mode: str
    bias_interaction_type: str
    signed_weights: bool
    quadratic_diode_param: dict[str, Any]
    exponential_diode_param: dict[str, Any]
    hard_sigmoid_param: dict[str, Any]
    double_diode_updater: str | None
    single_diode_updater: str | None
    adaptive_equilibrium: bool
    rel_tol: float
    vn_tol: float
    use_polish: bool
    max_newton_iters: int
    z_thresh: float
    exp_clip: float
    minimizer_impl: str
    damping: float
    overrelaxation_factor: float
    experimental_newton_max_steps: int
    experimental_newton_tol: float
    iv_data_path: str | None


def load_runtime_config(path: Path, *, pack_root: Path, num_iterations: int | None = None) -> RuntimeConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    non_linearity = _required_str(data, "non_linearity")
    quadratic = _required_dict(data, "quadratic_diode_param")
    exponential = _required_dict(data, "exponential_diode_param")
    hard_sigmoid = _required_dict(data, "hard_sigmoid_param")
    if non_linearity == "double_diode_exponential":
        _require_keys(exponential, ("I_s", "V_t", "V_off"), "exponential_diode_param")
    if non_linearity == "single_diode_exponential":
        _require_keys(exponential, ("I_s", "V_t", "V_off"), "exponential_diode_param")
    if non_linearity in ("hard_sigmoid", "double_diode"):
        _require_keys(hard_sigmoid, ("g_on", "g_off", "v_min", "v_max"), "hard_sigmoid_param")

    dims = data.get("dims")
    if not isinstance(dims, list) or len(dims) < 2:
        raise ValueError(f"Expected config 'dims' to be a list with at least two entries. Provided value: {dims!r}.")

    iv_path = data.get("iv_data_path") or data.get("LABS_IV_CURVE_PATH")
    if iv_path is not None:
        candidate = Path(iv_path)
        if not candidate.is_absolute():
            candidate = pack_root / candidate
        iv_path = str(candidate)

    return RuntimeConfig(
        dims=[int(item) for item in dims],
        non_linearity=non_linearity,
        weight_gains=[float(item) for item in _required_list(data, "weight_gains")],
        weight_min=float(_required(data, "weight_min")),
        weight_max=float(_required(data, "weight_max")),
        input_gain=float(_required(data, "input_gain")),
        voltage_amp=float(_required(data, "voltage_amp")),
        current_amp=float(_required(data, "current_amp")),
        batch_size=int(data.get("batch_size", 1)),
        num_iterations=int(num_iterations if num_iterations is not None else _required(data, "num_iterations")),
        seed=int(data["seed"]) if data.get("seed") is not None else None,
        bias_scale_mode=str(data.get("bias_scale_mode", "legacy")),
        bias_interaction_type=str(data.get("bias_interaction_type", "linear")),
        signed_weights=bool(data.get("signed_weights", False)),
        quadratic_diode_param=quadratic,
        exponential_diode_param=exponential,
        hard_sigmoid_param=hard_sigmoid,
        double_diode_updater=data.get("double_diode_updater"),
        single_diode_updater=data.get("single_diode_updater"),
        adaptive_equilibrium=bool(_required(data, "adaptive_equilibrium")),
        rel_tol=float(data.get("rel_tol", 1e-5)),
        vn_tol=float(data.get("vn_tol", 1e-6)),
        use_polish=bool(data.get("use_polish", True)),
        max_newton_iters=int(data.get("max_newton_iters", 32)),
        z_thresh=float(data.get("z_thresh", 1e10)),
        exp_clip=float(_required(data, "exp_clip")) if non_linearity in ("single_diode_exponential", "double_diode_exponential") else float(data.get("exp_clip", 100000.0)),
        minimizer_impl=str(_required(data, "minimizer_impl")),
        damping=float(data.get("damping", 0.5)),
        overrelaxation_factor=float(data.get("overrelaxation_factor", 1.1)),
        experimental_newton_max_steps=int(data.get("experimental_newton_max_steps", 100)),
        experimental_newton_tol=float(data.get("experimental_newton_tol", 1e-5)),
        iv_data_path=iv_path,
    )


def parse_layer_shapes(dims: list[int]) -> tuple[int, list[int], int, list[tuple[int, ...]]]:
    if dims[0] % 2 != 0:
        raise ValueError(f"Expected input layer size to be divisible by 2. Provided value: {dims[0]!r}.")
    shapes = [(int(dim),) for dim in dims]
    return dims[0] // 2, [int(item) for item in dims[1:-1]], int(dims[-1]), shapes


def _required(data: dict[str, Any], name: str) -> Any:
    if name not in data or data[name] is None:
        raise ValueError(f"Expected config field '{name}' to be present. Provided value: {data.get(name)!r}.")
    return data[name]


def _required_str(data: dict[str, Any], name: str) -> str:
    value = _required(data, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected config field '{name}' to be a non-empty string. Provided value: {value!r}.")
    return value


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
        raise ValueError(f"Expected config '{label}' to include keys {keys}. Provided value missing: {missing}.")
