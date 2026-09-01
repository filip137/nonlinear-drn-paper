"""Canonical, strict runtime configuration models.

The public loaders in this module never coerce values and never provide a
scientific default. JSON Schema validation happens before a runtime object is
constructed.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro.strict_config import load_validated_json, validate_document
from repro.execution import validate_execution_relations


@dataclass(frozen=True)
class RuntimeConfig:
    """A fully validated replay configuration."""

    document: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_runtime_relations(self.document)

    @property
    def data(self) -> dict[str, Any]:
        return self.document["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.document["model"]

    @property
    def simulation(self) -> dict[str, Any]:
        return self.document["simulation"]

    @property
    def equilibrium(self) -> dict[str, Any]:
        return self.document["equilibrium"]

    @property
    def dims(self) -> list[int]:
        return list(self.model["layer_widths"])

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
    def input_gain(self) -> float:
        return self.model["input_gain"]

    @property
    def voltage_amp(self) -> float:
        return self.simulation["amplification"]["voltage_factor"]

    @property
    def current_amp(self) -> float:
        return self.simulation["amplification"]["current_factor"]

    @property
    def batch_size(self) -> int:
        return self.data["loader"]["batch_size"]

    @property
    def num_iterations(self) -> int:
        if self.equilibrium["method"] == "fixed_sweeps":
            return self.equilibrium["sweeps"]
        return self.equilibrium["max_sweeps"]

    @property
    def seed(self) -> int:
        return self.data["seed"]

    @property
    def bias_scale_mode(self) -> str:
        return self.model["bias"]["scale_mode"]

    @property
    def bias_interaction_type(self) -> str:
        return self.model["bias"]["interaction"]

    @property
    def signed_weights(self) -> bool:
        return self.model["signed_weights"]

    @property
    def exponential_diode_param(self) -> dict[str, float]:
        if self.non_linearity not in {
            "single_diode_exponential",
            "double_diode_exponential",
        }:
            return {}
        physical = self.simulation["physical"]
        return {
            "I_s": physical["saturation_current"],
            "V_t": physical["thermal_voltage"],
            "V_off": physical["offset_voltage"],
        }

    @property
    def adaptive_equilibrium(self) -> bool:
        return self.equilibrium["method"] == "voltage_change"

    @property
    def mode(self) -> str:
        return self.equilibrium["update_order"]


def load_runtime_config(
    path: Path,
    *,
    pack_root: Path,
    equilibrium_override: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Load one v2 replay config and apply a schema-checked job sweep override."""

    document = load_validated_json(
        path,
        "replay-v2.schema.json",
        repo_root=pack_root,
    )
    if equilibrium_override is not None:
        document = _apply_equilibrium_override(document, equilibrium_override)
        validate_document(document, "replay-v2.schema.json", repo_root=pack_root)
    config = RuntimeConfig(document=document)
    from repro.minimizer_factory import simulation_assets

    simulation_assets(config.simulation, repo_root=pack_root)
    return config


def _apply_equilibrium_override(
    document: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(override, Mapping):
        raise ValueError(
            "Expected manifest overrides.equilibrium to be an object. "
            f"Provided value: {override!r}."
        )
    equilibrium = document["equilibrium"]
    method = equilibrium["method"]
    expected_key = "sweeps" if method == "fixed_sweeps" else "max_sweeps"
    if set(override) != {expected_key}:
        raise ValueError(
            "Expected the manifest equilibrium override to replace only the "
            f"active {expected_key!r} field for method {method!r}. "
            f"Provided keys: {sorted(override)}."
        )
    value = override[expected_key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"Expected manifest equilibrium override {expected_key!r} to be a "
            f"positive integer. Provided value: {value!r}."
        )
    expanded = copy.deepcopy(dict(document))
    expanded["equilibrium"][expected_key] = value
    return expanded


def parse_layer_shapes(
    dims: list[int],
) -> tuple[int, list[int], int, list[tuple[int, ...]]]:
    if dims[0] % 2 != 0:
        raise ValueError(
            "Expected input layer size to be divisible by 2. "
            f"Provided value: {dims[0]!r}."
        )
    shapes = [(item,) for item in dims]
    return dims[0] // 2, dims[1:-1], dims[-1], shapes


def _validate_runtime_relations(document: Mapping[str, Any]) -> None:
    model = document["model"]
    simulation = document["simulation"]
    widths = model["layer_widths"]
    if len(model["weight_gains"]) != len(widths) - 1:
        raise ValueError(
            "Expected one model.weight_gains value per adjacent replay layer. "
            f"Provided widths={widths!r}, gains={model['weight_gains']!r}."
        )
    bounds = model["weight_bounds"]
    if bounds["minimum"] > bounds["maximum"]:
        raise ValueError(
            "Expected model.weight_bounds.minimum not to exceed maximum."
        )
    if widths[0] != 128:
        raise ValueError(
            "Expected sklearn Digits replay model.layer_widths[0] to be 128 "
            "for the explicit signed-pair encoding. "
            f"duplication. Provided value: {widths[0]!r}."
        )
    output = model["output"]
    if output["classes"] != 10:
        raise ValueError(
            "Expected sklearn Digits replay model.output.classes to be 10. "
            f"Provided value: {output['classes']!r}."
        )
    expected_output = (
        output["classes"]
        if output["encoding"] == "single_ended"
        else 2 * output["classes"]
    )
    if widths[-1] != expected_output:
        raise ValueError(
            f"Expected replay output width {expected_output} for {output!r}; "
            f"provided {widths[-1]}."
        )
    if simulation["nonlinearity"] == "single_diode_exponential" and any(
        width % 2 for width in widths[1:-1]
    ):
        raise ValueError(
            "Expected every single-Shockley replay hidden width to be even. "
            f"Provided value: {widths[1:-1]!r}."
        )
    if document["equilibrium"]["initial_state"] != "zeros":
        raise ValueError("Expected replay equilibrium.initial_state to be 'zeros'.")
    execution = document.get("execution")
    if execution is not None:
        validate_execution_relations(execution)
    if execution is not None and (
        model["state_dtype"] != execution["backend"]["default_dtype"]
    ):
        raise ValueError(
            "Expected replay model.state_dtype to match "
            "execution.backend.default_dtype."
        )


__all__ = ["RuntimeConfig", "load_runtime_config", "parse_layer_shapes"]
