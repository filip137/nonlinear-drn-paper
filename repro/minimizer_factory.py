"""Construct the vendored minimizer from one canonical simulation block."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from repro.iv_data import load_iv_data


_SHOCKLEY_FAMILIES = {
    "single_diode_exponential",
    "double_diode_exponential",
}


def build_minimizer(
    *,
    fn: Any,
    free_layers: Sequence[Any],
    simulation: Mapping[str, Any],
    equilibrium: Mapping[str, Any],
    repo_root: Path,
) -> Any:
    """Translate validated v2 config vocabulary at the single model boundary."""

    from model.resistive.minimizer import MinimizerSettings, QuadraticMinimizer

    family = simulation["nonlinearity"]
    physical = simulation["physical"]
    amplification = simulation["amplification"]
    updater = simulation["updater"]

    adaptive, sweeps, relative_tolerance, absolute_tolerance = _equilibrium_values(
        equilibrium
    )
    updater_values = _updater_values(family, updater, repo_root=repo_root)

    settings = MinimizerSettings(
        rel_tol=relative_tolerance,
        vn_tol=absolute_tolerance,
        use_polish=updater_values["use_polish"],
        max_newton_iters=updater_values["max_newton_iters"],
        z_thresh=updater_values["z_thresh"],
        exp_clip=updater_values["exp_clip"],
        experimental_newton_tol=updater_values["pwl_voltage_tolerance"],
        b_clamp=updater_values["linear_coefficient_clamp"],
        pwl_extrapolation=updater_values["pwl_extrapolation"],
        pwl_nonconvergence_policy=updater_values["pwl_nonconvergence_policy"],
        lambertw_backend=updater_values["lambertw_backend"],
        lambertw_asymptotic_terms=updater_values["asymptotic_terms"],
        single_diode_min_a=updater_values["single_quadratic_min"],
        single_diode_polish_abs_tol=updater_values["single_polish_abs_tol"],
        single_diode_polish_rel_tol=updater_values["single_polish_rel_tol"],
        double_diode_polish_residual_tol=updater_values[
            "double_polish_residual_tol"
        ],
    )

    exponential = {}
    if family in _SHOCKLEY_FAMILIES:
        exponential = {
            "I_s": physical["saturation_current"],
            "V_t": physical["thermal_voltage"],
            "V_off": physical["offset_voltage"],
        }

    return QuadraticMinimizer(
        fn=fn,
        free_layers=free_layers,
        num_iterations=sweeps,
        mode=equilibrium["update_order"],
        non_linearity=family,
        quadratic_diode_param={},
        exponential_diode_param=exponential,
        hard_sigmoid_param={},
        voltage_amp=amplification["voltage_factor"],
        current_amp=amplification["current_factor"],
        minimizer_settings=settings,
        iv_data=updater_values["iv_data"],
        double_diode_updater=updater_values["double_selector"],
        adaptive_equilibrium=adaptive,
        overrelaxation_factor=updater["relaxation"],
        single_diode_updater=updater_values["single_selector"],
        damping=updater_values["pwl_damping"],
        experimental_newton_max_steps=updater_values["pwl_max_steps"],
    )


def simulation_assets(
    simulation: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Path]:
    if simulation["nonlinearity"] != "experimental":
        return {}
    curve = simulation["updater"]["curve"]
    path = _repo_file(curve, repo_root=repo_root)
    # Validate semantic NPZ contents before any dataset or numerical model is
    # constructed. Archived configs may also pin the bytes with a legacy hash.
    load_iv_data(path, expected_sha256=_legacy_curve_sha256(curve))
    return {"iv_curve": path}


def _equilibrium_values(
    equilibrium: Mapping[str, Any],
) -> tuple[bool, int, float | None, float | None]:
    method = equilibrium["method"]
    if method == "fixed_sweeps":
        return False, equilibrium["sweeps"], None, None
    if method == "voltage_change":
        return (
            True,
            equilibrium["max_sweeps"],
            equilibrium["relative_tolerance"],
            equilibrium["absolute_tolerance"],
        )
    raise ValueError(
        "Expected equilibrium.method to be 'fixed_sweeps' or 'voltage_change'. "
        f"Provided value: {method!r}."
    )


def _updater_values(
    family: str,
    updater: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    # Inactive settings are represented by None, never by invented numerical
    # values. QuadraticMinimizer validates the exact fields active for a family.
    inactive = {
        "use_polish": None,
        "max_newton_iters": None,
        "z_thresh": None,
        "exp_clip": None,
        "pwl_voltage_tolerance": None,
        "linear_coefficient_clamp": None,
        "pwl_extrapolation": None,
        "pwl_nonconvergence_policy": None,
        "lambertw_backend": None,
        "asymptotic_terms": None,
        "single_quadratic_min": None,
        "single_polish_abs_tol": None,
        "single_polish_rel_tol": None,
        "double_polish_residual_tol": None,
        "iv_data": None,
        "double_selector": None,
        "single_selector": None,
        "pwl_damping": None,
        "pwl_max_steps": None,
    }

    if family in _SHOCKLEY_FAMILIES:
        if updater["method"] != "lambert_w_v1":
            raise ValueError(
                "Expected a Shockley simulation updater method of 'lambert_w_v1'. "
                f"Provided value: {updater['method']!r}."
            )
        polish = updater["polish"]
        values = {
            **inactive,
            "use_polish": polish is not False,
            "max_newton_iters": None if polish is False else polish["max_steps"],
            "z_thresh": updater["asymptotic_threshold"],
            "exp_clip": updater["exponent_clip"],
            "linear_coefficient_clamp": updater["linear_coefficient_clamp"],
            "lambertw_backend": updater["backend"],
            "asymptotic_terms": updater["asymptotic_terms"],
        }
        # Relaxation is an exact configured algorithm choice.  Treating a value
        # merely close to one as one would silently discard a valid setting.
        relaxed = updater["relaxation"] != 1.0
        if family == "single_diode_exponential":
            if updater["dtype"] != "float64":
                raise ValueError(
                    "Expected single-Shockley updater dtype to be 'float64'. "
                    f"Provided value: {updater['dtype']!r}."
                )
            values["single_quadratic_min"] = updater["quadratic_coefficient_min"]
            values["single_selector"] = "overrelaxed" if relaxed else "custom"
            if polish is not False:
                values["single_polish_abs_tol"] = polish["absolute_tolerance"]
                values["single_polish_rel_tol"] = polish["relative_tolerance"]
        else:
            dtype = updater["dtype"]
            values["double_selector"] = (
                f"{dtype}_overrelaxed" if relaxed else dtype
            )
            if polish is not False:
                residual = polish["residual_tolerance"]
                if residual["rule"] != "batch_scaled_absolute":
                    raise ValueError(
                        "Expected double-Shockley polish residual rule to be "
                        f"'batch_scaled_absolute'. Provided value: {residual['rule']!r}."
                    )
                values["double_polish_residual_tol"] = residual["coefficient"]
        return values

    if family == "experimental":
        if updater["method"] != "piecewise_linear_newton_v1":
            raise ValueError(
                "Expected measured/PWL updater method to be "
                f"'piecewise_linear_newton_v1'. Provided value: {updater['method']!r}."
            )
        curve = updater["curve"]
        curve_path = _repo_file(curve, repo_root=repo_root)
        relaxed = updater["relaxation"] != 1.0
        return {
            **inactive,
            "pwl_voltage_tolerance": updater["voltage_tolerance"],
            "pwl_extrapolation": updater["extrapolation"],
            "pwl_nonconvergence_policy": updater["nonconvergence_policy"],
            "iv_data": load_iv_data(
                curve_path, expected_sha256=_legacy_curve_sha256(curve)
            ),
            "double_selector": "overrelaxed" if relaxed else "standard",
            "pwl_damping": updater["damping"],
            "pwl_max_steps": updater["max_steps"],
        }

    raise ValueError(
        "Expected simulation.nonlinearity to be a canonical paper family. "
        f"Provided value: {family!r}."
    )


def _legacy_curve_sha256(curve: str | Mapping[str, Any]) -> str | None:
    """Return the checksum carried only by legacy archived curve references."""

    return None if isinstance(curve, str) else curve["sha256"]


def _repo_file(reference: str | Mapping[str, Any], *, repo_root: Path) -> Path:
    root = repo_root.expanduser().resolve()
    configured_path = reference if isinstance(reference, str) else reference["path"]
    relative = Path(configured_path)
    if relative.is_absolute():
        raise ValueError(
            "Expected a repository-relative scientific asset path. "
            f"Provided value: {configured_path!r}."
        )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Expected a scientific asset path to resolve inside the repository. "
            f"Provided value: {configured_path!r}."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected the configured scientific asset to exist. Provided value: {path}."
        )
    return path


__all__ = ["build_minimizer", "simulation_assets"]
