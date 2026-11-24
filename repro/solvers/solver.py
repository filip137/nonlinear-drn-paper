"""Standalone equilibrium solvers for layered DRNs with full double-Shockley shunts.

The input is clamped. All hidden and output voltages are solved jointly by
Newton, from zeros by default. No CD result is used to initialize Newton.

For free layer l (zero based), A_ll = diag(sum(W_l, axis=0) +
av*ai*sum(W_{l+1}, axis=1)), with the outgoing term absent at the output.
A_l,l+1 = -ai W_{l+1}, A_l+1,l = -av W_{l+1}. The first-layer forcing is
x W_0; the input itself is not amplified. Bias currents can be supplied in
the same physical KCL units. Nonlinear shunts are present only on hidden nodes.

F(v) = A v - b + i(v); J(v) = A + diag(i'(v)). Similarity scaling by
D_l = (ai/av)**(l/2) makes D J D^-1 symmetric. A block Cholesky elimination
uses this structure without forming a dense whole-network Jacobian. The
numeric factors are recomputed at each Newton step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve


@dataclass(frozen=True)
class DoubleShockley:
    saturation_current: float = 1e-6
    thermal_voltage: float = 0.05
    offset_voltage: float = 0.8

    def __post_init__(self):
        values = (self.saturation_current, self.thermal_voltage, self.offset_voltage)
        if (
            not all(np.isfinite(values))
            or self.saturation_current < 0
            or self.thermal_voltage <= 0
            or self.offset_voltage < 0
        ):
            raise ValueError("Require finite Is >= 0, Vt > 0, and offset >= 0.")

    def current_derivative(self, v):
        """Full antiparallel law; never clip or change the device equation."""
        if self.saturation_current == 0:
            return np.zeros_like(v), np.zeros_like(v)
        # Overflow at a trial point is rejected by the line search. Multiplying
        # expm1 by the dominant branch avoids cancellation of the current near 0.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            magnitude = np.abs(v)
            forward = self.saturation_current * np.exp(
                (magnitude - self.offset_voltage) / self.thermal_voltage
            )
            reverse_ratio = np.exp(-2 * magnitude / self.thermal_voltage)
            current = (
                np.sign(v)
                * forward
                * (-np.expm1(-2 * magnitude / self.thermal_voltage))
            )
            derivative = forward * (1 + reverse_ratio) / self.thermal_voltage
        return current, derivative

    def coordinate_root(self, diagonal, forcing):
        """Solve diagonal*v + i(v) = forcing by safeguarded scalar Newton.

        Used only by the independent CD control. The coupled Newton solver
        never calls this function. All coordinates in a layer are independent.
        """
        if self.saturation_current == 0:
            return forcing / diagonal
        drive = np.abs(forcing)
        lo = np.zeros_like(drive)
        hi = np.minimum(
            drive / diagonal,
            self.offset_voltage
            + self.thermal_voltage * np.log1p(drive / self.saturation_current),
        )
        value = hi.copy()
        done = drive == 0
        for _ in range(100):
            current, derivative = self.current_derivative(value)
            residual = diagonal * value - drive + current
            correction = residual / (diagonal + derivative)
            close = np.abs(correction) <= 4 * np.finfo(np.float64).eps * np.maximum(
                1, value
            )
            lo = np.where((residual < 0) & ~done, value, lo)
            hi = np.where((residual > 0) & ~done, value, hi)
            candidate = value - correction
            candidate = np.where(
                (candidate > lo) & (candidate < hi), candidate, (lo + hi) / 2
            )
            candidate = np.where(close, np.maximum(value - correction, 0), candidate)
            value = np.where(done, value, candidate)
            done |= close
            if np.all(done):
                return np.sign(forcing) * value
        raise RuntimeError("Scalar CD update failed to converge.")


class LayeredDRN:
    """Nonnegative dense inter-layer conductances; arbitrary positive gains."""

    def __init__(
        self,
        weights: Sequence[np.ndarray],
        law: DoubleShockley,
        voltage_gain: float = 1.0,
        current_gain: float = 1.0,
        bias_currents: Sequence[np.ndarray] | None = None,
        *,
        dtype=np.float64,
    ):
        if not weights:
            raise ValueError(
                "At least one input-to-output conductance matrix is required."
            )
        if (
            not np.isfinite(voltage_gain)
            or not np.isfinite(current_gain)
            or min(voltage_gain, current_gain) <= 0
        ):
            raise ValueError("Amplifier gains must be finite and positive.")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Supported solver precisions are float32 and float64.")
        self.weights = tuple(
            np.array(w, dtype=self.dtype, order="C", copy=True) for w in weights
        )
        for index, weight in enumerate(self.weights):
            if (
                weight.ndim != 2
                or min(weight.shape) == 0
                or not np.isfinite(weight).all()
                or np.any(weight < 0)
            ):
                raise ValueError("Conductances must be finite nonnegative matrices.")
            if index and weight.shape[0] != self.weights[index - 1].shape[1]:
                raise ValueError("Adjacent weight dimensions do not match.")
        self.law, self.av, self.ai = law, float(voltage_gain), float(current_gain)
        self.widths = tuple(w.shape[1] for w in self.weights)
        self.starts = np.r_[0, np.cumsum(self.widths)]
        self.slices = tuple(
            slice(a, b) for a, b in zip(self.starts[:-1], self.starts[1:])
        )
        self.size, self.hidden = int(self.starts[-1]), int(self.starts[-2])
        self.diagonal = np.concatenate(
            [
                w.sum(0)
                + (
                    self.av * self.ai * self.weights[j + 1].sum(1)
                    if j + 1 < len(self.weights)
                    else 0
                )
                for j, w in enumerate(self.weights)
            ]
        )
        if np.any(self.diagonal <= 0):
            raise ValueError("Every free node must have positive incident conductance.")
        self.scale = np.concatenate(
            [
                np.full(width, (self.ai / self.av) ** (j / 2), dtype=self.dtype)
                for j, width in enumerate(self.widths)
            ]
        )
        if not np.isfinite(self.scale).all() or np.any(self.scale == 0):
            raise ValueError("Amplifier scaling exceeds the selected precision range.")
        self.symmetric_links = tuple(
            -self.dtype.type(np.sqrt(self.av * self.ai)) * w for w in self.weights[1:]
        )
        if bias_currents is None:
            self.bias = np.zeros(self.size, dtype=self.dtype)
        else:
            if len(bias_currents) != len(self.widths) or any(
                np.shape(b) != (width,) for b, width in zip(bias_currents, self.widths)
            ):
                raise ValueError("Supply one bias-current vector per free layer.")
            self.bias = np.concatenate(bias_currents).astype(self.dtype)
            if not np.isfinite(self.bias).all():
                raise ValueError("Bias currents must be finite.")

    def forcing(self, inputs):
        inputs = np.asarray(inputs, dtype=self.dtype)
        if (
            inputs.ndim not in (1, 2)
            or inputs.shape[-1] != self.weights[0].shape[0]
            or not np.isfinite(inputs).all()
        ):
            raise ValueError("Expected finite clamped input voltage vectors.")
        result = np.broadcast_to(self.bias, inputs.shape[:-1] + (self.size,)).copy()
        result[..., self.slices[0]] += inputs @ self.weights[0]
        return result

    def linear(self, v):
        result = self.diagonal * v
        for j, w in enumerate(self.weights[1:]):
            left, right = self.slices[j : j + 2]
            result[..., left] -= self.ai * (v[..., right] @ w.T)
            result[..., right] -= self.av * (v[..., left] @ w)
        return result

    def residual_derivative(self, v, forcing):
        current, derivative = self.law.current_derivative(v[..., : self.hidden])
        residual = self.linear(v) - forcing
        residual[..., : self.hidden] += current
        full_derivative = np.zeros_like(v)
        full_derivative[..., : self.hidden] = derivative
        return residual, full_derivative

    def residual(self, v, forcing):
        return self.residual_derivative(v, forcing)[0]

    def coordinate_root(self, layer_index, diagonal, forcing):
        """Local device solve; subclasses may supply node-specific orientations."""
        return self.law.coordinate_root(diagonal, forcing)

    def jacobian(self, v):
        """Analytic physical-KCL Jacobian, useful for independent checks."""
        v = np.asarray(v)
        if v.shape != (self.size,):
            raise ValueError("Jacobian expects one complete voltage vector.")
        _, derivative = self.residual_derivative(v, np.zeros_like(v))
        matrix = np.diag(self.diagonal + derivative)
        for j, w in enumerate(self.weights[1:]):
            left, right = self.slices[j : j + 2]
            matrix[left, right] = -self.ai * w
            matrix[right, left] = -self.av * w.T
        return matrix

    def linear_step(self, derivative, rhs, backend="block_cholesky"):
        """Solve J*step = rhs; no matrix inverse and no finite differences."""
        if backend == "dense_lu":
            matrix = np.diag(self.diagonal + derivative)
            for j, w in enumerate(self.weights[1:]):
                left, right = self.slices[j : j + 2]
                matrix[left, right] = -self.ai * w
                matrix[right, left] = -self.av * w.T
            return solve(matrix, rhs, assume_a="gen", check_finite=False)
        if backend != "block_cholesky":
            raise ValueError(f"Unknown linear solver: {backend}")
        diagonal = self.diagonal + derivative
        scaled_rhs = self.scale * rhs
        transformed_rhs, transformed_links = [], []
        for j, s in enumerate(self.slices):
            local_rhs = scaled_rhs[s].copy()
            if j == 0:
                u = local_rhs / diagonal[s]
                link = (
                    self.symmetric_links[j] / diagonal[s, None]
                    if j < len(self.symmetric_links)
                    else None
                )
            else:
                previous = self.symmetric_links[j - 1]
                schur = np.diag(diagonal[s]) - previous.T @ transformed_links[j - 1]
                local_rhs -= previous.T @ transformed_rhs[j - 1]
                factor = cho_factor(schur, lower=True, check_finite=False)
                u = cho_solve(factor, local_rhs, check_finite=False)
                link = (
                    cho_solve(factor, self.symmetric_links[j], check_finite=False)
                    if j < len(self.symmetric_links)
                    else None
                )
            transformed_rhs.append(u)
            if link is not None:
                transformed_links.append(link)
        solution = [transformed_rhs[-1]]
        for j in range(len(self.slices) - 2, -1, -1):
            solution.append(transformed_rhs[j] - transformed_links[j] @ solution[-1])
        return np.concatenate(solution[::-1]) / self.scale

    def residual_voltage_norm(self, residual):
        """Max |KCL current / fixed linear diagonal conductance|, in volts."""
        return np.max(np.abs(residual) / self.diagonal, axis=-1)

    def residual_norm(self, residual, kind="voltage"):
        """Stopping norm: absolute KCL current (A), or scaled residual (V)."""
        if kind == "current":
            return np.max(np.abs(residual), axis=-1)
        if kind == "voltage":
            return self.residual_voltage_norm(residual)
        raise ValueError(f"Unknown residual norm: {kind}")


@dataclass
class SolveResult:
    voltages: np.ndarray
    converged: bool
    iterations: int
    residual_voltage: float
    max_current_residual: float
    backtracks: int = 0
    residual_evaluations: int = 0
    reason: str = "converged"


def _state(circuit, forcing, initial):
    forcing = np.asarray(forcing, dtype=circuit.dtype)
    if forcing.shape != (circuit.size,) or not np.isfinite(forcing).all():
        raise ValueError("Expected one finite forcing vector.")
    v = (
        np.zeros(circuit.size, dtype=circuit.dtype)
        if initial is None
        else np.array(initial, dtype=circuit.dtype, copy=True)
    )
    if v.shape != forcing.shape or not np.isfinite(v).all():
        raise ValueError(
            "Initial voltages must match the forcing vector and be finite."
        )
    return v, forcing


def newton(
    circuit: LayeredDRN,
    forcing,
    *,
    tolerance=1e-10,
    max_iterations=100,
    initial=None,
    backend="block_cholesky",
    max_backtracks=60,
    trace: Callable | None = None,
    residual_kind="voltage",
    damping="armijo",
    fixed_alpha=0.5,
    voltage_rtol=None,
    voltage_atol=0.0,
) -> SolveResult:
    """Global Newton with Armijo, full steps, or a fixed damping factor.

    Tolerance is A for current, V for voltage. Full/fixed steps are accepted
    without a merit test or backtracking, including steps that worsen residuals.
    The Armijo merit function remains the conductance-scaled residual 2-norm,
    independently of the chosen stopping norm. Trace uses the stopping units.
    With voltage_rtol, the repository CD rule replaces the residual stop: after
    an accepted step, max|V_new-V_old| <= voltage_rtol*max|V_old| + voltage_atol.
    """
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or not isinstance(max_iterations, int)
        or max_iterations < 0
        or not isinstance(max_backtracks, int)
        or max_backtracks < 0
    ):
        raise ValueError("Invalid Newton convergence settings.")
    if backend not in ("dense_lu", "block_cholesky"):
        raise ValueError(f"Unknown linear solver: {backend}")
    if voltage_rtol is not None and (
        not np.isfinite([voltage_rtol, voltage_atol]).all()
        or min(voltage_rtol, voltage_atol) < 0
        or voltage_rtol + voltage_atol == 0
    ):
        raise ValueError(
            "Voltage-change stopping requires finite nonnegative tolerances, not both zero."
        )
    if damping not in ("armijo", "none", "fixed"):
        raise ValueError(f"Unknown Newton damping: {damping}")
    if damping == "fixed" and (
        not np.isfinite(fixed_alpha) or not 0 < fixed_alpha <= 1
    ):
        raise ValueError("Fixed damping requires 0 < alpha <= 1.")
    v, forcing = _state(circuit, forcing, initial)
    residual, derivative = circuit.residual_derivative(v, forcing)
    evaluations, backtracks = 1, 0
    reason = "maximum_iterations"
    change = scale = None
    for iteration in range(max_iterations + 1):
        norm = float(circuit.residual_norm(residual, residual_kind))
        if trace is not None:
            trace(iteration, norm)
        if not np.isfinite(norm) or not np.isfinite(derivative).all():
            reason = "nonfinite_initial_or_accepted_state"
            break
        if (
            norm <= tolerance
            if voltage_rtol is None
            else change is not None and change <= voltage_rtol * scale + voltage_atol
        ):
            return SolveResult(
                v,
                True,
                iteration,
                float(circuit.residual_voltage_norm(residual)),
                float(np.max(np.abs(residual))),
                backtracks,
                evaluations,
            )
        if iteration == max_iterations:
            break
        try:
            step = circuit.linear_step(derivative, -residual, backend)
        except np.linalg.LinAlgError:
            reason = "linear_solve_failed"
            break
        if damping != "armijo":
            alpha = 1.0 if damping == "none" else fixed_alpha
            change, scale = float(np.max(np.abs(alpha * step))), float(
                np.max(np.abs(v))
            )
            v = v + alpha * step
            residual, derivative = circuit.residual_derivative(v, forcing)
            evaluations += 1
            continue
        scaled = residual / circuit.diagonal
        merit_norm = float(np.linalg.norm(scaled))
        accepted = False
        for halving in range(max_backtracks + 1):
            alpha = 2.0 ** (-halving)
            candidate = v + alpha * step
            trial_residual, trial_derivative = circuit.residual_derivative(
                candidate, forcing
            )
            evaluations += 1
            trial_norm = float(np.linalg.norm(trial_residual / circuit.diagonal))
            # Equivalent to Armijo on half the squared residual norm for the
            # exact Newton direction, with c1=1e-4. Avoid squaring huge values.
            if (
                np.isfinite(trial_norm)
                and trial_norm <= np.sqrt(1 - 2e-4 * alpha) * merit_norm
            ):
                change, scale = float(np.max(np.abs(candidate - v))), float(
                    np.max(np.abs(v))
                )
                v, residual, derivative = candidate, trial_residual, trial_derivative
                backtracks += halving
                accepted = True
                break
        if not accepted:
            reason = "line_search_failed"
            break
    return SolveResult(
        v,
        False,
        iteration,
        float(circuit.residual_voltage_norm(residual)),
        float(np.max(np.abs(residual))),
        backtracks,
        evaluations,
        reason,
    )


def coordinate_descent(
    circuit: LayeredDRN,
    forcing,
    *,
    tolerance=1e-10,
    max_sweeps=4096,
    relaxation=1.0,
    initial=None,
    residual_kind="voltage",
) -> SolveResult:
    """Matched full-law CD control; same odd/even layer schedule as the repo.

    All updates, including the linear output update, use the archived relaxation,
    as in the repository. This is a NumPy control, not the production Lambert-W
    implementation. Both this control and Newton use the same physical KCL.
    """
    if (
        tolerance <= 0
        or not np.isfinite(tolerance)
        or not 0 < relaxation < 2
        or max_sweeps < 0
    ):
        raise ValueError("Invalid CD convergence settings.")
    v, forcing = _state(circuit, forcing, initial)
    residual = circuit.residual(v, forcing)
    order = list(range(0, len(circuit.slices), 2)) + list(
        range(1, len(circuit.slices), 2)
    )
    for sweep in range(max_sweeps + 1):
        norm = float(circuit.residual_norm(residual, residual_kind))
        if norm <= tolerance:
            return SolveResult(
                v,
                True,
                sweep,
                float(circuit.residual_voltage_norm(residual)),
                float(np.max(np.abs(residual))),
                residual_evaluations=sweep + 1,
            )
        if sweep == max_sweeps or not np.isfinite(norm):
            break
        for j in order:
            s = circuit.slices[j]
            local_forcing = forcing[s].copy()
            if j:
                local_forcing += circuit.av * (
                    v[circuit.slices[j - 1]] @ circuit.weights[j]
                )
            if j + 1 < len(circuit.slices):
                local_forcing += circuit.ai * (
                    v[circuit.slices[j + 1]] @ circuit.weights[j + 1].T
                )
                root = circuit.coordinate_root(j, circuit.diagonal[s], local_forcing)
            else:
                root = local_forcing / circuit.diagonal[s]
            v[s] += relaxation * (root - v[s])
        residual = circuit.residual(v, forcing)
    return SolveResult(
        v,
        False,
        sweep,
        float(circuit.residual_voltage_norm(residual)),
        float(np.max(np.abs(residual))),
        residual_evaluations=sweep + 1,
        reason="maximum_sweeps" if np.isfinite(residual).all() else "nonfinite_state",
    )
