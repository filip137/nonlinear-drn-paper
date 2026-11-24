"""Block-Cholesky Newton that reuses its Jacobian factorization (chord / modified Newton).

solver.newton refactorizes J at every step; with block Cholesky that costs one Schur
factorization per layer boundary (W^3) per step, while applying an existing factorization
costs only triangular solves and matrix-vector products (W^2), like a CD sweep.

newton_cached keeps the factorization across steps and refreshes it at the current point
only when progress stalls: when an accepted step reduces the merit (the conductance-scaled
residual 2-norm, as in solver.newton) by less than a factor theta, or when a short line search
along the stale direction fails. With a fresh factorization the step is the Newton step with
the same Armijo backtracking as solver.newton. The stop is max|KCL| <= tolerance.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .solver import _state


class BlockCholesky:
    """LayeredDRN.linear_step(derivative, ., 'block_cholesky'), split into factor and solve."""

    def __init__(self, circuit, derivative):
        self.circuit = circuit
        diagonal = circuit.diagonal + derivative
        self.first = diagonal[circuit.slices[0]]
        self.factors, self.links = [None], []
        links = circuit.symmetric_links
        for j, s in enumerate(circuit.slices):
            if j == 0:
                link = links[0] / self.first[:, None] if links else None
            else:
                previous = links[j - 1]
                schur = np.diag(diagonal[s]) - previous.T @ self.links[j - 1]
                factor = cho_factor(schur, lower=True, check_finite=False)
                self.factors.append(factor)
                link = (
                    cho_solve(factor, links[j], check_finite=False)
                    if j < len(links)
                    else None
                )
            if link is not None:
                self.links.append(link)

    def solve(self, rhs):
        circuit = self.circuit
        scaled = circuit.scale * rhs
        transformed = []
        for j, s in enumerate(circuit.slices):
            local = scaled[s].copy()
            if j == 0:
                u = local / self.first
            else:
                local -= circuit.symmetric_links[j - 1].T @ transformed[j - 1]
                u = cho_solve(self.factors[j], local, check_finite=False)
            transformed.append(u)
        solution = [transformed[-1]]
        for j in range(len(circuit.slices) - 2, -1, -1):
            solution.append(transformed[j] - self.links[j] @ solution[-1])
        return np.concatenate(solution[::-1]) / circuit.scale


@dataclass
class CachedResult:
    voltages: np.ndarray
    converged: bool
    iterations: int
    factorizations: int
    residual_evaluations: int
    reason: str = "converged"


def newton_cached(
    circuit,
    forcing,
    *,
    tolerance=1e-9,
    max_iterations=200,
    theta=0.5,
    stale_backtracks=3,
    max_backtracks=60,
    initial=None,
    trace=None,
):
    """Modified Newton with Jacobian reuse; trace(iteration, max|KCL|, v) after every step."""
    if (
        not np.isfinite([tolerance, theta]).all()
        or tolerance <= 0
        or not 0 < theta <= 1
        or any(
            not isinstance(n, int) or n < 0
            for n in (max_iterations, stale_backtracks, max_backtracks)
        )
    ):
        raise ValueError("Invalid cached Newton convergence settings.")
    v, forcing = _state(circuit, forcing, initial)
    residual, derivative = circuit.residual_derivative(v, forcing)
    evaluations, factorizations, factor = 1, 0, None
    norm = float(np.max(np.abs(residual)))
    if trace is not None:
        trace(0, norm, v)
    reason = "maximum_iterations"
    completed = 0
    for iteration in range(max_iterations + 1):
        if not np.isfinite(norm) or not np.isfinite(derivative).all():
            reason = "nonfinite_initial_or_accepted_state"
            break
        if norm <= tolerance:
            return CachedResult(v, True, completed, factorizations, evaluations)
        if iteration == max_iterations:
            break
        merit = float(np.linalg.norm(residual / circuit.diagonal))
        while True:
            fresh = factor is None
            if fresh:
                try:
                    factor = BlockCholesky(circuit, derivative)
                except np.linalg.LinAlgError:
                    return CachedResult(
                        v,
                        False,
                        completed,
                        factorizations,
                        evaluations,
                        "linear_solve_failed",
                    )
                factorizations += 1
            step = factor.solve(-residual)
            accepted = False
            for halving in range((max_backtracks if fresh else stale_backtracks) + 1):
                alpha = 2.0 ** (-halving)
                candidate = v + alpha * step
                trial_residual, trial_derivative = circuit.residual_derivative(
                    candidate, forcing
                )
                evaluations += 1
                trial = float(np.linalg.norm(trial_residual / circuit.diagonal))
                if np.isfinite(trial) and trial <= np.sqrt(1 - 2e-4 * alpha) * merit:
                    accepted = True
                    break
            if accepted or fresh:
                break
            factor = None  # the stale direction failed: refactorize here and retry
        if not accepted:
            reason = "line_search_failed"
            break
        v, residual, derivative = candidate, trial_residual, trial_derivative
        completed += 1
        norm = float(np.max(np.abs(residual)))
        if trace is not None:
            trace(completed, norm, v)
        if trial > theta * merit:
            factor = None  # slow progress: refresh the Jacobian at the next step
    return CachedResult(v, False, completed, factorizations, evaluations, reason)
