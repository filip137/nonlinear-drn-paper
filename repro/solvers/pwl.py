"""Measured-table Newton law and the sequential CD used for transistor shunts.

Currents are signed; paired PMOS tables must not be mirrored. End segments are
extended beyond the measured grid, matching the review's L2/L3 and Newton arms.
"""
from __future__ import annotations

import numpy as np

from .fast_cd import AdaptiveRelaxation
from .solver import SolveResult, _state


class TableLaw:
    """Monotone currents on a common voltage grid, assigned by hidden-node kind.

    ``currents`` is (device kinds, knots), in amperes; ``kinds`` selects a row
    for each hidden node. Divide by current_unit_A to use network current units.
    """

    def __init__(self, voltage, currents, kinds, *, current_unit_A=1.0):
        self.v = np.array(voltage, dtype=np.float64, copy=True)
        current = np.atleast_2d(np.array(currents, dtype=np.float64, copy=True))
        raw_kinds = np.asarray(kinds)
        if (
            self.v.ndim != 1
            or self.v.size < 2
            or current.ndim != 2
            or current.shape[0] < 1
            or current.shape[1] != self.v.size
            or not np.isfinite(self.v).all()
            or not np.isfinite(current).all()
            or not np.all(np.diff(self.v) > 0)
            or not np.all(np.diff(current, axis=1) >= 0)
        ):
            raise ValueError(
                "Expected increasing finite voltage knots and monotone finite currents."
            )
        if (
            raw_kinds.ndim != 1
            or not np.issubdtype(raw_kinds.dtype, np.integer)
            or np.any(raw_kinds < 0)
            or np.any(raw_kinds >= len(current))
        ):
            raise ValueError(
                "Supply a valid integer device kind for every hidden node."
            )
        if not np.isfinite(current_unit_A) or current_unit_A <= 0:
            raise ValueError("current_unit_A must be finite and positive.")
        self.current_unit_A = float(current_unit_A)
        self.kind = raw_kinds.astype(np.intp, copy=True)
        self.i = current / self.current_unit_A
        self.slope = np.diff(self.i, axis=1) / np.diff(self.v)
        self.intercept = self.i[:, :-1] - self.slope * self.v[:-1]

    @classmethod
    def paired(
        cls, voltage, nmos_current, pmos_current, hidden_widths, *, current_unit_A=1e-7
    ):
        """First half NMOS, second half PMOS, separately in each hidden layer."""
        if not hidden_widths or any(
            not isinstance(w, (int, np.integer)) or w < 2 or w % 2
            for w in hidden_widths
        ):
            raise ValueError("Paired tables require positive even hidden widths.")
        kinds = np.concatenate([np.arange(w) >= w // 2 for w in hidden_widths]).astype(
            np.intp
        )
        return cls(
            voltage,
            np.stack([nmos_current, pmos_current]),
            kinds,
            current_unit_A=current_unit_A,
        )

    def current_derivative(self, v):
        v = np.asarray(v)
        if v.ndim < 1 or v.shape[-1] != self.kind.size:
            raise ValueError("Voltage vector must match the hidden-node device kinds.")
        segment = np.clip(
            np.searchsorted(self.v, v, side="right") - 1, 0, len(self.v) - 2
        )
        kind = np.broadcast_to(self.kind, segment.shape)
        slope = self.slope[kind, segment]
        return self.intercept[kind, segment] + slope * v, slope


class PWLCoordinateDescent:
    """One-input float64 sequential sweep; optionally compile the whole sweep.

    With ``adaptive=True, compiled=True`` this is the transistor review's L3.
    Compilation is lazy and uses PyTorch's standard compiler cache.
    """

    def __init__(self, circuit, *, adaptive=True, compiled=True):
        import torch

        if (
            not isinstance(circuit.law, TableLaw)
            or circuit.law.kind.size != circuit.hidden
        ):
            raise ValueError(
                "Expected a TableLaw with one device kind per hidden node."
            )
        if circuit.dtype != np.dtype(np.float64) or len(circuit.weights) < 2:
            raise ValueError("PWL CD requires a float64 circuit with a hidden layer.")
        self.circuit = circuit
        self.w = [torch.tensor(w, dtype=torch.float64) for w in circuit.weights]
        self.depth = len(self.w) - 1
        self.av, self.ai = circuit.av, circuit.ai
        self.output_diagonal = self.w[-1].sum(0)
        self.v = torch.tensor(circuit.law.v, dtype=torch.float64)
        self.boundaries, self.derivatives = [], []
        for k in range(self.depth):
            diagonal = self.w[k].sum(0) + self.av * self.ai * self.w[k + 1].sum(1)
            kinds = circuit.law.kind[circuit.slices[k]]
            current = torch.tensor(circuit.law.i[kinds], dtype=torch.float64)
            self.boundaries.append((diagonal[:, None] * self.v + current).contiguous())
            self.derivatives.append(
                (diagonal[:, None] + current.diff(dim=1) / self.v.diff()).contiguous()
            )
        self.knots = self.v.numel()
        self.steps = int(np.ceil(np.log2(self.knots)))
        self.relaxation = AdaptiveRelaxation() if adaptive else None
        self.sweep = (
            torch.compile(self._sweep, dynamic=False, fullgraph=True)
            if compiled
            else self._sweep
        )
        self.reset(np.zeros(circuit.size))

    def _sweep(self, z, forcing, omega):
        import torch

        out = list(z)
        for k in range(self.depth):
            incoming = (
                forcing[k] if k == 0 else forcing[k] + self.av * out[k - 1] @ self.w[k]
            )
            rhs = (incoming + self.ai * out[k + 1] @ self.w[k + 1].T)[0]
            bounds, slope, v = self.boundaries[k], self.derivatives[k], self.v
            lo = torch.zeros_like(rhs, dtype=torch.long)
            hi = torch.full_like(lo, self.knots - 1)
            for _ in range(self.steps):
                mid = (lo + hi + 1) // 2
                below = bounds.gather(1, mid[:, None])[:, 0] <= rhs
                lo = torch.where(below, mid, lo)
                hi = torch.where(below, hi, mid - 1)
            segment = lo.clamp(max=self.knots - 2)[:, None]
            root = (
                v[segment[:, 0]]
                + (rhs - bounds.gather(1, segment)[:, 0])
                / slope.gather(1, segment)[:, 0]
            )
            out[k] = (out[k][0] + omega * (root - out[k][0]))[None]
        output = (forcing[-1] + self.av * out[-2] @ self.w[-1]) / self.output_diagonal
        out[-1] = out[-1] + omega * (output - out[-1])
        return out

    def reset(self, forcing):
        import torch

        _, forcing = _state(self.circuit, forcing, None)
        self.z = [torch.zeros((1, w.shape[1]), dtype=torch.float64) for w in self.w]
        self.forcing = [
            torch.tensor(forcing[s][None], dtype=torch.float64)
            for s in self.circuit.slices
        ]
        if self.relaxation is not None:
            self.relaxation.reset()

    def step(self):
        import torch

        omega = 1.0 if self.relaxation is None else self.relaxation.omega
        old = self.z
        with torch.inference_mode():
            self.z = self.sweep(
                self.z, self.forcing, torch.tensor(omega, dtype=torch.float64)
            )
        if self.relaxation is not None:
            # Keep Torch dot products, as in the review's adaptive policy.
            self.relaxation.observe(
                torch.cat([(a - b)[0] for a, b in zip(self.z, old)])
            )

    def state(self):
        import torch

        return torch.cat(self.z, dim=1)[0].numpy().copy()

    def solve(self, forcing, *, tolerance=1e-9, max_sweeps=4096):
        """Reset to zero and stop on max|KCL| in network current units."""
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0
            or not isinstance(max_sweeps, int)
            or max_sweeps < 0
        ):
            raise ValueError("Invalid PWL CD convergence settings.")
        self.reset(forcing)
        for sweep in range(max_sweeps + 1):
            v = self.state()
            residual = self.circuit.residual(v, forcing)
            norm = float(np.max(np.abs(residual)))
            if norm <= tolerance or not np.isfinite(norm) or sweep == max_sweeps:
                break
            self.step()
        reason = (
            "converged"
            if norm <= tolerance
            else ("maximum_sweeps" if np.isfinite(norm) else "nonfinite_state")
        )
        return SolveResult(
            v,
            norm <= tolerance,
            sweep,
            float(self.circuit.residual_voltage_norm(residual)),
            norm,
            residual_evaluations=sweep + 1,
            reason=reason,
        )
