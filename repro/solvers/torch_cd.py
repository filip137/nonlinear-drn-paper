"""Single-input CPU float64 adapter for the repository's fused Torch CD sweeps."""
import numpy as np
import torch

from repro.minimizer_factory import build_minimizer
from repro.small_network import PACK_ROOT, _build_network
from .single_diode_solver import SingleDiodeDRN
from .solver import DoubleShockley, LayeredDRN, SolveResult


def shockley_circuit(weights, simulation):
    physical = simulation["physical"]
    gains = simulation["amplification"]
    av, ai = gains["voltage_factor"], gains["current_factor"]
    family = simulation["nonlinearity"]
    if family == "single_diode_exponential":
        return SingleDiodeDRN(weights, physical, av, ai)
    if family == "double_diode_exponential":
        return LayeredDRN(weights, DoubleShockley(**physical), av, ai)
    raise ValueError("Expected a single or double Shockley simulation.")


class TorchCD:
    """Fused Shockley updates with adaptive relaxation and a full-law KCL stop.

    Supply a v2 simulation block and input-to-output matrices. Each solve starts
    at zero and accepts one clamped input vector, including any input gain.
    """

    def __init__(self, weights, simulation, *, compiled=False):
        self.circuit = shockley_circuit(weights, simulation)
        sizes = (self.circuit.weights[0].shape[0], *self.circuit.widths)
        # Select the over-relaxed updaters. The adaptive policy sets omega to
        # one at the start of each solve and then estimates it from the sweeps.
        simulation = dict(simulation, updater={
            **simulation["updater"], "dtype": "float64", "relaxation": 1.4,
        })
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            energy, self.network = _build_network(
                sizes=sizes, matrices=self.circuit.weights, inputs=np.zeros((1, sizes[0])),
                input_gain=1.0, input_encoding="direct", bias_method="none",
                simulation=simulation, device=torch.device("cpu"),
            )
        self.layers = self.network.free_layers()
        for layer in self.layers:
            layer.state = layer.state.double()
        self.minimizer = build_minimizer(
            fn=energy, free_layers=self.layers, simulation=simulation, repo_root=PACK_ROOT,
            equilibrium=dict(method="voltage_change", max_sweeps=1,
                             relative_tolerance=1e-12, absolute_tolerance=1e-12,
                             update_order="asynchronous"),
        )
        self.minimizer.use_fused_updates(compile=compiled)
        self.minimizer.set_adaptive_relaxation(estimator="two_term", initial=1.0)

    @torch.no_grad()
    def solve(self, inputs, *, tolerance=1e-9, max_sweeps=8192, vol_tol=None):
        if (
            (tolerance is None and vol_tol is None)
            or any(not np.isfinite(v) or v <= 0 for v in (tolerance, vol_tol) if v is not None)
            or not isinstance(max_sweeps, int) or max_sweeps < 0
        ):
            raise ValueError("Expected positive stopping tolerances and nonnegative sweep limit.")
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 1:
            raise ValueError("Expected one clamped input vector.")
        forcing = self.circuit.forcing(inputs)
        self.network.layers()[0].state = torch.from_numpy(inputs.copy()).unsqueeze(0)
        for layer in self.layers:
            layer.state.zero_()
        for sweep in range(max_sweeps + 1):
            state = torch.cat([layer.state for layer in self.layers], dim=1)[0].numpy()
            residual = np.abs(self.circuit.residual(state, forcing))
            norm = float(residual.max())
            voltage_ok = vol_tol is None or (sweep > 0 and self.minimizer.final_max_voltage_change <= vol_tol)
            converged = voltage_ok and (tolerance is None or norm <= tolerance) and bool(np.isfinite(norm))
            if converged or not np.isfinite(norm) or sweep == max_sweeps:
                break
            self.minimizer.compute_equilibrium(resume=sweep > 0)
        reason = "converged" if converged else "maximum_sweeps"
        if not np.isfinite(norm):
            reason = "nonfinite_residual"
        return SolveResult(
            voltages=state, converged=converged, iterations=sweep,
            residual_voltage=float((residual / self.circuit.diagonal).max()),
            max_current_residual=norm, residual_evaluations=sweep + 1, reason=reason,
        )
