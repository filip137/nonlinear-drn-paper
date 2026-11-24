#!/usr/bin/env python3
"""Compare review solvers on the network in an existing small-network config.

This comparison uses CPU float64 and the stopping settings below. The config
supplies topology, inputs, device parameters, amplifier gains, and Torch CD
coordinate-update settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from repro.small_network import load_small_network_config  # noqa: E402
from repro.solvers import (
    AdaptiveRelaxation,
    TorchCD,
    FastCD,
    newton_cached,
    newton,
)  # noqa: E402
from repro.solvers.torch_cd import shockley_circuit  # noqa: E402


SOLVERS = ("fast_cd", "torch_cd", "dense_lu", "block_cholesky", "newton_cached")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PACK_ROOT / "configs/small_network/example.json"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="FastCD voltage-change relative tolerance",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="FastCD voltage-change absolute tolerance (V)",
    )
    parser.add_argument("--max-sweeps", type=int, default=8192)
    parser.add_argument(
        "--current-tolerance",
        type=float,
        default=1e-9,
        help="Torch CD and Newton max|KCL| tolerance (A)",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--solver", choices=SOLVERS, action="append",
                        help="Run only this solver (repeatable). Default: all five.")
    args = parser.parse_args(argv)
    selected = args.solver or SOLVERS
    config = load_small_network_config(args.config, repo_root=PACK_ROOT)
    network, simulation = config["network"], config["simulation"]
    family = simulation["nonlinearity"]
    if family not in ("single_diode_exponential", "double_diode_exponential"):
        parser.error(
            "This example compares Shockley solvers; use TableLaw/PWLCoordinateDescent for measured tables."
        )
    law = "single" if family == "single_diode_exponential" else "double"
    weights = [np.asarray(w, dtype=np.float64) for w in network["conductances"]]
    inputs = (
        np.asarray(network["input_voltages"], dtype=np.float64) * network["input_gain"]
    )
    physical, gains = simulation["physical"], simulation["amplification"]
    av, ai = gains["voltage_factor"], gains["current_factor"]
    torch_cd = TorchCD(weights, simulation) if "torch_cd" in selected else None
    circuit = torch_cd.circuit if torch_cd else shockley_circuit(weights, simulation)
    cd = FastCD(weights, law, av, ai, physical=physical) if "fast_cd" in selected else None
    records = []
    with threadpool_limits(limits=1):
        for index, forcing in enumerate(circuit.forcing(inputs)):
            for solver in selected:
                if solver == "fast_cd":
                    iterations = cd.solve(forcing[circuit.slices[0]], AdaptiveRelaxation(),
                                          rtol=args.rtol, atol=args.atol, max_sweeps=args.max_sweeps)
                    state, converged, reason = cd.state, cd.converged, cd.reason
                else:
                    if solver == "torch_cd":
                        result = torch_cd.solve(inputs[index], tolerance=args.current_tolerance,
                                                max_sweeps=args.max_sweeps)
                    elif solver == "newton_cached":
                        result = newton_cached(circuit, forcing, tolerance=args.current_tolerance,
                                               max_iterations=args.max_iterations)
                    else:
                        result = newton(circuit, forcing, backend=solver, residual_kind="current",
                                        tolerance=args.current_tolerance, max_iterations=args.max_iterations)
                    state, converged = result.voltages, result.converged
                    iterations, reason = result.iterations, result.reason
                records.append(dict(
                    input=index, solver=solver, converged=converged, iterations=iterations,
                    reason=reason, voltages=state.tolist(),
                    max_current_residual=float(np.abs(circuit.residual(state, forcing)).max()),
                ))
    print(
        json.dumps(
            dict(
                device="cpu",
                dtype="float64",
                threads=1,
                cd_stop=dict(
                    rtol=args.rtol, atol=args.atol, max_sweeps=args.max_sweeps
                ),
                torch_cd_stop=dict(current_tolerance=args.current_tolerance, max_sweeps=args.max_sweeps),
                newton_stop=dict(
                    current_tolerance=args.current_tolerance,
                    max_iterations=args.max_iterations,
                ),
                results=records,
            ),
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if all(record["converged"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
