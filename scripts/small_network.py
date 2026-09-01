#!/usr/bin/env python3
"""Editable example for a hand-specified nonlinear resistive network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))

from repro import simulate_small_network  # noqa: E402


def main() -> int:
    # These are physical node counts: input -> hidden -> output.
    layer_sizes = [2, 4, 2]

    # Matrix rows belong to the preceding layer and columns to the next layer.
    conductances = [
        np.array(
            [
                [1.0, 0.2, 0.8, 0.4],
                [0.3, 1.1, 0.5, 0.9],
            ]
        ),
        np.array(
            [
                [0.8, 0.2],
                [0.4, 1.0],
                [1.1, 0.3],
                [0.2, 0.7],
            ]
        ),
    ]

    # Each row is one input-voltage set. No positive/negative channels are added.
    input_voltages = np.array(
        [
            [0.2, -0.1],
            [0.7, 0.3],
            [-0.4, 0.8],
        ]
    )

    # Choose "single", "double", or "pwl".
    non_linearity = "double"
    shockley_parameters = {
        "I_s": 1e-6,
        "V_t": 0.05,
        "V_off": 0.8,
    }

    result = simulate_small_network(
        layer_sizes=layer_sizes,
        conductances=conductances,
        input_voltages=input_voltages,
        non_linearity=non_linearity,
        shockley_parameters=shockley_parameters,
        adaptive_equilibrium=True,
        max_sweeps=128,
    )

    np.set_printoptions(precision=6, suppress=True)
    for index, voltages in enumerate(result.hidden_voltages, start=1):
        print(f"hidden_voltages[{index}]:")
        print(voltages)
    print("output_voltages:")
    print(result.output_voltages)
    print(f"converged: {result.converged}")
    print(f"sweeps: {result.sweeps}")
    print(f"final_max_voltage_change: {result.final_max_voltage_change:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
