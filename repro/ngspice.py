"""Independent ngspice DC operating points for hand-specified Shockley networks."""

from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def _operating_point_deck(network, simulation, inputs, nodes):
    gains = simulation["amplification"]
    physical = simulation["physical"]
    vt = physical["thermal_voltage"]
    offset = physical["offset_voltage"]
    saturation = physical["saturation_current"]
    double = simulation["nonlinearity"] == "double_diode_exponential"
    lines = ["DRN operating point", ".options reltol=1e-10 abstol=1e-14 vntol=1e-12"]
    sources = [f"in{i}" for i in range(len(inputs))]
    for name, value in zip(sources, inputs):
        lines.append(f"V{name} {name} 0 {value:.17g}")

    for layer, (matrix, targets) in enumerate(zip(network["conductances"], nodes)):
        for i, row in enumerate(matrix):
            for j, conductance in enumerate(row):
                if conductance:
                    lines.append(f"R{layer}_{i}_{j} {sources[i]} {targets[j]} {1/conductance:.17g}")
        if layer == len(nodes) - 1:
            break
        sources = []
        for j, node in enumerate(targets):
            # Sense the current delivered by the voltage amplifier and reflect
            # it into the hidden node with the configured current gain.
            drive = f"drive{layer}_{j}"
            lines.extend([
                f"E{node} amp{node} 0 {node} 0 {gains['voltage_factor']}",
                f"Vs{node} amp{node} {drive} 0",
                f"F{node} {node} 0 Vs{node} {gains['current_factor']}",
            ])
            sources.append(drive)
            forward = f"exp((v({node})-{offset})/{vt})"
            reverse = f"exp((-v({node})-{offset})/{vt})"
            if double:
                law = f"{forward}-{reverse}"
            elif j < len(targets) // 2:
                law = f"{forward}-1"
            else:
                law = f"1-{reverse}"
            lines.append(f"B{node} {node} 0 I={saturation}*({law})")

    vectors = " ".join(f"v({name})" for layer in nodes for name in layer)
    lines.extend([".control", "set wr_singlescale", "set numdgt=17", "op",
                  f"wrdata voltages.dat {vectors}", "quit", ".endc", ".end"])
    return "\n".join(lines) + "\n"


def ngspice_voltages(document):
    """Return one row of hidden and output voltages per configured input."""
    network, simulation = document["network"], document["simulation"]
    if simulation["nonlinearity"] not in ("single_diode_exponential", "double_diode_exponential"):
        raise ValueError("The ngspice comparison supports single and double Shockley networks.")
    executable = shutil.which("ngspice")
    if executable is None:
        raise RuntimeError("Install ngspice and put its executable on PATH to use --compare-ngspice.")
    nodes = [[f"n{layer}_{j}" for j in range(width)]
             for layer, width in enumerate(network["layer_sizes"][1:])]
    count = sum(map(len, nodes))
    states = []
    for index, inputs in enumerate(network["input_voltages"]):
        clamped = np.asarray(inputs) * network["input_gain"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            deck = _operating_point_deck(network, simulation, clamped, nodes)
            (directory / "network.cir").write_text(deck, encoding="ascii")
            run = subprocess.run([executable, "-n", "-b", "network.cir"], cwd=directory,
                                 capture_output=True, text=True)
            output = directory / "voltages.dat"
            if run.returncode or not output.is_file():
                raise RuntimeError(f"ngspice failed for input row {index}:\n{run.stdout}\n{run.stderr}")
            values = np.loadtxt(output, ndmin=2)
            if values.shape != (1, count + 1) or not np.isfinite(values).all():
                raise ValueError(f"Invalid ngspice operating point for input row {index}.")
            states.append(values[0, 1:])
    return np.stack(states)
