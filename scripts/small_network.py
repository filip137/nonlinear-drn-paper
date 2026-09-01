#!/usr/bin/env python3
"""Simulate a complete hand-specified network configuration."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Sequence


PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "repro" / "vendor"))

from repro import (  # noqa: E402
    load_small_network_config,
    make_input_voltage_sweep,
    simulate_small_network,
)
from repro.strict_config import pretty_json_text, validate_document  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a complete hand-specified resistive-network config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACK_ROOT / "configs" / "small_network" / "example.json",
    )
    parser.add_argument(
        "--sweep-inputs",
        action="store_true",
        help="Generate Cartesian input rows in a new complete config before running.",
    )
    parser.add_argument("--sweep-min", type=float)
    parser.add_argument("--sweep-max", type=float)
    parser.add_argument("--sweep-points", type=int)
    parser.add_argument(
        "--write-config",
        type=Path,
        help="Required with --sweep-inputs; receives the complete generated config.",
    )
    args = parser.parse_args(argv)
    sweep_only = (args.sweep_min, args.sweep_max, args.sweep_points, args.write_config)
    if not args.sweep_inputs and any(value is not None for value in sweep_only):
        parser.error(
            "--sweep-min, --sweep-max, --sweep-points, and --write-config "
            "are valid only with --sweep-inputs"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PACK_ROOT / config_path

    run_config: Path | dict = config_path
    if args.sweep_inputs:
        missing = [
            name
            for name, value in (
                ("--sweep-min", args.sweep_min),
                ("--sweep-max", args.sweep_max),
                ("--sweep-points", args.sweep_points),
                ("--write-config", args.write_config),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "--sweep-inputs requires explicit " + ", ".join(missing)
            )
        document = copy.deepcopy(
            load_small_network_config(config_path, repo_root=PACK_ROOT)
        )
        inputs = make_input_voltage_sweep(
            document["network"]["layer_sizes"][0],
            voltage_min=args.sweep_min,
            voltage_max=args.sweep_max,
            num_points=args.sweep_points,
            dtype=document["network"]["state_dtype"],
        )
        document["network"]["input_voltages"] = inputs.tolist()
        document["provenance"]["generation_overrides"] = [
            {"pointer": "/network/input_voltages", "value": inputs.tolist()}
        ]
        validate_document(
            document,
            "small-network-v2.schema.json",
            repo_root=PACK_ROOT,
        )
        generated = args.write_config.expanduser()
        if not generated.is_absolute():
            generated = Path.cwd() / generated
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(pretty_json_text(document), encoding="utf-8")
        run_config = generated
        print(f"generated_config: {generated}")

    result = simulate_small_network(run_config, repo_root=PACK_ROOT)
    print(f"input_voltage_sets: {result.output_voltages.shape[0]}")
    for index, voltages in enumerate(result.hidden_voltages, start=1):
        print(f"hidden_voltages[{index}]:")
        print(voltages)
    print("output_voltages:")
    print(result.output_voltages)
    print(f"converged: {result.converged}")
    print(f"sweeps: {result.sweeps}")
    print(f"final_max_voltage_change: {result.final_max_voltage_change:.9g}")
    print(f"config_sha256: {result.receipt['resolved_config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
