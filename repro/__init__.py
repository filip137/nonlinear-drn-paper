"""Public experiment interface for the nonlinear-DRN paper artifact."""

from repro.runner import DRNRunSpec, build_training_config, run_drn
from repro.small_network import (
    DEFAULT_SHOCKLEY_PARAMETERS,
    SmallNetworkResult,
    simulate_small_network,
)


__all__ = [
    "DEFAULT_SHOCKLEY_PARAMETERS",
    "DRNRunSpec",
    "SmallNetworkResult",
    "build_training_config",
    "run_drn",
    "simulate_small_network",
]
