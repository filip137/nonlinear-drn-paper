"""Equilibrium solvers used in the paper revisions (CPU, one input per solve)."""

from .fast_cd import AdaptiveRelaxation, FastCD
from .newton_cached import BlockCholesky, CachedResult, newton_cached
from .pwl import PWLCoordinateDescent, TableLaw
from .single_diode_solver import SingleDiodeDRN, SingleShockley
from .solver import DoubleShockley, LayeredDRN, SolveResult, coordinate_descent, newton
from .torch_cd import TorchCD

__all__ = [
    "AdaptiveRelaxation",
    "BlockCholesky",
    "CachedResult",
    "DoubleShockley",
    "FastCD",
    "LayeredDRN",
    "PWLCoordinateDescent",
    "SingleDiodeDRN",
    "SingleShockley",
    "SolveResult",
    "TableLaw",
    "TorchCD",
    "coordinate_descent",
    "newton",
    "newton_cached",
]
