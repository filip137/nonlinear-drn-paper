"""Full oriented single-Shockley law for the existing equilibrium solvers.

First half of each hidden layer: s=+1; second half: s=-1.
i(v) = s Is expm1((s v - Voff)/Vt). Outputs have no diode.
No clipping or large-signal approximation is applied here.
"""
from dataclasses import dataclass

import numpy as np

from .solver import DoubleShockley, LayeredDRN


def orientations(width):
    return np.where(np.arange(width) < width // 2, 1.0, -1.0)


@dataclass(frozen=True)
class SingleShockley(DoubleShockley):
    signs: tuple = (1.0,)

    def __post_init__(self):
        super().__post_init__()
        if not self.signs or not all(s in (-1, 1) for s in self.signs):
            raise ValueError(
                "Single-diode orientations must be a nonempty sequence of +/-1."
            )

    def current_derivative(self, v):
        if self.saturation_current == 0:
            return np.zeros_like(v), np.zeros_like(v)
        signs = np.asarray(self.signs, dtype=v.dtype)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            exponent = (signs * v - self.offset_voltage) / self.thermal_voltage
            current = signs * self.saturation_current * np.expm1(exponent)
            derivative = (self.saturation_current / self.thermal_voltage) * np.exp(
                exponent
            )
        return current, derivative

    def coordinate_root(self, diagonal, forcing):
        if self.saturation_current == 0:
            return forcing / diagonal
        signs = np.asarray(self.signs, dtype=forcing.dtype)
        drive = signs * forcing
        Is, off, vt = self.saturation_current, self.offset_voltage, self.thermal_voltage
        # In u=s*v coordinates the equation is d*u+Is*expm1((u-off)/vt)=drive.
        # F(off)=d*off-drive gives a bracket without overflowing trial exponentials.
        lo = np.minimum(drive / diagonal, off)
        hi = np.minimum(
            (drive + Is) / diagonal,
            off + vt * np.log1p(np.maximum(drive - diagonal * off, 0.0) / Is),
        )
        value = hi.copy()
        done = np.zeros_like(value, dtype=bool)
        for _ in range(100):
            exponent = (value - off) / vt
            residual = diagonal * value - drive + Is * np.expm1(exponent)
            derivative = diagonal + (Is / vt) * np.exp(exponent)
            correction = residual / derivative
            close = np.abs(correction) <= 4 * np.finfo(value.dtype).eps * np.maximum(
                1.0, np.abs(value)
            )
            lo = np.where((residual < 0) & ~done, value, lo)
            hi = np.where((residual > 0) & ~done, value, hi)
            candidate = value - correction
            candidate = np.where(
                (candidate > lo) & (candidate < hi), candidate, (lo + hi) / 2
            )
            candidate = np.where(close, value - correction, candidate)
            value = np.where(done, value, candidate)
            done |= close
            if np.all(done):
                return signs * value
        raise RuntimeError("Single-Shockley coordinate solve failed to converge.")


class SingleDiodeDRN(LayeredDRN):
    def __init__(
        self,
        weights,
        physical,
        voltage_gain=1.0,
        current_gain=1.0,
        bias_currents=None,
        *,
        dtype=np.float64,
    ):
        # Validate dimensions before constructing node orientations.
        weights = LayeredDRN(
            weights,
            DoubleShockley(**physical),
            voltage_gain,
            current_gain,
            bias_currents,
            dtype=dtype,
        ).weights
        layer_signs = [orientations(w.shape[1]) for w in weights[:-1]]
        self.layer_laws = [
            SingleShockley(**physical, signs=tuple(s)) for s in layer_signs
        ]
        law = SingleShockley(
            **physical,
            signs=tuple(np.concatenate(layer_signs)) if layer_signs else (1.0,),
        )
        super().__init__(
            weights, law, voltage_gain, current_gain, bias_currents, dtype=dtype
        )

    def coordinate_root(self, layer_index, diagonal, forcing):
        return self.layer_laws[layer_index].coordinate_root(diagonal, forcing)
