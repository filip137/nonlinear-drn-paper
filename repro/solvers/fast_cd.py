"""CPU float64 red-black CD with compiled Lambert-W coordinate updates.

Ported from the FastCD used in the revised-paper experiments. The double
update uses the dominant exponential; Newton evaluates both branches.
"""
import cmath
import math

import numpy as np

from ._kernels import load_kernels
from .solver import DoubleShockley, LayeredDRN

B_CLAMP = 1e6  # |b| clamp (A); never binds on the paper networks
DOUBLE_STEPS, DOUBLE_TOL = 7, 1e-6  # polish cap; A per node (polish=True only)
SINGLE_STEPS, SINGLE_ABS, SINGLE_REL = 8, 1e-6, 1e-6

PATHS = ("off", "small", "warm", "halley", "series", "polish_steps")
EPS = float(np.finfo(np.float64).eps)


class AdaptiveRelaxation:
    """Estimate SOR relaxation from the last three sweep updates (two-term Young estimator)."""

    def __init__(self, initial=1.0, maximum=1.95, settle=3, agreement=0.05, step=0.01):
        if (
            not np.isfinite([initial, maximum, agreement, step]).all()
            or not 0 < initial <= maximum < 2
            or agreement < 0
            or step < 0
            or not isinstance(settle, int)
            or settle < 3
        ):
            raise ValueError(
                "Invalid adaptive relaxation settings (settle must be at least 3)."
            )
        self.initial, self.maximum, self.settle, self.agreement, self.step = (
            initial,
            maximum,
            settle,
            agreement,
            step,
        )
        self.reset()

    def reset(self):
        self.omega, self.updates, self.since_change, self.last_mu2 = (
            self.initial,
            [],
            0,
            None,
        )

    def mu2(self):
        d0, d1, d2 = self.updates
        g00, g01, g11 = float(d0 @ d0), float(d0 @ d1), float(d1 @ d1)
        if g00 == 0.0 or g11 == 0.0:
            return None
        det = g00 * g11 - g01 * g01
        if det <= 1e-10 * g00 * g11:
            lam = complex(float(d2 @ d1) / g11)
        else:
            r1, r0 = float(d2 @ d1), float(d2 @ d0)
            alpha, beta = (r1 * g00 - r0 * g01) / det, (r0 * g11 - r1 * g01) / det
            root = cmath.sqrt(alpha * alpha + 4.0 * beta)
            lam = max((alpha + root) / 2.0, (alpha - root) / 2.0, key=abs)
        if not 0.0 < abs(lam) < 1.0:
            return None
        return abs(lam + self.omega - 1.0) ** 2 / (self.omega * self.omega * abs(lam))

    def observe(self, update):
        self.updates = self.updates[-2:] + [update]
        self.since_change += 1
        if self.since_change < self.settle:
            return
        mu2 = self.mu2()
        last, self.last_mu2 = self.last_mu2, mu2
        if (
            mu2 is None
            or last is None
            or not 0.0 <= mu2 < 1.0
            or abs(mu2 - last) > self.agreement * max(1.0 - mu2, 1e-3)
        ):
            return
        new = max(1.0, min(2.0 / (1.0 + math.sqrt(1.0 - mu2)), self.maximum))
        if abs(new - self.omega) > self.step:
            self.omega, self.since_change, self.last_mu2 = new, 0, None


class FastCD:
    """weights[0] is input -> layer 0, weights[k] is layer k-1 -> k (nonnegative). Batch 1."""

    def __init__(
        self,
        weights,
        law,
        av=1.0,
        ai=1.0,
        *,
        small=True,
        warm=True,
        polish=False,
        physical=None,
    ):
        if law not in ("single", "double"):
            raise ValueError("FastCD law must be 'single' or 'double'.")
        device = DoubleShockley(**(physical or {}))
        if device.saturation_current <= 0:
            raise ValueError("FastCD requires positive saturation current.")
        circuit = LayeredDRN(weights, device, av, ai)
        self.W = list(circuit.weights)
        Is, vt, voff = (
            device.saturation_current,
            device.thermal_voltage,
            device.offset_voltage,
        )
        self.kernels = load_kernels()
        self.law, self.av, self.ai, self.L = law, float(av), float(ai), len(self.W)
        self.small, self.warm, self.polish = int(small), int(warm), int(polish)
        widths = [w.shape[1] for w in self.W]
        self.two_a = [
            w.sum(0)
            + (self.av * self.ai * self.W[k + 1].sum(1) if k + 1 < self.L else 0.0)
            for k, w in enumerate(self.W)
        ]
        if law == "double":
            self.lp = [np.log(Is / (d * vt)) for d in self.two_a]
        else:
            self.lp = [math.log(Is) - np.log(d) - math.log(vt) for d in self.two_a]
        self.sign = [np.where(np.arange(n) < n // 2, 1.0, -1.0) for n in widths]
        # Flat buffers with one view per layer: states, new values, b, last W per node.
        size = sum(widths)
        self.v, self.out, self.bflat, self.wflat = (np.zeros(size) for _ in range(4))
        self.old, self.d, self.scratch = (np.zeros(size) for _ in range(3))
        bounds = np.r_[0, np.cumsum(widths)]
        views = lambda flat: [
            flat[s:e] for s, e in zip(bounds[:-1], bounds[1:])
        ]  # noqa: E731
        self.z, self.new, self.b, self.w_store = (
            views(self.v),
            views(self.out),
            views(self.bflat),
            views(self.wflat),
        )
        self.counts = np.zeros(len(PATHS), dtype=np.int64)
        self.minus_two_a_out = -self.two_a[-1]
        self.args = []
        for k in range(self.L - 1):
            p = lambda array: array.ctypes.data  # noqa: E731
            if law == "double":
                self.args.append(
                    (
                        1,
                        widths[k],
                        p(self.two_a[k]),
                        p(self.lp[k]),
                        p(self.b[k]),
                        p(self.z[k]),
                        p(self.new[k]),
                        p(self.w_store[k]),
                        Is,
                        vt,
                        voff,
                        B_CLAMP,
                        self.small,
                        self.warm,
                        self.polish,
                        DOUBLE_STEPS,
                        DOUBLE_TOL,
                    )
                )
            else:
                self.args.append(
                    (
                        1,
                        widths[k],
                        p(self.two_a[k]),
                        p(self.lp[k]),
                        p(self.b[k]),
                        p(self.sign[k]),
                        p(self.z[k]),
                        p(self.new[k]),
                        p(self.w_store[k]),
                        Is,
                        vt,
                        voff,
                        B_CLAMP,
                        self.small,
                        self.warm,
                        self.polish,
                        SINGLE_STEPS,
                        SINGLE_ABS,
                        SINGLE_REL,
                    )
                )
        self.kernel = (
            self.kernels.double_update
            if law == "double"
            else self.kernels.single_update
        )
        self.counts_ptr = self.counts.ctypes.data
        self.converged = False
        self.iterations_performed = 0
        self.final_max_voltage_change = self.final_reference_scale = 0.0
        self.reason = "not_started"

    def _coupling(self, k, feed):
        """b_k = -(c_k v_{k-1} W_k + ai W_{k+1} v_{k+1}); feed = x W_0 (the input term, fixed per input)."""
        t = self.b[k]
        if k == 0:
            np.copyto(t, feed)
        else:
            np.matmul(self.z[k - 1], self.W[k], out=t)
            t *= self.av
        if k + 1 < self.L:
            t += self.ai * (self.W[k + 1] @ self.z[k + 1])
        np.negative(t, out=t)

    def _update(self, k, omega):
        if k + 1 < self.L:
            code = self.kernel(*self.args[k], omega, self.counts_ptr)
            if code:
                raise FloatingPointError(f"node kernel returned {code} on layer {k}")
        else:  # linear output layer
            new, z = self.new[k], self.z[k]
            np.divide(self.b[k], self.minus_two_a_out, out=new)
            if omega != 1.0:
                new -= z
                new *= omega
                new += z
        np.copyto(self.z[k], self.new[k])

    def solve(self, feed, relaxation, *, rtol=1e-5, atol=1e-6, max_sweeps=512, stop=None):
        """From zeros to the voltage-change stop. relaxation: a fixed omega or an
        AdaptiveRelaxation. Returns the sweep count; the states are in self.z.

        An optional stop(state, max_voltage_change) replaces the voltage-change test.
        Check ``converged``: reaching max_sweeps does not imply convergence.
        Instances own mutable state and should not be shared by concurrent solves.
        """
        feed = np.asarray(feed, dtype=np.float64)
        if feed.shape != self.z[0].shape or not np.isfinite(feed).all():
            raise ValueError(
                "Expected one finite first-layer forcing vector (input @ weights[0])."
            )
        if (
            not isinstance(max_sweeps, int)
            or max_sweeps < 0
            or not np.isfinite([rtol, atol]).all()
            or min(rtol, atol) < 0
        ):
            raise ValueError("Invalid CD convergence settings.")
        adaptive = relaxation if isinstance(relaxation, AdaptiveRelaxation) else None
        if adaptive is None and (not np.isfinite(relaxation) or not 0 < relaxation < 2):
            raise ValueError("Fixed relaxation must satisfy 0 < omega < 2.")
        if adaptive is not None:
            adaptive.reset()
        self.v.fill(0.0)
        self.wflat.fill(0.0)
        self.converged = False
        self.reason = "maximum_sweeps"
        self.iterations_performed = 0
        self.final_max_voltage_change = self.final_reference_scale = 0.0
        # A current-only criterion may already hold at zero. Voltage change
        # is undefined until the first sweep.
        if stop is not None and stop(self.v, math.inf):
            self.converged = True
            self.reason = "converged"
            return 0
        order = [*range(0, self.L, 2), *range(1, self.L, 2)]
        for sweep in range(1, max_sweeps + 1):
            omega = adaptive.omega if adaptive is not None else relaxation
            np.copyto(self.old, self.v)
            for k in order:
                self._coupling(k, feed)
                self._update(k, omega)
            np.subtract(self.v, self.old, out=self.d)
            change = float(np.abs(self.d, out=self.scratch).max())
            scale = float(np.abs(self.old, out=self.scratch).max())
            self.iterations_performed = sweep
            self.final_max_voltage_change, self.final_reference_scale = change, scale
            if not np.isfinite(change):
                self.reason = "nonfinite_state"
                break
            if adaptive is not None and change > 64.0 * EPS * scale:
                adaptive.observe(self.d.copy())
            reached = change <= rtol * scale + atol if stop is None else stop(self.v, change)
            if reached:
                self.converged = True
                self.reason = "converged"
                break
        return self.iterations_performed

    @property
    def state(self):
        return self.v.copy()
