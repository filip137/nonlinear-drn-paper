"""Validation-only coordinate-descent minimizers for the digits repro pack.

This is a paper-runtime subset of labs/custom_minimizer.py. It keeps the
updaters needed to train and validate the three bundled nonlinearities:
single-diode exponential, double-diode exponential, and measured/PWL I-V.
Anderson acceleration, timing instrumentation, and exploratory nonlinearities
are intentionally omitted.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from model.minimizer.minimizer import LayerUpdater, Minimizer
from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.minimizer import (
    ExponentialDoubleDiodeUpdater,
    ExponentialSingleDiodeUpdater,
    lambertw,
)

DEFAULT_IV_CURVE_PATH = None

NONLINEARITY_ALIASES = {
    "doublediodeexponential": "double_diode_exponential",
    "singlediodeexponential": "single_diode_exponential",
    "experimental": "experimental",
}

DOUBLE_DIODE_UPDATER_ALIASES = {
    "custom": "float64",
    "float64": "float64",
    "float32": "float32",
    "float64overrelaxed": "float64_overrelaxed",
    "float32overrelaxed": "float32_overrelaxed",
    "overrelaxed": "float32_overrelaxed",
    "overrelated": "float32_overrelaxed",
}

SINGLE_DIODE_UPDATER_ALIASES = {
    "custom": "custom",
    "standard": "standard",
    "overrelaxed": "overrelaxed",
    "overrelated": "overrelaxed",
}

EXPERIMENTAL_UPDATER_ALIASES = {
    "standard": "standard",
    "custom": "standard",
    "overrelaxed": "overrelaxed",
    "overrelated": "overrelaxed",
}


class NonFiniteDiodeError(FloatingPointError):
    """Raised when a diode update produces non-finite voltages."""


def _resolve_b_clip_from_env() -> float | None:
    value = os.environ.get("DRN_B_CLAMP")
    return float(value) if value is not None else None


def _canonical_name(value: object, aliases: dict[str, str], *, field: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    compact_key = key.replace("_", "")
    if compact_key in aliases:
        return aliases[compact_key]
    expected = ", ".join(sorted(set(aliases.values())))
    raise ValueError(f"Expected {field} to be one of {expected}. Provided value: {value!r}.")


@dataclass
class MinimizerSettings:
    rel_tol: float
    vn_tol: float
    use_polish: bool
    max_newton_iters: int
    z_thresh: float
    exp_clip: float
    experimental_newton_tol: float = 1e-5

    def __post_init__(self):
        self.rel_tol = float(self.rel_tol)
        self.vn_tol = float(self.vn_tol)
        self.use_polish = bool(self.use_polish)
        self.max_newton_iters = int(self.max_newton_iters)
        self.z_thresh = float(self.z_thresh)
        self.exp_clip = float(self.exp_clip)
        self.experimental_newton_tol = float(self.experimental_newton_tol)
        positive_fields = {
            "rel_tol": self.rel_tol,
            "vn_tol": self.vn_tol,
            "exp_clip": self.exp_clip,
            "experimental_newton_tol": self.experimental_newton_tol,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Expected {name} to be finite and positive. "
                    f"Provided value: {value!r}."
                )
        if not math.isfinite(self.z_thresh) or self.z_thresh <= 1.0:
            raise ValueError(
                "Expected z_thresh to be finite and greater than 1 for the large-z "
                "Lambert-W expansion. "
                f"Provided value: {self.z_thresh!r}."
            )
        if self.max_newton_iters < 0:
            raise ValueError(
                "Expected max_newton_iters to be non-negative. "
                f"Provided value: {self.max_newton_iters!r}."
            )


def _load_iv_data(path: Path) -> torch.Tensor:
    data = np.load(path)
    if "iv" in data:
        iv = data["iv"]
    elif "i" in data and "v" in data:
        iv = np.stack([data["i"], data["v"]], axis=0)
    else:
        raise ValueError(f"{path} must contain 'iv' or both 'i' and 'v' arrays.")
    return torch.as_tensor(iv)


class CustomMinimizer(Minimizer):
    def __init__(
        self,
        fn,
        updaters,
        num_iterations,
        mode,
        voltage_amp,
        current_amp,
        adaptive_equilibrium,
        *,
        settings: MinimizerSettings,
    ):
        super().__init__(fn, updaters, num_iterations, mode, voltage_amp, current_amp)
        self._adaptive_equilibrium = bool(adaptive_equilibrium)
        self._force_fixed_iterations = any(
            type(updater) is ExponentialSingleDiodeUpdater for updater in updaters
        )
        self._settings = settings

    @staticmethod
    def _max_abs_delta(new_state, old_state):
        return float((new_state - old_state).abs().max().item())

    def _step_group(self, layer_group):
        if not layer_group:
            return
        self.step(layer_group)

    def _step_odd_even(self, layer_group_odd, layer_group_even):
        if not layer_group_odd and not layer_group_even:
            return
        if not layer_group_odd:
            self._step_group(layer_group_even)
            return
        if not layer_group_even:
            self._step_group(layer_group_odd)
            return

        self.step(layer_group_odd)
        self.step(layer_group_even)

    def compute_equilibrium(self):
        """Compute the minimum of the function wrt the free layers

        Performs num_iterations iterations of the minimization process.

        Returns:
            layers: dictionary of Tensors. The state of the layers at equilibrium
        """

        max_num_of_iterations = len(self._list_layers) // 2

        use_adaptive = self._adaptive_equilibrium and not self._force_fixed_iterations
        if not use_adaptive:
            # Original fixed-iteration behaviour: no tolerance-based early stop.
            if len(self._list_layers) >= 2:
                layer_group_odd, layer_group_even = self._list_layers[0], self._list_layers[1]
                for _ in range(max_num_of_iterations):
                    self._step_odd_even(
                        layer_group_odd,
                        layer_group_even,
                    )
            else:
                for layer_group in self._list_layers:
                    self._step_group(layer_group)
        else:
            rtol = self._settings.rel_tol
            vntol = self._settings.vn_tol

            for _ in range(max_num_of_iterations):
                layer_group_odd, layer_group_even = self._list_layers[0], self._list_layers[1]

                odd_old = [u._layer.state.detach().clone() for u in layer_group_odd]
                even_old = [u._layer.state.detach().clone() for u in layer_group_even]

                self._step_odd_even(
                    layer_group_odd,
                    layer_group_even,
                )

                inf_delta = 0.0
                inf_ref = 0.0
                for updater, old_state in zip(layer_group_odd, odd_old):
                    inf_delta = max(inf_delta, self._max_abs_delta(updater._layer.state, old_state))
                    inf_ref = max(inf_ref, float(old_state.abs().max().item()))
                for updater, old_state in zip(layer_group_even, even_old):
                    inf_delta = max(inf_delta, self._max_abs_delta(updater._layer.state, old_state))
                    inf_ref = max(inf_ref, float(old_state.abs().max().item()))

                # Adaptive stopping follows the same voltage infinity-norm test
                # used to choose the bundled validation iteration budgets.
                threshold = rtol * inf_ref + vntol
                if inf_delta <= threshold:
                    break

        layers = {layer.name: layer.state for layer in self._layers}

        return layers


class Float64ExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Float64 exponential double-diode coordinate updater."""

    def __init__(self, layer, fn, diode_params, settings: MinimizerSettings):
        super().__init__(layer, fn, diode_params)
        self._settings = settings
        self._b_clip = _resolve_b_clip_from_env()

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            v = self.lambert_hidden(
                a,
                b,
                self._Is,
                self._v_off,
                self._Vt,
                z_thresh=self._settings.z_thresh,
                polish=self._settings.use_polish,
                exp_clip=self._settings.exp_clip,
                max_inner_iter=self._settings.max_newton_iters,
            )
            return v
        return -b / (2.0 * a)

    def lambertw_large(self, z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w += L2 / L1
        if terms >= 3:
            w += (L2 * (-2 + L2)) / (2 * L1**2)
        if terms >= 4:
            w += (L2 * (6 - 9 * L2 + 2 * L2**2)) / (6 * L1**3)
        return w

    def lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh, polish, exp_clip, max_inner_iter):
        out_dtype = a.dtype
        device = a.device

        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)
        if getattr(self, "_b_clip", None) is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vt, dtype=torch.float64, device=device)
        v_off64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)

        A = torch.where(
            b64 <= 0,
            (b64 - I_s64) / (2.0 * a64),
            (b64 + I_s64) / (2.0 * a64),
        )

        exp_arg_add = (-(A + v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)

        z_add = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_add)
        z_rev = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_rev)
        z = torch.where(b64 > 0, z_rev, z_add)

        use_asym = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=torch.float64, device=z.device)

        if small_mask.any():
            z_small = torch.clamp(z[small_mask].to(torch.float64), max=float(z_thresh))
            W0_small = lambertw(z_small).real.to(W0.dtype)
            W0[small_mask] = W0_small

        if use_asym.any():
            z_large = torch.clamp(z[use_asym].to(torch.float64), min=1.0)
            W0_large = self.lambertw_large(z_large).to(torch.float64)
            W0[use_asym] = W0_large

        x = torch.where(b64 > 0, vt64 * W0 - A, -vt64 * W0 - A)

        if polish:
            current_tol = 1e-6 * x.shape[0]
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for _ in range(1, max_inner_iter):
                mask_pos = b64 <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]
                    ap64 = a64[mask_pos]
                    bp64 = b64[mask_pos]
                    ep_arg = ((xp - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos] = 2 * ap64 * xp + (bp64 - I_s) + I_s * ep
                    df[mask_pos] = 2 * ap64 + (I_s / vt) * ep

                if mask_neg.any():
                    xn = x[mask_neg]
                    an64 = a64[mask_neg]
                    bn64 = b64[mask_neg]
                    en_arg = ((-xn - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg] = 2 * an64 * xn + (bn64 + I_s) - I_s * en
                    df[mask_neg] = 2 * an64 + (I_s / vt) * en

                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    break

                x = x - f / df
                if torch.norm(f) < current_tol:
                    break

        return x.to(out_dtype)


class Float32ExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Float32 double-diode updater with optional SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        settings: MinimizerSettings,
        *,
        overrelaxation_factor: float = 1.0,
    ):
        super().__init__(layer, fn, diode_params)
        self._settings = settings
        self._b_clip = _resolve_b_clip_from_env()
        self._overrelaxation_factor = float(overrelaxation_factor)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)

        if isinstance(self._layer, NonlinearResistiveLayer):
            v = self.lambert_hidden(
                a,
                b,
                self._Is,
                self._v_off,
                self._Vt,
                z_thresh=self._settings.z_thresh,
                polish=self._settings.use_polish,
                exp_clip=self._settings.exp_clip,
                max_inner_iter=self._settings.max_newton_iters,
            )
        else:
            v = -b / (2.0 * a)

        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v
        return self._layer.state + omega * (v - self._layer.state)

    def lambertw_large(self, z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w += L2 / L1
        if terms >= 3:
            w += (L2 * (-2 + L2)) / (2 * L1**2)
        if terms >= 4:
            w += (L2 * (6 - 9 * L2 + 2 * L2**2)) / (6 * L1**3)
        return w

    def lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh, polish, exp_clip, max_inner_iter):
        out_dtype = a.dtype
        device = a.device

        # Keep the updater in float32 and only run Lambert-W evaluation in float64.
        work_dtype = torch.float32
        a_work = a.to(work_dtype)
        b_work = b.to(work_dtype)
        if getattr(self, "_b_clip", None) is not None:
            b_work = b_work.clamp(min=-self._b_clip, max=self._b_clip)
        I_s_work = torch.as_tensor(I_s, dtype=work_dtype, device=device)
        vt_work = torch.as_tensor(vt, dtype=work_dtype, device=device)
        v_off_work = torch.as_tensor(v_off, dtype=work_dtype, device=device)

        A = torch.where(
            b_work <= 0,
            (b_work - I_s_work) / (2.0 * a_work),
            (b_work + I_s_work) / (2.0 * a_work),
        )

        exp_arg_add = (-(A + v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
        z_add = (I_s_work / (2.0 * a_work * vt_work)) * torch.exp(exp_arg_add)
        z_rev = (I_s_work / (2.0 * a_work * vt_work)) * torch.exp(exp_arg_rev)
        z = torch.where(b_work > 0, z_rev, z_add)

        use_asym = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=work_dtype, device=device)

        if small_mask.any():
            z_small = torch.clamp(z[small_mask], max=float(z_thresh)).to(torch.float64)
            W0_small = lambertw(z_small).real.to(work_dtype)
            W0[small_mask] = W0_small

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0).to(torch.float64)
            W0_large = self.lambertw_large(z_large).to(work_dtype)
            W0[use_asym] = W0_large

        x = torch.where(b_work > 0, vt_work * W0 - A, -vt_work * W0 - A)

        if polish:
            current_tol = 1e-6 * x.shape[0]
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for _ in range(1, max_inner_iter):
                mask_pos = b_work <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]
                    ap_work = a_work[mask_pos]
                    bp_work = b_work[mask_pos]
                    ep_arg = ((xp - v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos] = 2 * ap_work * xp + (bp_work - I_s_work) + I_s_work * ep
                    df[mask_pos] = 2 * ap_work + (I_s_work / vt_work) * ep

                if mask_neg.any():
                    xn = x[mask_neg]
                    an_work = a_work[mask_neg]
                    bn_work = b_work[mask_neg]
                    en_arg = ((-xn - v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg] = 2 * an_work * xn + (bn_work + I_s_work) - I_s_work * en
                    df[mask_neg] = 2 * an_work + (I_s_work / vt_work) * en

                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    break

                x = x - f / df
                if torch.norm(f) < current_tol:
                    break

        return x.to(out_dtype)


class OverRelaxedFloat64DoubleDiodeUpdater(Float64ExponentialDoubleDiodeUpdater):
    """Float64 double-diode updater with SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        settings: MinimizerSettings,
        *,
        overrelaxation_factor: float = 1.3,
    ):
        super().__init__(layer, fn, diode_params, settings)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd

        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        return v_relaxed


class CustomExponentialSingleDiodeUpdater(ExponentialSingleDiodeUpdater):
    """Single-diode updater with explicit forward/reverse orientation split."""

    def __init__(self, layer, fn, diode_params, settings: MinimizerSettings):
        super().__init__(layer, fn, diode_params)
        self._settings = settings
        self._b_clip = _resolve_b_clip_from_env()

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)

        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

        # The bundled single-diode checkpoints split each free layer into
        # forward-oriented nodes followed by reverse-oriented nodes.
        width = b.shape[1]
        reverse_mask_nodes = torch.arange(width, device=b.device) >= (width // 2)
        reverse_mask = reverse_mask_nodes.unsqueeze(0).expand_as(b)

        v_fwd = self.lambert_single_forward(
            a,
            b,
            self._Is,
            self._v_off,
            self._Vt,
            z_thresh=self._settings.z_thresh,
            polish=self._settings.use_polish,
            abs_tol=1e-6,
            rel_tol=1e-6,
            exp_clip=self._settings.exp_clip,
            a_min=1e-30,
            max_inner_iter=self._settings.max_newton_iters,
        )

        v_rev = self.lambert_single_reverse(
            a,
            b,
            self._Is,
            self._v_off,
            self._Vt,
            z_thresh=self._settings.z_thresh,
            polish=self._settings.use_polish,
            abs_tol=1e-6,
            rel_tol=1e-6,
            exp_clip=self._settings.exp_clip,
            a_min=1e-30,
            max_inner_iter=self._settings.max_newton_iters,
        )

        return torch.where(reverse_mask, v_rev, v_fwd)

    def lambertw_large(self, z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w = w + L2 / L1
        if terms >= 3:
            w = w + (L2 * (-2.0 + L2)) / (2.0 * L1**2)
        if terms >= 4:
            w = w + (L2 * (6.0 - 9.0 * L2 + 2.0 * L2**2)) / (6.0 * L1**3)
        return w

    def _lambertw0(self, z):
        if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
            return torch.special.lambertw(z).real
        return lambertw(z).real

    def lambert_single_forward(
        self,
        a,
        b,
        I_s,
        v_on,
        vT,
        z_thresh,
        polish,
        abs_tol,
        rel_tol,
        exp_clip,
        a_min,
        max_inner_iter,
    ):
        """Solve ``2av + b + Is(exp((v - v_on) / vT) - 1) = 0``."""
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)
        if self._b_clip is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)

        i_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vT, dtype=torch.float64, device=device)
        v_on64 = torch.as_tensor(v_on, dtype=torch.float64, device=device)

        a_shift = (b64 - i_s64) / (2.0 * a64)
        exp_arg = (-(a_shift + v_on64) / vt64).clamp(max=exp_clip)
        z = (i_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg)
        if not torch.isfinite(z).all():
            raise NonFiniteDiodeError("CustomExponentialSingleDiodeUpdater: non-finite z in lambert_single_forward.")

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)

        x = -a_shift - vt64 * W0

        if polish:
            for _ in range(max_inner_iter):
                arg = (x - v_on64) / vt64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f = 2.0 * a64 * x + b64 + i_s64 * (e - 1.0)
                df = 2.0 * a64 + (i_s64 / vt64) * e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    raise NonFiniteDiodeError(
                        "CustomExponentialSingleDiodeUpdater: non-finite f/df in lambert_single_forward."
                    )

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(
                    2.0 * torch.abs(a64) * torch.abs(x)
                    + torch.abs(b64)
                    + torch.abs(i_s64)
                    + 1.0
                )
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break

        return x.to(out_dtype)

    def lambert_single_reverse(
        self,
        a,
        b,
        I_s,
        v_on,
        vT,
        z_thresh,
        polish,
        abs_tol,
        rel_tol,
        exp_clip,
        a_min,
        max_inner_iter,
    ):
        """Solve ``2av + b - Is(exp(-(v + v_on) / vT) - 1) = 0``."""
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)
        if self._b_clip is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)

        i_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vT, dtype=torch.float64, device=device)
        v_on64 = torch.as_tensor(v_on, dtype=torch.float64, device=device)

        a_shift = (b64 + i_s64) / (2.0 * a64)
        exp_arg = ((-v_on64 + a_shift) / vt64).clamp(min=-exp_clip, max=exp_clip)
        z = (i_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg)
        if not torch.isfinite(z).all():
            raise NonFiniteDiodeError("CustomExponentialSingleDiodeUpdater: non-finite z in lambert_single_reverse.")

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)

        x = -a_shift + vt64 * W0

        if polish:
            for _ in range(max_inner_iter):
                arg = -(x + v_on64) / vt64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f = 2.0 * a64 * x + b64 + i_s64 - i_s64 * e
                df = 2.0 * a64 + (i_s64 / vt64) * e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    raise NonFiniteDiodeError(
                        "CustomExponentialSingleDiodeUpdater: non-finite f/df in lambert_single_reverse."
                    )

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(
                    2.0 * torch.abs(a64) * torch.abs(x)
                    + torch.abs(b64)
                    + torch.abs(i_s64)
                    + 1.0
                )
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break

        return x.to(out_dtype)


class OverRelaxedSingleDiodeUpdater(CustomExponentialSingleDiodeUpdater):
    """Single-diode updater with SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        settings: MinimizerSettings,
        *,
        overrelaxation_factor: float = 1.3,
    ):
        super().__init__(layer, fn, diode_params, settings)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd
        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        return v_relaxed


class ExperimentalIVCurveUpdater(LayerUpdater):
    """Layer updater for measured piecewise-linear I-V curves."""

    def __init__(
        self,
        layer,
        fn,
        iv_data,
        *,
        damping: float = 0.5,
        max_newton_steps: int = 100,
        newton_tol: float = 1e-5,
        clamp: bool = True,
    ):
        super().__init__(layer, fn)
        self._iv_data = iv_data
        self._damping = float(damping)
        self._max_newton_steps = int(max_newton_steps)
        self._newton_tol = float(newton_tol)
        if not math.isfinite(self._damping) or self._damping <= 0.0:
            raise ValueError(
                "Expected damping to be finite and positive. "
                f"Provided value: {self._damping!r}."
            )
        if self._max_newton_steps < 1:
            raise ValueError(
                "Expected experimental_newton_max_steps to be at least 1. "
                f"Provided value: {self._max_newton_steps!r}."
            )
        if not math.isfinite(self._newton_tol) or self._newton_tol <= 0.0:
            raise ValueError(
                "Expected experimental_newton_tol to be finite and positive. "
                f"Provided value: {self._newton_tol!r}."
            )
        extrapolation = os.environ.get("LABS_IV_EXTRAPOLATION", "clamp").strip().lower()
        if extrapolation not in {"clamp", "linear"}:
            extrapolation = "clamp"
        # "linear" => allow extrapolation beyond v_min/v_max using end-segment slopes.
        self._clamp = bool(clamp) and extrapolation != "linear"
        self._extrapolation = extrapolation
        # Standard quadratic coefficients (matching QuadraticUpdater interface)
        self._a = fn.a_coef_fn(layer)
        self._b = fn.b_coef_fn(layer)

    def pre_activate(self):
        b = self._b()
        a = self._a()
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)
        i_data = self._iv_data[0].to(device=a.device, dtype=a.dtype)
        v_data = self._iv_data[1].to(device=a.device, dtype=a.dtype)
        slope = (i_data[1:] - i_data[:-1]) / (v_data[1:] - v_data[:-1])
        v_min = v_data[0]
        v_max = v_data[-1]
        eps = torch.finfo(a.dtype).eps

        def newton_step(v):
            idx = (torch.bucketize(v, v_data) - 1).clamp(0, len(v_data) - 2)
            deriv = slope[idx]
            i = i_data[idx] + (v - v_data[idx]) * deriv
            fun = 2 * a * v + b + i
            fun_prime = 2 * a + deriv + eps
            v_new = v - self._damping * (fun / fun_prime)
            if self._clamp:
                v_new = torch.clamp(v_new, min=v_min, max=v_max)
            return v_new

        max_number_steps = self._max_newton_steps
        v_old = -b / (2 * a)
        tol = self._newton_tol
        active = torch.ones_like(v_old, dtype=torch.bool)
        for _ in range(max_number_steps):
            v = newton_step(v_old)
            delta = torch.abs(v - v_old)
            converged = delta < tol
            active = active & ~converged
            v_old = v
            if not active.any().item():
                break
        return v_old


class OverRelaxedExperimentalIVCurveUpdater(ExperimentalIVCurveUpdater):
    """Experimental I-V curve updater with SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        iv_data,
        *,
        overrelaxation_factor: float = 1.3,
        damping: float = 0.5,
        max_newton_steps: int = 100,
        newton_tol: float = 1e-5,
        clamp: bool = True,
    ):
        super().__init__(
            layer,
            fn,
            iv_data,
            damping=damping,
            max_newton_steps=max_newton_steps,
            newton_tol=newton_tol,
            clamp=clamp,
        )
        self._overrelaxation_factor = float(overrelaxation_factor)
        v_vals = self._iv_data[1]
        self._v_min = float(v_vals.min().item())
        self._v_max = float(v_vals.max().item())

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd
        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        if self._clamp:
            v_relaxed = torch.clamp(v_relaxed, min=self._v_min, max=self._v_max)
        return v_relaxed


class CustomQuadraticMinimizer(CustomMinimizer):
    """Digits-core minimizer for the three paper nonlinearities."""

    def __init__(
        self,
        fn,
        free_layers,
        num_iterations,
        mode,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        voltage_amp,
        current_amp,
        iv_data,
        iv_data_path,
        double_diode_updater,
        adaptive_equilibrium,
        overrelaxation_factor,
        single_diode_updater,
        *,
        minimizer_settings: MinimizerSettings,
        damping: float = 0.5,
        experimental_newton_max_steps: int = 100,
    ):
        _ = quadratic_diode_param
        exponential_params = dict(exponential_diode_param)
        overrelaxation_factor = float(overrelaxation_factor)
        if not math.isfinite(overrelaxation_factor) or overrelaxation_factor <= 0.0:
            raise ValueError(
                "Expected overrelaxation_factor to be finite and positive. "
                f"Provided value: {overrelaxation_factor!r}."
            )
        non_linearity = _canonical_name(
            non_linearity,
            NONLINEARITY_ALIASES,
            field="non_linearity",
        )

        if non_linearity == "double_diode_exponential":
            updater_name = _canonical_name(
                double_diode_updater,
                DOUBLE_DIODE_UPDATER_ALIASES,
                field="double_diode_updater",
            )
            if updater_name == "float64":
                updaters = [
                    Float64ExponentialDoubleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        minimizer_settings,
                    )
                    for layer in free_layers
                ]
            elif updater_name in {"float32", "float32_overrelaxed"}:
                omega = overrelaxation_factor if updater_name == "float32_overrelaxed" else 1.0
                updaters = [
                    Float32ExponentialDoubleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        minimizer_settings,
                        overrelaxation_factor=omega,
                    )
                    for layer in free_layers
                ]
            elif updater_name == "float64_overrelaxed":
                updaters = [
                    OverRelaxedFloat64DoubleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        minimizer_settings,
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
            else:
                raise AssertionError(updater_name)
        elif non_linearity == "single_diode_exponential":
            updater_name = _canonical_name(
                single_diode_updater,
                SINGLE_DIODE_UPDATER_ALIASES,
                field="single_diode_updater",
            )
            if updater_name == "custom":
                updaters = [
                    CustomExponentialSingleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        minimizer_settings,
                    )
                    for layer in free_layers
                ]
            elif updater_name == "standard":
                updaters = [
                    ExponentialSingleDiodeUpdater(layer, fn, exponential_params)
                    for layer in free_layers
                ]
            elif updater_name == "overrelaxed":
                updaters = [
                    OverRelaxedSingleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        minimizer_settings,
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
            else:
                raise AssertionError(updater_name)
        elif non_linearity == "experimental":
            if iv_data is None:
                env_path = os.environ.get("LABS_IV_CURVE_PATH")
                candidate_path = (
                    env_path
                    if env_path
                    else (iv_data_path if iv_data_path else DEFAULT_IV_CURVE_PATH)
                )
                if candidate_path is None:
                    raise FileNotFoundError(
                        "Expected experimental IV curve path via LABS_IV_CURVE_PATH or config "
                        "'iv_data_path' (existing .npz file). Provided value: None."
                    )
                path = Path(candidate_path)
                if not path.exists():
                    raise FileNotFoundError(
                        "Expected experimental IV curve path via LABS_IV_CURVE_PATH or config "
                        f"'iv_data_path' (existing .npz file). Provided value: {path}"
                    )
                iv_data = _load_iv_data(path)
            updater_name = _canonical_name(
                double_diode_updater,
                EXPERIMENTAL_UPDATER_ALIASES,
                field="double_diode_updater",
            )
            if updater_name == "overrelaxed":
                updaters = [
                    OverRelaxedExperimentalIVCurveUpdater(
                        layer,
                        fn,
                        iv_data,
                        overrelaxation_factor=overrelaxation_factor,
                        damping=damping,
                        max_newton_steps=experimental_newton_max_steps,
                        newton_tol=minimizer_settings.experimental_newton_tol,
                    )
                    for layer in free_layers
                ]
            elif updater_name == "standard":
                updaters = [
                    ExperimentalIVCurveUpdater(
                        layer,
                        fn,
                        iv_data,
                        damping=damping,
                        max_newton_steps=experimental_newton_max_steps,
                        newton_tol=minimizer_settings.experimental_newton_tol,
                    )
                    for layer in free_layers
                ]
            else:
                raise AssertionError(updater_name)
        else:
            raise AssertionError(non_linearity)

        super().__init__(
            fn,
            updaters,
            num_iterations,
            mode,
            voltage_amp,
            current_amp,
            adaptive_equilibrium=adaptive_equilibrium,
            settings=minimizer_settings,
        )

        for updater in self._updaters:
            updater.voltage_amp = self.voltage_amp
            updater.current_amp = self.current_amp

        self._non_linearity = non_linearity
        self._exponential_params = exponential_params
