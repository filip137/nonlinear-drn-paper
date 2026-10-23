"""Coordinate-descent minimizers for deep resistive networks.

The original quadratic updaters and the paper's configurable Shockley and
measured-I-V variants intentionally share this module.  File loading and
experiment configuration remain in ``repro`` so the model layer stays usable
on its own.
"""

import math
import os
from dataclasses import dataclass

import torch

from model.minimizer.minimizer import LayerUpdater, Minimizer
from model.resistive.layer import ConvLayer, NonlinearResistiveLayer


def _load_lambertw_backend(backend: str):
    """Load exactly the Lambert-W implementation selected by configuration."""

    if backend == "torch_special":
        if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
            return torch.special.lambertw
        raise ImportError(
            "Configured Lambert-W backend 'torch_special' is unavailable in this "
            "PyTorch build; select 'torchlambertw' or install a PyTorch build that "
            "provides torch.special.lambertw."
        )
    if backend == "torchlambertw":
        try:
            import torchlambertw.special as tw_special
        except Exception as exc:
            raise ImportError(
                "Configured Lambert-W backend 'torchlambertw' is unavailable; "
                "install the repository's torchlambertw dependency."
            ) from exc
        return tw_special.lambertw
    raise ValueError(
        "Expected lambertw_backend to be 'torchlambertw' or 'torch_special'. "
        f"Provided value: {backend!r}."
    )


class NonFiniteDiodeError(FloatingPointError):
    """Raised when a diode update produces non-finite voltages."""


def _require_finite_tensor(tensor, *, context):
    if not torch.isfinite(tensor).all():
        raise NonFiniteDiodeError(f"{context}: produced a non-finite tensor.")
    return tensor


def _require_choice(value: object, choices: tuple[str, ...], *, field: str) -> str:
    """Reject aliases, spelling normalization, and missing selectors."""

    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(repr(choice) for choice in choices)
        raise ValueError(
            f"Expected {field} to be exactly one of {expected}. "
            f"Provided value: {value!r}."
        )
    return value


@dataclass(frozen=True)
class MinimizerSettings:
    """Complete numerical policy for the paper's nonlinear updaters.

    Every field is required so constructing a minimizer cannot silently select a
    numerical default or inherit process environment state.
    """

    rel_tol: float | None
    vn_tol: float | None
    use_polish: bool | None
    max_newton_iters: int | None
    z_thresh: float | None
    exp_clip: float | None
    experimental_newton_tol: float | None
    b_clamp: float | None
    pwl_extrapolation: str | None
    pwl_nonconvergence_policy: str | None
    lambertw_backend: str | None
    lambertw_asymptotic_terms: int | None
    single_diode_min_a: float | None
    single_diode_polish_abs_tol: float | None
    single_diode_polish_rel_tol: float | None
    double_diode_polish_residual_tol: float | None

    def __post_init__(self):
        if self.use_polish is not None and not isinstance(self.use_polish, bool):
            raise ValueError(
                "Expected use_polish to be a boolean. "
                f"Provided value: {self.use_polish!r}."
            )
        numeric_fields = {
            "rel_tol": self.rel_tol,
            "vn_tol": self.vn_tol,
            "z_thresh": self.z_thresh,
            "exp_clip": self.exp_clip,
            "experimental_newton_tol": self.experimental_newton_tol,
            "b_clamp": self.b_clamp,
            "single_diode_min_a": self.single_diode_min_a,
            "single_diode_polish_abs_tol": self.single_diode_polish_abs_tol,
            "single_diode_polish_rel_tol": self.single_diode_polish_rel_tol,
            "double_diode_polish_residual_tol": self.double_diode_polish_residual_tol,
        }
        for name, value in numeric_fields.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Expected {name} to be a JSON number. Provided value: {value!r}."
                )
        if self.max_newton_iters is not None and (
            isinstance(self.max_newton_iters, bool)
            or not isinstance(self.max_newton_iters, int)
        ):
            raise ValueError(
                "Expected max_newton_iters to be an integer. "
                f"Provided value: {self.max_newton_iters!r}."
            )
        if self.lambertw_asymptotic_terms is not None and (
            isinstance(self.lambertw_asymptotic_terms, bool)
            or not isinstance(self.lambertw_asymptotic_terms, int)
        ):
            raise ValueError(
                "Expected lambertw_asymptotic_terms to be an integer. "
                f"Provided value: {self.lambertw_asymptotic_terms!r}."
            )
        positive_fields = {
            "rel_tol": self.rel_tol,
            "vn_tol": self.vn_tol,
            "exp_clip": self.exp_clip,
            "experimental_newton_tol": self.experimental_newton_tol,
            "b_clamp": self.b_clamp,
            "single_diode_min_a": self.single_diode_min_a,
            "single_diode_polish_abs_tol": self.single_diode_polish_abs_tol,
            "single_diode_polish_rel_tol": self.single_diode_polish_rel_tol,
            "double_diode_polish_residual_tol": self.double_diode_polish_residual_tol,
        }
        for name, value in positive_fields.items():
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Expected {name} to be finite and positive. "
                    f"Provided value: {value!r}."
                )
        if self.z_thresh is not None and (
            not math.isfinite(self.z_thresh) or self.z_thresh <= 1.0
        ):
            raise ValueError(
                "Expected z_thresh to be finite and greater than 1 for the large-z "
                "Lambert-W expansion. "
                f"Provided value: {self.z_thresh!r}."
            )
        if self.max_newton_iters is not None and self.max_newton_iters < 0:
            raise ValueError(
                "Expected max_newton_iters to be non-negative. "
                f"Provided value: {self.max_newton_iters!r}."
            )
        if self.lambertw_asymptotic_terms is not None and not (
            1 <= self.lambertw_asymptotic_terms <= 4
        ):
            raise ValueError(
                "Expected lambertw_asymptotic_terms to be between 1 and 4. "
                f"Provided value: {self.lambertw_asymptotic_terms!r}."
            )
        if self.pwl_extrapolation is not None:
            _require_choice(
                self.pwl_extrapolation,
                ("clamp", "linear"),
                field="pwl_extrapolation",
            )
        if self.pwl_nonconvergence_policy is not None:
            _require_choice(
                self.pwl_nonconvergence_policy,
                ("accept_last", "error"),
                field="pwl_nonconvergence_policy",
            )
        if self.lambertw_backend is not None:
            _require_choice(
                self.lambertw_backend,
                ("torchlambertw", "torch_special"),
                field="lambertw_backend",
            )


def _require_settings(
    settings: MinimizerSettings,
    fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    missing = [name for name in fields if getattr(settings, name) is None]
    if missing:
        raise ValueError(
            f"Expected explicit {context} minimizer settings {missing}. "
            "Inactive settings must be None; active settings must come from the "
            "resolved configuration."
        )


_CONV_LAYER_TYPES = (ConvLayer,)


class QuadraticUpdater(LayerUpdater):
    """
    Class to update a layer assuming to the function to minimize is quadratic.

    We assume the function E to minimize is a quadractic function of the layer z,
    E(z) = a z^2 + b z + c, for some coefficients a, b and c. Furthermore we assume that the coefficient a is positive.
    A `quadratic update' sets the layer's pre-activation to - b/2a,

    Methods
    -------
    pre_activate():
        Computes the value of the layer that achieves the minimum of the function, given other variables fixed.
    """

    def __init__(self, layer, fn):
        """Creates an instance of LayerUpdater

        Args:
            layer (Layer): the layer to update
            fn (Function): the function to minimize
        """

        super().__init__(layer, fn)

        self._a = fn.a_coef_fn(layer)  # this is a method, not an attribute
        self._b = fn.b_coef_fn(layer)  # this is a method, not an attribute

    def pre_activate(self):
        """Computes the value of the layer that achieves the minimum of the function, given other variables fixed.

        It is assumed that all interactions the layer is involved in are quadratic in the layer's state, i.e. the interaction's energy is of the form
        E_i(z) = a_i z^2 + b_i z + c_i, where z is the layer's state.
        Thus, the global energy of the network is also quadratic in the layer's state, i.e. of the form E(z) = a * z^2 + b * z + c,
        with coefficients a = sum_i a_i and b = sum_i b_i.
        The minimum of this quadractic function in R is obtained at the point z = - b / 2*a (pre-activation)

        Note that the minimum in [min_interval, max_interval] is obtained by clipping - b / 2*a between min_interval and max_interval (activation)

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """

        b = self._b()
        a = self._a()
        return - b / (2. * a)


class AdaptiveQuadraticUpdater(QuadraticUpdater):
    """
    Class to update a layer assuming to the function to minimize is quadratic.

    We assume the function E to minimize is a quadractic function of the layer z,
    E(z) = a z^2 + b z + c, for some coefficients a, b and c. Furthermore we assume that the coefficient a is positive.
    A `quadratic update' sets the layer's pre-activation to - b/2a,
    In this case the diode has a finite conductance when it conducts. An optional
    off-region (deadband) can be provided via diode_params["v_off"] to avoid
    penalties for small voltages around 0.

    Methods
    -------
    pre_activate():
        Computes the value of the layer that achieves the minimum of the function, given other variables fixed.
    """

    def __init__(self, layer, fn, diode_params, *, b_clamp: float):
        super().__init__(layer, fn)
        self._diode_conductance = diode_params["diode_conductance"]
        # Optional off region (deadband) around 0V where no penalty is applied.
        self._v_off = float(diode_params.get("v_off", 0.0))
        self._b_clip = float(b_clamp)

    def pre_activate(self):
        b = self._b()
        b = b.clamp(min=-self._b_clip, max=self._b_clip)
        a = self._a()
        if torch.is_tensor(a):
            a = a.expand_as(b)

        # Check if this is a NonlinearResistiveLayer
        if isinstance(self._layer, NonlinearResistiveLayer):
            # Calculate normal pre-activation
            #normal_pre_activation = -b / (2. * a)
            # the non-linearity is applied before the amplification
            # we need to scale the coefficients a, b to calculate v before the amplification stage
            a_pre = a
            b_pre = b
            preamp_voltage = -b_pre / (2. * a_pre)
            # Apply penalty logic
            dimension = self._layer._shape[0] // 2
            excitatory_pre = preamp_voltage[:, :dimension]
            inhibitory_pre = preamp_voltage[:, dimension:]

            # Check constraint violations (with optional off region)
            v_off = self._v_off
            excitatory_violations = excitatory_pre > v_off   # Excitatory should be ≤ v_off
            inhibitory_violations = inhibitory_pre < -v_off  # Inhibitory should be ≥ -v_off

            # If violations exist, apply penalties and recalculate
            if excitatory_violations.any() or inhibitory_violations.any():
                a_add = torch.zeros_like(a_pre)
                b_sub = torch.zeros_like(b_pre)

                # Apply diode conductance penalty where excitatory > v_off
                if excitatory_violations.any():
                    a_add[:, :dimension] = torch.where(
                        excitatory_violations,
                        a_add[:, :dimension] + 0.5 * self._diode_conductance,
                        a_add[:, :dimension],
                    )
                    b_sub[:, :dimension] = torch.where(
                        excitatory_violations,
                        b_sub[:, :dimension] - self._diode_conductance * v_off,
                        b_sub[:, :dimension],
                    )

                # Apply diode conductance penalty where inhibitory < -v_off
                if inhibitory_violations.any():
                    a_add[:, dimension:] = torch.where(
                        inhibitory_violations,
                        a_add[:, dimension:] + 0.5 * self._diode_conductance,
                        a_add[:, dimension:],
                    )
                    b_sub[:, dimension:] = torch.where(
                        inhibitory_violations,
                        b_sub[:, dimension:] + self._diode_conductance * v_off,
                        b_sub[:, dimension:],
                    )

                a_penalized = a_pre + a_add
                b_penalized = b_pre + b_sub

                # Recalculate pre_activate with penalized a and b
                penalized_preamp = -b_penalized / (2. * a_penalized)
                return penalized_preamp

            return preamp_voltage

        else:
            # For other layer types, just return normal calculation
            return -b / (2. * a)


class QuadraticDoubleDiodeUpdaterOffset(QuadraticUpdater):
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._diode_conductance = diode_params["diode_conductance"]
        self._v_off = diode_params.get("v_off", 0.0)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)

        if isinstance(self._layer, NonlinearResistiveLayer):
            c = torch.as_tensor(self._diode_conductance, dtype=b.dtype, device=b.device)
            v_off = torch.as_tensor(self._v_off, dtype=b.dtype, device=b.device)

            v_lin = -b / (2.0 * a)
            mask_lin = v_lin.abs() <= v_off

            # Positive branch: v >= v_off
            A = c
            Bp = 2.0 * a - 2.0 * c * v_off
            Cp = b + c * v_off**2
            disc_p = (Bp**2 - 4.0 * A * Cp).clamp(min=0)
            v_pos1 = (-Bp + torch.sqrt(disc_p)) / (2.0 * A)
            v_pos2 = (-Bp - torch.sqrt(disc_p)) / (2.0 * A)
            v_pos = torch.where(v_pos1 >= v_off, v_pos1, v_pos2)

            # Negative branch: v <= -v_off
            Bn = 2.0 * c * v_off - 2.0 * a
            Cn = c * v_off**2 - b
            disc_n = (Bn**2 - 4.0 * A * Cn).clamp(min=0)
            v_neg1 = (-Bn + torch.sqrt(disc_n)) / (2.0 * A)
            v_neg2 = (-Bn - torch.sqrt(disc_n)) / (2.0 * A)
            v_neg = torch.where(v_neg1 <= -v_off, v_neg1, v_neg2)

            v = v_lin.clone()
            mask_pos = (b < 0) & ~mask_lin
            mask_neg = (b > 0) & ~mask_lin
            v = torch.where(mask_pos, v_pos, v)
            v = torch.where(mask_neg, v_neg, v)
            return v

        return -b / (2.0 * a)



class QuadraticDoubleDiodeUpdater(QuadraticUpdater):
    """
    Quadratic updater for the double diode non-linearity where the current through tiode is a quadratic function of its voltage
    The energy function is of the form:
    E(z) = a z^2 + b z + c + d abs(z^3), for some coefficients a, b and c. Furthermore we assume that the coefficient a is positive.

    TODO FOR ALL THE CLASSES - SPLIT IN PRE_ACTIVATE/ACTIVATE?
    """

    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._diode_conductance = diode_params["diode_conductance"]

    def pre_activate(self):
        b = self._b()
        a = self._a()
        a = a.expand_as(b)
        # Check if this is a NonlinearResistiveLayer
        if isinstance(self._layer, NonlinearResistiveLayer):
            # the non-linearity is applied before the amplification
            # we need to scale the coefficients a, b to calculate v before the amplification stage
            b_pre = b
            a_pre = a
            mask_pos = b < 0 #this is okay, b is defined as -2*sum over j,k v_j*g_j,k, so we can write i = 2ax+b+i_nonlin
            mask_neg = b > 0
            c = self._diode_conductance

            # Apply penalty logic
            dimension = self._layer._shape[0]
            v = torch.zeros_like(b)


            #We can do this because we know that a is positive and the energy function is convex
            v[mask_pos] = (-a_pre[mask_pos] + torch.sqrt(a_pre[mask_pos]**2 - c*b_pre[mask_pos])) / c
            v[mask_neg] = (a_pre[mask_neg] - torch.sqrt(a_pre[mask_neg]**2 + c*b_pre[mask_neg])) / c

            post_amp_voltage = v

            return post_amp_voltage



        else:
            # For other layer types, just return normal calculation
            return -b / (2. * a)

class ExponentialExactDoubleDiodeUpdater(QuadraticUpdater):
    """
    Exact antiparallel exponential diodes (sinh model) solved by safeguarded Newton.
    """

    def __init__(self, layer, fn, diode_params, *, b_clamp: float):
        super().__init__(layer, fn)
        self._Is = diode_params["I_s"]
        self._Vt = diode_params["V_t"]
        self._v_off = diode_params["V_off"]
        self._b_clip = float(b_clamp)

    def pre_activate(self):
        b = self._b().clamp(min=-self._b_clip, max=self._b_clip)
        a = self._a().expand_as(b)

        if isinstance(self._layer, NonlinearResistiveLayer):
            return self.newton_sinh_hidden(a, b, self._Is, self._v_off, self._Vt)
        else:
            return -b / (2.0 * a)

    @staticmethod
    def _safe_sinh_cosh(u, u_max=80.0):
        """
        Stable sinh/cosh for float64 without overflowing too early.
        Uses exp approximation outside [-u_max, u_max].
        """
        u = u.to(torch.float64)
        ua = torch.abs(u)
        small = ua <= u_max

        sinh = torch.empty_like(u)
        cosh = torch.empty_like(u)

        # small region: use torch.sinh/cosh
        if small.any():
            us = u[small]
            sinh[small] = torch.sinh(us)
            cosh[small] = torch.cosh(us)

        # large region: sinh(u) ~ sign(u)*0.5*exp(|u|), cosh(u) ~ 0.5*exp(|u|)
        if (~small).any():
            ul = u[~small]
            expa = torch.exp(torch.clamp(torch.abs(ul), max=700.0))  # float64 exp limit ~ 709
            cosh[~small] = 0.5 * expa
            sinh[~small] = 0.5 * expa * torch.sign(ul)

        return sinh, cosh

    def newton_sinh_hidden(
        self,
        a, b, I_s, v_off, vt,
        max_newton_iter=20,
        max_bracket_iter=20,
        abs_tol=1e-9,
        rel_tol=1e-9,
    ):
        """
        Solve: 2*a*v + b + K*sinh(v/vt) = 0, with K = 2*Is*exp(-v_off/vt)
        using safeguarded Newton (bracket + Newton, fallback to bisection).
        """

        out_dtype = a.dtype
        device = a.device

        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)

        Is  = torch.as_tensor(I_s,   dtype=torch.float64, device=device)
        vt  = torch.as_tensor(vt,    dtype=torch.float64, device=device)
        vo  = torch.as_tensor(v_off, dtype=torch.float64, device=device)

        # K = 2 Is exp(-v_off/vt)
        K = 2.0 * Is * torch.exp(-vo / vt)

        # helper: g(v), g'(v)
        def g_and_gp(v):
            u = v / vt
            sh, ch = self._safe_sinh_cosh(u)
            g  = 2.0 * a64 * v + b64 + K * sh
            gp = 2.0 * a64 + (K / vt) * ch
            return g, gp

        # initial guess: ignore diode (linear)
        v = -b64 / (2.0 * a64)

        # --- bracketing (monotone => unique root) ---
        # Want lo such that g(lo) <= 0 and hi such that g(hi) >= 0
        # Start with symmetric interval around v
        delta = torch.maximum(torch.abs(v), vt).clamp_min(1e-6)
        lo = v - delta
        hi = v + delta

        glo, _ = g_and_gp(lo)
        ghi, _ = g_and_gp(hi)

        # expand where not bracketed
        need_lo = glo > 0.0   # lo too high
        need_hi = ghi < 0.0   # hi too low

        for _ in range(max_bracket_iter):
            if not (need_lo.any() or need_hi.any()):
                break

            # expand step exponentially
            delta = delta * 2.0

            if need_lo.any():
                lo2 = v - delta
                glo2, _ = g_and_gp(lo2)
                # update only where still need lo
                lo = torch.where(need_lo, lo2, lo)
                glo = torch.where(need_lo, glo2, glo)

            if need_hi.any():
                hi2 = v + delta
                ghi2, _ = g_and_gp(hi2)
                hi = torch.where(need_hi, hi2, hi)
                ghi = torch.where(need_hi, ghi2, ghi)

            need_lo = glo > 0.0
            need_hi = ghi < 0.0

        # --- safeguarded Newton ---
        # Keep lo/hi bracketing and project Newton steps into the bracket.
        for _ in range(max_newton_iter):
            gv, gpv = g_and_gp(v)

            # stopping criterion (abs + rel)
            scale = 1.0 + torch.abs(b64) + torch.abs(2.0 * a64 * v)
            if torch.max(torch.abs(gv) / scale) < (abs_tol + rel_tol):
                break

            step = gv / gpv
            v_new = v - step

            # If Newton goes out of bracket or is invalid, use bisection
            bad = (v_new <= lo) | (v_new >= hi) | torch.isnan(v_new) | torch.isinf(v_new)
            v_bis = 0.5 * (lo + hi)
            v_new = torch.where(bad, v_bis, v_new)

            g_new, _ = g_and_gp(v_new)

            # Update bracket: g increasing => g<0 means root is above, g>0 means root is below
            lo = torch.where(g_new < 0.0, v_new, lo)
            hi = torch.where(g_new > 0.0, v_new, hi)

            v = v_new

            # Also stop if bracket collapses
            if torch.max(torch.abs(hi - lo)) < (abs_tol + rel_tol) * (1.0 + torch.max(torch.abs(v))):
                break

        return v.to(out_dtype)

class ExponentialDoubleDiodeUpdater(QuadraticUpdater):
    """Coordinate minimizer for the manuscript's piecewise exponential law.

    For v != 0, i(v) = sign(v) I_s exp((abs(v) - V_off) / V_t).
    At zero, the primitive has subgradient [-c, c], with
    c = I_s exp(-V_off / V_t); the minimizer is zero when abs(b) <= c.
    """

    _work_dtype = torch.float64

    def __init__(self, layer, fn, diode_params, settings: MinimizerSettings):
        super().__init__(layer, fn)
        self._Is = diode_params["I_s"]
        self._Vt = diode_params["V_t"]
        self._v_off = diode_params["V_off"]
        self._settings = settings
        self._b_clip = settings.b_clamp
        self._lambertw = _load_lambertw_backend(settings.lambertw_backend)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            return self.lambert_hidden(
                a, b, self._Is, self._v_off, self._Vt,
                z_thresh=self._settings.z_thresh,
                polish=self._settings.use_polish,
                exp_clip=self._settings.exp_clip,
                max_inner_iter=self._settings.max_newton_iters,
            )
        return -b / (2.0 * a)

    def lambertw_large(self, z, terms):
        return self.lambertw_large_log(torch.log(z), terms)

    def lambertw_large_log(self, L1, terms):
        """Asymptotic W0(z) from L1 = log(z), so z itself never has to be formed."""
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
        work_dtype = self._work_dtype
        a_work = a.to(work_dtype)
        b_work = b.to(work_dtype)
        if self._b_clip is not None:
            b_work = b_work.clamp(min=-self._b_clip, max=self._b_clip)
        Is = torch.as_tensor(I_s, dtype=work_dtype, device=a.device)
        vt = torch.as_tensor(vt, dtype=work_dtype, device=a.device)
        v_off = torch.as_tensor(v_off, dtype=work_dtype, device=a.device)
        context = type(self).__name__

        # By symmetry, solve for the nonnegative magnitude t = abs(v):
        # 2 a t - abs(b) + I_s exp((t - V_off) / V_t) = 0.
        # There is no branch-dependent +/- I_s constant in this law.
        drive = b_work.abs()
        active = drive > Is * torch.exp(-v_off / vt)
        A = drive / (2.0 * a_work)
        # Lambert-W argument in the log domain. Its exponent (A - V_off) / V_t uses the
        # diode-free voltage A and outgrows exp_clip at high input gain; clipping it there
        # made the clipped law's root a false fixed point. The root's own exponent, used
        # by the polish below, stays below log(b_clamp / I_s), so exp_clip never binds there.
        log_prefactor = torch.log(Is / (2.0 * a_work * vt))
        log_z = log_prefactor + (A - v_off) / vt
        # -inf (I_s = 0) is exact: W0(0) = 0 gives the quadratic root.
        if not torch.isfinite(log_z).all() and (active & (torch.isnan(log_z) | (log_z == math.inf))).any():
            raise NonFiniteDiodeError(f"{context} log Lambert-W argument: produced a non-finite tensor.")

        # The zero branch requires neither Lambert W nor Newton refinement.
        use_asym = active & (log_z > math.log(z_thresh))
        small_mask = active & ~use_asym
        W0 = torch.zeros_like(A)
        if small_mask.any():
            z_small = torch.exp(log_z[small_mask].to(torch.float64))
            W0[small_mask] = self._lambertw(z_small).real.to(work_dtype)
        if use_asym.any():
            W0[use_asym] = self.lambertw_large_log(
                log_z[use_asym].to(torch.float64), terms=self._settings.lambertw_asymptotic_terms,
            ).to(work_dtype)
        _require_finite_tensor(W0, context=f"{context} Lambert-W value")

        magnitude = torch.where(active, (A - vt * W0).clamp(min=0.0), 0.0)
        # For large z, A - V_t W0 subtracts nearly equal voltages; W0 e^W0 = z gives the
        # same root as V_off + V_t (log W0 - log_prefactor) without the cancellation.
        if use_asym.any():
            magnitude[use_asym] = v_off + vt * (torch.log(W0[use_asym]) - log_prefactor[use_asym])
        _require_finite_tensor(magnitude, context=f"{context} initial voltage")
        if polish and active.any():
            current_tol = self._settings.double_diode_polish_residual_tol * b_work.shape[0]
            t = magnitude[active]
            a_active = a_work[active]
            drive_active = drive[active]
            for _ in range(max_inner_iter):
                exp_arg = ((t - v_off) / vt).clamp(min=-exp_clip, max=exp_clip)
                current = Is * torch.exp(exp_arg)
                residual = 2.0 * a_active * t - drive_active + current
                derivative = 2.0 * a_active + current / vt
                _require_finite_tensor(residual, context=f"{context} Newton residual")
                _require_finite_tensor(derivative, context=f"{context} Newton derivative")
                t = (t - residual / derivative).clamp(min=0.0)
                _require_finite_tensor(t, context=f"{context} polished voltage")
                if torch.norm(residual) < current_tol:
                    break
            magnitude[active] = t
        return (-torch.sign(b_work) * magnitude).to(out_dtype)


class ExponentialSingleDiodeUpdater(QuadraticUpdater):
    """
    Supports two orientations of a single Shockley diode.

    Forward orientation (as in your derivation):
        2 a v + b + I_s (exp((v - v_off)/vT) - 1) = 0

    Reversed orientation (your corrected equation):
        2 a v + b - I_s (exp((-(v + v_off))/vT) - 1) = 0
      <=> 2 a v + (b + I_s) - I_s exp(-(v + v_off)/vT) = 0

    You can apply a boolean mask to choose orientation per element.
    """

    def __init__(self, layer, fn, diode_params, settings: MinimizerSettings):
        super().__init__(layer, fn)
        self._Is   = diode_params["I_s"]
        self._Vt   = diode_params["V_t"]
        self._v_off = diode_params.get("V_off")
        self._settings = settings
        self._b_clip = settings.b_clamp
        self._lambertw = _load_lambertw_backend(settings.lambertw_backend)

        # Optional: per-element orientation selection
        # True  -> reversed diode
        # False -> forward diode
        self._reverse_mask = diode_params.get("reverse_mask", None)

    def _debug_tensor(self, name, tensor):
        if not os.environ.get("DRN_DEBUG_DIODE"):
            return
        if not torch.is_tensor(tensor):
            print(f"[ExponentialSingleDiodeUpdater] {name} is not a tensor: {type(tensor)}")
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            finite_min = finite_vals.min().item()
            finite_max = finite_vals.max().item()
            finite_mean = finite_vals.mean().item()
        else:
            finite_min = None
            finite_max = None
            finite_mean = None
        print(
            "[ExponentialSingleDiodeUpdater] non-finite in "
            f"{name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}"
        )
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
            breakpoint()

    def pre_activate(self):
        b = self._b()                     # [N, M]
        a = self._a().expand_as(b)         # [N, M]
        self._debug_tensor("a", a)
        self._debug_tensor("b", b)

        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

        # node-wise mask: [M] -> broadcast to [N, M]
        M = b.shape[1]
        reverse_mask_nodes = torch.arange(M, device=b.device) >= (M // 2)  # [M]
        reverse_mask = reverse_mask_nodes.unsqueeze(0).expand_as(b)         # [N, M]

        solver_settings = {
            "z_thresh": self._settings.z_thresh,
            "polish": self._settings.use_polish,
            "abs_tol": self._settings.single_diode_polish_abs_tol,
            "rel_tol": self._settings.single_diode_polish_rel_tol,
            "exp_clip": self._settings.exp_clip,
            "a_min": self._settings.single_diode_min_a,
            "max_inner_iter": self._settings.max_newton_iters,
        }
        v_fwd = self.lambert_single_forward(
            a, b, self._Is, self._v_off, self._Vt, **solver_settings
        )
        v_rev = self.lambert_single_reverse(
            a, b, self._Is, self._v_off, self._Vt, **solver_settings
        )
        self._debug_tensor("v_fwd", v_fwd)
        self._debug_tensor("v_rev", v_rev)

        return torch.where(reverse_mask, v_rev, v_fwd)


    # ---------- Lambert-W helpers ----------

    def lambertw_large(self, z, terms):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w = w + L2 / L1
        if terms >= 3:
            w = w + (L2 * (-2.0 + L2)) / (2.0 * L1**2)
        if terms >= 4:
            w = w + (L2 * (6.0 - 9.0*L2 + 2.0*L2**2)) / (6.0 * L1**3)
        return w

    def _lambertw0(self, z):
        return self._lambertw(z).real

    def _lambertw_from_log(self, log_z, z_thresh, *, refine):
        """Evaluate W0(exp(log_z)) without forming an overflowing exp(log_z).

        The configured backend is used only within its FP32 exponent range:
        torchlambertw currently casts even FP64 arguments to FP32. Above it, the
        asymptotic series is evaluated from log_z itself. With refine, Newton steps
        on w + log(w) = log_z bring either value to FP64 accuracy; with the voltage
        polish enabled they are redundant and skipped (except for crude seeds).
        """
        log_threshold = min(math.log(z_thresh), math.log(torch.finfo(torch.float32).max) - 2.0)
        large = log_z > log_threshold
        w = torch.empty_like(log_z)
        if not large.all():
            w[~large] = self._lambertw0(torch.exp(log_z[~large])).to(torch.float64)
        if large.any():
            l = log_z[large]
            # Near z=1 the asymptotic series is unsuitable; start from W=1 and refine.
            crude = l < 8.0
            refine = refine or bool(crude.any())
            l = l.clamp_min(8.0)
            l2 = torch.log(l)
            value = l - l2
            terms = self._settings.lambertw_asymptotic_terms
            if terms >= 2:
                value = value + l2 / l
            if terms >= 3:
                value = value + l2 * (l2 - 2.0) / (2.0 * l.square())
            if terms >= 4:
                value = value + l2 * (6.0 - 9.0*l2 + 2.0*l2.square()) / (6.0*l.pow(3))
            w[large] = torch.where(crude, 1.0, value)
        if refine:
            # W(z) = 0 only when exp(log_z) underflows; those entries stay exact.
            positive = w > 0.0
            for _ in range(3):
                w = torch.where(positive, w - (w + torch.log(w) - log_z) * (w / (w + 1.0)), w)
        return w

    def _lambert_single_oriented(
        self, a, b, I_s, v_off, vT, *, sign, z_thresh, polish,
        abs_tol, rel_tol, a_min, max_inner_iter,
    ):
        """Solve the full Shockley coordinate equation in u=sign*v."""
        out_dtype, device = a.dtype, a.device
        diagonal = 2.0 * a.to(torch.float64).clamp_min(a_min)
        b_work = b.to(torch.float64)
        if self._b_clip is not None:
            b_work = b_work.clamp(-self._b_clip, self._b_clip)
        diagonal, b_work = torch.broadcast_tensors(diagonal, b_work)
        _require_finite_tensor(b_work, context="Single-Shockley linear coefficient")
        if I_s == 0:
            result = -b_work / diagonal
            return _require_finite_tensor(result.to(out_dtype), context="Single-Shockley linear voltage")
        Is = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt = torch.as_tensor(vT, dtype=torch.float64, device=device)
        off = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        oriented_b = sign * b_work
        linear = (Is - oriented_b) / diagonal
        log_prefactor = torch.log(Is) - torch.log(diagonal) - torch.log(vt)
        log_z = log_prefactor + (linear - off) / vt
        w = self._lambertw_from_log(log_z, z_thresh, refine=not polish)
        # Equivalent reconstruction prevents subtraction of nearly equal,
        # enormous voltages when the finite diode voltage is close to off.
        u = torch.where(log_z > 0.0, off + vt * (torch.log(w) - log_prefactor), linear - vt * w)

        if polish:
            for _ in range(max_inner_iter):
                # Started at the Lambert-W root, the exponent stays near
                # log(|b| / I_s); 700 only guards FP64 overflow and never binds there.
                e = torch.exp(((u - off) / vt).clamp(max=700.0))
                residual = diagonal*u + oriented_b + Is*(e - 1.0)
                derivative = diagonal + (Is/vt)*e
                u = u - residual / derivative
                max_abs_f = residual.abs().max()
                magnitude = (diagonal*u.abs() + b_work.abs() + Is + 1.0).max()
                if (max_abs_f < abs_tol) and (max_abs_f / (magnitude + a_min) < rel_tol):
                    break
        return _require_finite_tensor((sign*u).to(out_dtype), context="Single-Shockley returned voltage")

    def lambert_single_forward(
        self, a, b, I_s, v_off, vT,
        z_thresh=1e10, polish=True, abs_tol=1e-10, rel_tol=1e-10,
        exp_clip=10000.0, a_min=1e-30, max_inner_iter=32,
    ):
        """Solve 2av+b+Is*expm1((v-v_off)/vT)=0 without exponent clipping.

        exp_clip is retained for configuration/API compatibility only.
        """
        return self._lambert_single_oriented(
            a, b, I_s, v_off, vT, sign=1.0, z_thresh=z_thresh, polish=polish,
            abs_tol=abs_tol, rel_tol=rel_tol, a_min=a_min, max_inner_iter=max_inner_iter,
        )

    def lambert_single_reverse(
        self, a, b, I_s, v_off, vT,
        z_thresh=1e10, polish=True, abs_tol=1e-10, rel_tol=1e-10,
        exp_clip=10000.0, a_min=1e-30, max_inner_iter=32,
    ):
        """Solve 2av+b-Is*expm1((-v-v_off)/vT)=0 with the same stable kernel."""
        return self._lambert_single_oriented(
            a, b, I_s, v_off, vT, sign=-1.0, z_thresh=z_thresh, polish=polish,
            abs_tol=abs_tol, rel_tol=rel_tol, a_min=a_min, max_inner_iter=max_inner_iter,
        )


class HardSigmoidUpdater(QuadraticUpdater):
    """
    Two-diode symmetric clamp:
        v < v_min  -> diode to v_min conducts - diode's conductance is g_on
        v_min<=v<=v_max -> diode's conductance is g_off
        v > v_max  -> diode to v_max conducts- diode's conductance is g_on
    The base quadratic is assumed E(v) = a*v*v + b*v + const, so v* = -b/(2a).
    """
    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self.g_on = diode_params.get("g_on")
        self.g_off = diode_params.get("g_off")

        self._vmin = float(diode_params["v_min"])
        self._vmax = float(diode_params["v_max"])

    def pre_activate(self):
        a = self._a()  # same shape as b
        b = self._b()

        if (not isinstance(self._layer, NonlinearResistiveLayer)) and (not isinstance(self._layer, _CONV_LAYER_TYPES)):
            # plain quadratic update
            return -b / (2.0 * a)

        # Unconstrained minimizer (no diode conduction)
        # The off-region energy is 0.5 * g_off * v^2, so under the
        # E(v) = a v^2 + b v + const convention it contributes 0.5 * g_off to a.
        a_off = a + 0.5 * self.g_off
        v_free = -b / (2.0 * a_off)

        # Masks for violations
        below = v_free < self._vmin
        above = v_free > self._vmax

        if not (below.any() or above.any()):
            # Inside the dead-zone: no diode conduction
            return v_free

        # Allocate per-element penalties/shifts
        # For E = a*v^2 + b*v:
        a_add = torch.zeros_like(a)   # to be added to 'a'
        b_sub = torch.zeros_like(b)   # to be subtracted from 'b'

        # Lower diode ON: add 0.5*gd to 'a' and subtract gd*vmin from 'b'
        if below.any():
            a_add = torch.where(below, a_add + 0.5 * self.g_on, a_add)
            b_sub = torch.where(below, b_sub - self.g_on* self._vmin, b_sub)

        # Upper diode ON: add 0.5*gd to 'a' and subtract gd*vmax from 'b'
        if above.any():
            a_add = torch.where(above, a_add + 0.5 * self.g_on, a_add)
            b_sub = torch.where(above, b_sub - self.g_on * self._vmax, b_sub)

        a_eff = a + a_add
        b_eff = b + b_sub #

        # Exact per-coordinate minimizer in the active regime
        v_star = -b_eff / (2.0 * a_eff)

        return v_star

class _EquilibriumMinimizer(Minimizer):
    """Coordinate descent with optional voltage-tolerance early stopping."""

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
        self._iterations_performed = 0
        self._final_max_voltage_change = math.inf
        self._final_reference_scale = math.inf

    @property
    def iterations_performed(self):
        """Number of complete sweeps performed by the latest equilibrium call."""

        return self._iterations_performed

    @property
    def final_max_voltage_change(self):
        """Largest state change in the final adaptive/fixed sweep."""

        return self._final_max_voltage_change

    @property
    def final_reference_scale(self):
        """Largest pre-sweep voltage magnitude for the latest sweep."""

        return self._final_reference_scale

    @staticmethod
    def _max_abs_delta(new_state, old_state):
        return float((new_state - old_state).abs().max().item())

    def _step_group(self, layer_group):
        if layer_group:
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
        """Compute equilibrium using fixed sweeps or adaptive stopping."""

        if self._mode != "asynchronous":
            previous = [updater._layer.state.detach().clone() for updater in self._updaters]
            result = Minimizer.compute_equilibrium(self)
            self._iterations_performed = self._num_iterations
            self._final_max_voltage_change = max(
                (
                    self._max_abs_delta(updater._layer.state, old_state)
                    for updater, old_state in zip(self._updaters, previous)
                ),
                default=0.0,
            )
            self._final_reference_scale = max(
                (float(state.abs().max().item()) for state in previous),
                default=0.0,
            )
            return result

        max_num_of_iterations = len(self._list_layers) // 2
        use_adaptive = self._adaptive_equilibrium and not self._force_fixed_iterations
        self._iterations_performed = 0
        self._final_max_voltage_change = math.inf
        self._final_reference_scale = math.inf
        if not use_adaptive:
            if len(self._list_layers) >= 2:
                layer_group_odd, layer_group_even = self._list_layers[0], self._list_layers[1]
                for _ in range(max_num_of_iterations):
                    old_states = [
                        updater._layer.state.detach().clone()
                        for updater in (*layer_group_odd, *layer_group_even)
                    ]
                    self._step_odd_even(layer_group_odd, layer_group_even)
                    updaters = (*layer_group_odd, *layer_group_even)
                    self._final_max_voltage_change = max(
                        (
                            self._max_abs_delta(updater._layer.state, old_state)
                            for updater, old_state in zip(updaters, old_states)
                        ),
                        default=0.0,
                    )
                    self._final_reference_scale = max(
                        (float(state.abs().max().item()) for state in old_states),
                        default=0.0,
                    )
                    self._iterations_performed += 1
            else:
                for layer_group in self._list_layers:
                    old_states = [
                        updater._layer.state.detach().clone() for updater in layer_group
                    ]
                    self._step_group(layer_group)
                    self._final_max_voltage_change = max(
                        (
                            self._max_abs_delta(updater._layer.state, old_state)
                            for updater, old_state in zip(layer_group, old_states)
                        ),
                        default=0.0,
                    )
                    self._final_reference_scale = max(
                        (float(state.abs().max().item()) for state in old_states),
                        default=0.0,
                    )
                    self._iterations_performed += 1
        else:
            rtol = self._settings.rel_tol
            vntol = self._settings.vn_tol
            for _ in range(max_num_of_iterations):
                layer_group_odd, layer_group_even = self._list_layers[0], self._list_layers[1]
                odd_old = [u._layer.state.detach().clone() for u in layer_group_odd]
                even_old = [u._layer.state.detach().clone() for u in layer_group_even]
                self._step_odd_even(layer_group_odd, layer_group_even)

                inf_delta = 0.0
                inf_ref = 0.0
                for updater, old_state in zip(layer_group_odd, odd_old):
                    inf_delta = max(inf_delta, self._max_abs_delta(updater._layer.state, old_state))
                    inf_ref = max(inf_ref, float(old_state.abs().max().item()))
                for updater, old_state in zip(layer_group_even, even_old):
                    inf_delta = max(inf_delta, self._max_abs_delta(updater._layer.state, old_state))
                    inf_ref = max(inf_ref, float(old_state.abs().max().item()))

                self._iterations_performed += 1
                self._final_max_voltage_change = inf_delta
                self._final_reference_scale = inf_ref
                if inf_delta <= rtol * inf_ref + vntol:
                    break

        return {layer.name: layer.state for layer in self._layers}


class Float64ExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Float64 exponential double-diode coordinate updater."""


class Float32ExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Float32 double-diode updater with optional SOR-style overrelaxation."""

    _work_dtype = torch.float32

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        settings: MinimizerSettings,
        *,
        overrelaxation_factor: float,
    ):
        super().__init__(layer, fn, diode_params, settings)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def pre_activate(self):
        v = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v
        return self._layer.state + omega * (v - self._layer.state)


class OverRelaxedFloat64DoubleDiodeUpdater(Float64ExponentialDoubleDiodeUpdater):
    """Float64 double-diode updater with SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        settings: MinimizerSettings,
        *,
        overrelaxation_factor: float,
    ):
        super().__init__(layer, fn, diode_params, settings)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd
        v_old = self._layer.state
        return v_old + omega * (v_cd - v_old)


class ConfigurableExponentialSingleDiodeUpdater(ExponentialSingleDiodeUpdater):
    """Single-diode updater with explicit forward/reverse orientation split."""

    def __init__(self, layer, fn, diode_params, settings: MinimizerSettings):
        super().__init__(layer, fn, diode_params, settings)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)

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
            abs_tol=self._settings.single_diode_polish_abs_tol,
            rel_tol=self._settings.single_diode_polish_rel_tol,
            exp_clip=self._settings.exp_clip,
            a_min=self._settings.single_diode_min_a,
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
            abs_tol=self._settings.single_diode_polish_abs_tol,
            rel_tol=self._settings.single_diode_polish_rel_tol,
            exp_clip=self._settings.exp_clip,
            a_min=self._settings.single_diode_min_a,
            max_inner_iter=self._settings.max_newton_iters,
        )
        return torch.where(reverse_mask, v_rev, v_fwd)

    def lambertw_large(self, z, terms):
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
        return self._lambertw(z).real

    def lambert_single_forward(
        self, a, b, I_s, v_on, vT, z_thresh, polish, abs_tol, rel_tol,
        exp_clip, a_min, max_inner_iter,
    ):
        """Preserve the configurable API while sharing the full-law kernel."""
        return super().lambert_single_forward(
            a, b, I_s, v_on, vT, z_thresh, polish, abs_tol, rel_tol,
            exp_clip, a_min, max_inner_iter,
        )

    def lambert_single_reverse(
        self, a, b, I_s, v_on, vT, z_thresh, polish, abs_tol, rel_tol,
        exp_clip, a_min, max_inner_iter,
    ):
        """Reverse orientation of the same unclipped Shockley equation."""
        return super().lambert_single_reverse(
            a, b, I_s, v_on, vT, z_thresh, polish, abs_tol, rel_tol,
            exp_clip, a_min, max_inner_iter,
        )


class OverRelaxedSingleDiodeUpdater(ConfigurableExponentialSingleDiodeUpdater):
    """Single-diode updater with SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        settings: MinimizerSettings,
        *,
        overrelaxation_factor: float,
    ):
        super().__init__(layer, fn, diode_params, settings)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd
        v_old = self._layer.state
        return v_old + omega * (v_cd - v_old)


class ExperimentalIVCurveUpdater(LayerUpdater):
    """Layer updater for a measured piecewise-linear I-V curve tensor."""

    def __init__(
        self,
        layer,
        fn,
        iv_data,
        *,
        damping: float,
        max_newton_steps: int,
        newton_tol: float,
        extrapolation: str,
        nonconvergence_policy: str,
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
        self._extrapolation = _require_choice(
            extrapolation,
            ("clamp", "linear"),
            field="pwl_extrapolation",
        )
        self._clamp = self._extrapolation == "clamp"
        self._nonconvergence_policy = _require_choice(
            nonconvergence_policy,
            ("accept_last", "error"),
            field="pwl_nonconvergence_policy",
        )
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

        v_old = -b / (2 * a)
        _require_finite_tensor(v_old, context="ExperimentalIVCurveUpdater initial voltage")
        active = torch.ones_like(v_old, dtype=torch.bool)
        for _ in range(self._max_newton_steps):
            v = newton_step(v_old)
            _require_finite_tensor(v, context="ExperimentalIVCurveUpdater Newton voltage")
            converged = torch.abs(v - v_old) < self._newton_tol
            active = active & ~converged
            v_old = v
            if not active.any().item():
                break
        if active.any().item() and self._nonconvergence_policy == "error":
            raise RuntimeError(
                "ExperimentalIVCurveUpdater exhausted max_steps before every "
                f"coordinate reached voltage_tolerance; remaining={int(active.sum().item())}."
            )
        return v_old


class OverRelaxedExperimentalIVCurveUpdater(ExperimentalIVCurveUpdater):
    """Measured I-V updater with SOR-style overrelaxation."""

    def __init__(
        self,
        layer,
        fn,
        iv_data,
        *,
        overrelaxation_factor: float,
        damping: float,
        max_newton_steps: int,
        newton_tol: float,
        extrapolation: str,
        nonconvergence_policy: str,
    ):
        super().__init__(
            layer,
            fn,
            iv_data,
            damping=damping,
            max_newton_steps=max_newton_steps,
            newton_tol=newton_tol,
            extrapolation=extrapolation,
            nonconvergence_policy=nonconvergence_policy,
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


class QuadraticMinimizer(_EquilibriumMinimizer):
    """
    Coordinate-descent minimizer that relies on quadratic closed-form updates.

    Args:
        fn (Function): energy to minimise.
        free_layers (Sequence[Layer]): layers updated during minimisation.
        num_iterations (int): number of coordinate-descent sweeps.
        mode (str): update ordering ('forward', 'backward', 'synchronous', 'asynchronous').
        non_linearity (str): non-linearity identifier.
        quadratic_diode_param (Mapping): parameters for quadratic-style non-linearities.
        exponential_diode_param (Mapping): parameters for exponential non-linearities.
        hard_sigmoid_param (Mapping): parameters for hard-sigmoid style non-linearities.
        voltage_amp (float): voltage amplification factor.
        current_amp (float): current amplification factor.
    """

    def __init__(
        self,
        fn,
        free_layers,
        num_iterations,
        mode,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        hard_sigmoid_param,
        voltage_amp,
        current_amp,
        *,
        minimizer_settings: MinimizerSettings,
        iv_data,
        double_diode_updater,
        adaptive_equilibrium,
        overrelaxation_factor,
        single_diode_updater,
        damping,
        experimental_newton_max_steps,
    ):
        quadratic_params = dict(quadratic_diode_param)
        exponential_params = dict(exponential_diode_param)
        hard_sigmoid_params = dict(hard_sigmoid_param)
        if not isinstance(minimizer_settings, MinimizerSettings):
            raise TypeError(
                "Expected an explicit minimizer_settings MinimizerSettings object. "
                f"Provided value: {minimizer_settings!r}."
            )
        settings = minimizer_settings
        if not isinstance(adaptive_equilibrium, bool):
            raise ValueError(
                "Expected adaptive_equilibrium to be a boolean. "
                f"Provided value: {adaptive_equilibrium!r}."
            )
        overrelaxation_factor = float(overrelaxation_factor)
        if not math.isfinite(overrelaxation_factor) or overrelaxation_factor <= 0.0:
            raise ValueError(
                "Expected overrelaxation_factor to be finite and positive. "
                f"Provided value: {overrelaxation_factor!r}."
            )

        non_linearity = _require_choice(
            non_linearity,
            (
                "perfect_diode",
                "lpw_diode",
                "double_diode_quadratic",
                "double_diode_exponential",
                "single_diode_exponential",
                "experimental",
                "hard_sigmoid",
                "linear",
            ),
            field="non_linearity",
        )

        if adaptive_equilibrium:
            _require_settings(
                settings,
                ("rel_tol", "vn_tol"),
                context="adaptive-equilibrium",
            )
        if non_linearity == "lpw_diode":
            _require_settings(settings, ("b_clamp",), context="LPW-diode")
        elif non_linearity in {
            "double_diode_exponential",
            "single_diode_exponential",
        }:
            _require_settings(
                settings,
                (
                    "use_polish",
                    "z_thresh",
                    "exp_clip",
                    "b_clamp",
                    "lambertw_backend",
                    "lambertw_asymptotic_terms",
                ),
                context="Shockley",
            )
            if settings.use_polish:
                _require_settings(
                    settings,
                    ("max_newton_iters",),
                    context="Shockley polish",
                )
            if non_linearity == "single_diode_exponential":
                _require_settings(
                    settings,
                    ("single_diode_min_a",),
                    context="single-Shockley",
                )
                if settings.use_polish:
                    _require_settings(
                        settings,
                        (
                            "single_diode_polish_abs_tol",
                            "single_diode_polish_rel_tol",
                        ),
                        context="single-Shockley polish",
                    )
            elif settings.use_polish:
                _require_settings(
                    settings,
                    ("double_diode_polish_residual_tol",),
                    context="double-Shockley polish",
                )
        elif non_linearity == "experimental":
            _require_settings(
                settings,
                (
                    "experimental_newton_tol",
                    "pwl_extrapolation",
                    "pwl_nonconvergence_policy",
                ),
                context="measured/PWL",
            )

        if non_linearity == 'perfect_diode':
            updaters = [QuadraticUpdater(layer, fn) for layer in free_layers]
        elif non_linearity == 'lpw_diode':
            updaters = [
                AdaptiveQuadraticUpdater(
                    layer,
                    fn,
                    quadratic_params,
                    b_clamp=settings.b_clamp,
                )
                for layer in free_layers
            ]
        elif non_linearity == 'double_diode_quadratic':
            updaters = [QuadraticDoubleDiodeUpdaterOffset(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == 'double_diode_exponential':
            updater_name = _require_choice(
                double_diode_updater,
                (
                    "float64",
                    "float32",
                    "float64_overrelaxed",
                    "float32_overrelaxed",
                ),
                field="double_diode_updater",
            )
            if updater_name == "float64":
                updaters = [
                    Float64ExponentialDoubleDiodeUpdater(
                        layer, fn, exponential_params, settings
                    )
                    for layer in free_layers
                ]
            elif updater_name in {"float32", "float32_overrelaxed"}:
                omega = (
                    overrelaxation_factor
                    if updater_name == "float32_overrelaxed"
                    else 1.0
                )
                updaters = [
                    Float32ExponentialDoubleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        settings,
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
                        settings,
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
            else:
                raise AssertionError(updater_name)
        elif non_linearity == 'single_diode_exponential':
            updater_name = _require_choice(
                single_diode_updater,
                ("custom", "standard", "overrelaxed"),
                field="single_diode_updater",
            )
            if updater_name == "custom":
                updaters = [
                    ConfigurableExponentialSingleDiodeUpdater(
                        layer, fn, exponential_params, settings
                    )
                    for layer in free_layers
                ]
            elif updater_name == "standard":
                updaters = [
                    ExponentialSingleDiodeUpdater(
                        layer, fn, exponential_params, settings
                    )
                    for layer in free_layers
                ]
            elif updater_name == "overrelaxed":
                updaters = [
                    OverRelaxedSingleDiodeUpdater(
                        layer,
                        fn,
                        exponential_params,
                        settings,
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
            else:
                raise AssertionError(updater_name)
        elif non_linearity == 'experimental':
            if iv_data is None:
                raise ValueError(
                    "Expected iv_data to be a 2xN tensor for the measured/PWL "
                    "nonlinearity. Provided value: None."
                )
            # Kept on the legacy argument name only while callers migrate to the
            # typed family-specific updater block.
            updater_name = _require_choice(
                double_diode_updater,
                ("standard", "overrelaxed"),
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
                        newton_tol=settings.experimental_newton_tol,
                        extrapolation=settings.pwl_extrapolation,
                        nonconvergence_policy=settings.pwl_nonconvergence_policy,
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
                        newton_tol=settings.experimental_newton_tol,
                        extrapolation=settings.pwl_extrapolation,
                        nonconvergence_policy=settings.pwl_nonconvergence_policy,
                    )
                    for layer in free_layers
                ]
            else:
                raise AssertionError(updater_name)
        elif non_linearity == 'hard_sigmoid':
            required = {"g_on", "g_off", "v_min", "v_max"}
            missing = sorted(required.difference(hard_sigmoid_params))
            if missing:
                raise ValueError(
                    "Expected hard_sigmoid_param to contain g_on, g_off, v_min, "
                    f"and v_max. Provided value: {hard_sigmoid_params!r}."
                )
            updaters = [HardSigmoidUpdater(layer, fn, hard_sigmoid_params) for layer in free_layers]
        elif non_linearity == 'linear':
            updaters = [QuadraticUpdater(layer, fn) for layer in free_layers]
        else:
            raise ValueError(
                "non_linearity must be 'perfect_diode', 'double_diode_quadratic', "
                "'lpw_diode', 'double_diode_exponential', 'single_diode_exponential', "
                "'experimental', 'hard_sigmoid', or 'linear'; got {}".format(non_linearity)
            )

        _EquilibriumMinimizer.__init__(
            self,
            fn,
            updaters,
            num_iterations,
            mode,
            voltage_amp,
            current_amp,
            adaptive_equilibrium=adaptive_equilibrium,
            settings=settings,
        )

        for updater in self._updaters:
            updater.voltage_amp = self.voltage_amp
            updater.current_amp = self.current_amp

        self._non_linearity = non_linearity
        self._quadratic_params = quadratic_params
        self._exponential_params = exponential_params
        self._hard_sigmoid_params = hard_sigmoid_params
        self._settings = settings

    def __str__(self):
        if self._non_linearity == 'perfect_diode':
            return 'Quadratic minimizer (perfect diode) -- mode={}, num_iterations={}'.format(self._mode, self._num_iterations)
        if self._non_linearity == 'lpw_diode':
            return 'Quadratic minimizer (LPW diode, conductance={}, v_off={}) -- mode={}, num_iterations={}'.format(
                self._quadratic_params["diode_conductance"],
                self._quadratic_params.get("v_off", 0.0),
                self._mode,
                self._num_iterations)
        if self._non_linearity == 'double_diode_quadratic':
            return 'Quadratic minimizer (double diode quadratic, conductance={}) -- mode={}, num_iterations={}'.format(
                self._quadratic_params["diode_conductance"], self._mode, self._num_iterations)
        if self._non_linearity == 'double_diode_exponential':
            return 'Quadratic minimizer (double diode exponential, I_s={}, V_t={}, V_off={}) -- mode={}, num_iterations={}'.format(
                self._exponential_params["I_s"],
                self._exponential_params["V_t"],
                self._exponential_params["V_off"],
                self._mode,
                self._num_iterations)
        if self._non_linearity == 'single_diode_exponential':
            return 'Quadratic minimizer (single diode exponential, I_s={}, V_t={}, V_off={}) -- mode={}, num_iterations={}'.format(
                self._exponential_params["I_s"],
                self._exponential_params["V_t"],
                self._exponential_params["V_off"],
                self._mode,
                self._num_iterations)
        if self._non_linearity == 'hard_sigmoid':
            return 'Quadratic minimizer (hard sigmoid, g_on={}, g_off={}, v_min={}, v_max={}) -- mode={}, num_iterations={}'.format(
                self._hard_sigmoid_params.get("g_on"),
                self._hard_sigmoid_params.get("g_off"),
                self._hard_sigmoid_params["v_min"],
                self._hard_sigmoid_params["v_max"],
                self._mode,
                self._num_iterations)
        return 'Quadratic minimizer (linear) -- mode={}, num_iterations={}'.format(self._mode, self._num_iterations)
