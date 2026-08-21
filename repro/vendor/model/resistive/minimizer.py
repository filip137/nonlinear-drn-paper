import os
import socket
import sys
from pathlib import Path
from model.minimizer.minimizer import LayerUpdater, Minimizer
from model.resistive.layer import NonlinearResistiveLayer, ConvLayer
import torch

_HOSTNAME = socket.gethostname()


def _load_lambertw():
    if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
        return torch.special.lambertw
    try:
        import torchlambertw.special as tw_special
        return tw_special.lambertw
    except Exception as exc:
        def _missing_lambertw(*args, _import_error=exc, **kwargs):
            raise ImportError(
                "Lambertw backend unavailable; install torchlambertw or use a PyTorch build "
                "with torch.special.lambertw."
            ) from _import_error

        return _missing_lambertw


lambertw = _load_lambertw()


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

    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._diode_conductance = diode_params["diode_conductance"]
        # Optional off region (deadband) around 0V where no penalty is applied.
        self._v_off = float(diode_params.get("v_off", 0.0))
        b_clip_env = os.environ.get("DRN_B_CLAMP")
        self._b_clip = float(b_clip_env) if b_clip_env is not None else None

    def pre_activate(self):
        b = self._b()
        if self._b_clip is not None:
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

    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._Is = diode_params["I_s"]
        self._Vt = diode_params["V_t"]
        self._v_off = diode_params["V_off"]
        b_clip_env = os.environ.get("DRN_B_CLAMP")
        self._b_clip = float(b_clip_env) if b_clip_env is not None else None

    def pre_activate(self):
        b = self._b()
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
    """
    Quadratic updater for the double diode non-linearity where the current through diode is
    an exponential function of its voltage.
    """

    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._Is = diode_params["I_s"]
        self._Vt = diode_params["V_t"]
        self._v_off = diode_params["V_off"]
        b_clip_env = os.environ.get("DRN_B_CLAMP")
        self._b_clip = float(b_clip_env) if b_clip_env is not None else None

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            v = self.lambert_hidden(a, b, self._Is, self._v_off, self._Vt)
            return v
        else:
            return -b / (2.0 * a)



        # asymptotic expansion for large z
    def lambertw_large(self, z, terms=4):
            L1 = torch.log(z)
            L2 = torch.log(L1)
            w = L1 - L2
            if terms >= 2:
                w += L2 / L1
            if terms >= 3:
                w += (L2 * (-2 + L2)) / (2 * L1**2)
            if terms >= 4:
                w += (L2 * (6 - 9*L2 + 2*L2**2)) / (6 * L1**3)
            return w

    def lambert_hidden(self, a, b, I_s, v_off, vt,
                       z_thresh=1e10, polish=True, exp_clip=10000.0):
        """
        Robust Lambert solver for the piecewise exponential diode equation.
        All computations are done in float64, result cast back to input dtype.
        """

        # remember original dtype (likely float32)
        out_dtype = a.dtype
        device = a.device

        # promote to float64 for stable computation
        a_pre =  a
        b_pre = b
        a64     = a_pre.to(torch.float64)
        b64     = b_pre.to(torch.float64)
        if getattr(self, "_b_clip", None) is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)
        I_s64   = torch.as_tensor(I_s,   dtype=torch.float64, device=device)
        vt64    = torch.as_tensor(vt,    dtype=torch.float64, device=device)
        v_off64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)

        # shift A
        A = torch.where(b64 <= 0, (b64 - I_s64) / (2.0 * a64),
                                   (b64 + I_s64) / (2.0 * a64))

        # Lambert arguments (clamp exponent to avoid overflow)
        exp_arg_add = (-(A + v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        z_add = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_add)  # b <= 0
        z_rev = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_rev)  # b > 0
        z = torch.where(b64 > 0, z_rev, z_add)

        # decide branch on raw z
        use_asym   = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=torch.float64, device=z.device)

        # --- direct path (small/medium z) ---
        if small_mask.any():
            z_small = torch.clamp(z[small_mask].to(torch.float64), max=float(z_thresh))

            W0_small = lambertw(z_small).real.to(W0.dtype)                # float64

            W0[small_mask] = W0_small                                       # dtype match

        # --- asymptotic path (large z) ---
        if use_asym.any():
            z_large  = torch.clamp(z[use_asym].to(torch.float64), min=1.0)  # avoid log(log(z<1))
            W0_large = self.lambertw_large(z_large).to(torch.float64)
            W0[use_asym] = W0_large

    # back-substitute (vt64, A are already float64)
        x = torch.where(b64 > 0, vt64 * W0 - A, -vt64 * W0 - A)
        if torch.isnan(x).any():
            print("NaNs detected; continuing")



        # optional Newton polish
        if polish:
            max_inner_iter = 64
            current_tol = 1e-7 * x.shape[0]
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for k in range(1, max_inner_iter):
                mask_pos = b64 <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]; ap64 = a64[mask_pos]; bp64 = b64[mask_pos]
                    ep_arg = ((xp - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos]  = 2*ap64*xp + (bp64 - I_s) + I_s*ep
                    df[mask_pos] = 2*ap64 + (I_s/vt)*ep

                if mask_neg.any():
                    xn = x[mask_neg]; an64 = a64[mask_neg]; bn64 = b64[mask_neg]
                    en_arg = ((-xn - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg]  = 2*an64*xn + (bn64 + I_s) - I_s*en
                    df[mask_neg] = 2*an64 + (I_s/vt)*en


                x = x - f/df
                if torch.norm(f) < current_tol:
                    break

        # cast back to original dtype (float32)
        x_post = x
        return x_post.to(out_dtype)



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

    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn)
        self._Is   = diode_params["I_s"]
        self._Vt   = diode_params["V_t"]
        self._v_off = diode_params.get("V_off")

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

        v_fwd = self.lambert_single_forward(a, b, self._Is, self._v_off, self._Vt)  # [N, M]
        v_rev = self.lambert_single_reverse(a, b, self._Is, self._v_off, self._Vt)  # [N, M]
        self._debug_tensor("v_fwd", v_fwd)
        self._debug_tensor("v_rev", v_rev)

        return torch.where(reverse_mask, v_rev, v_fwd)


    # ---------- Lambert-W helpers ----------

    def lambertw_large(self, z, terms=4):
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
        if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
            return torch.special.lambertw(z).real
        return lambertw(z).real  # assumes you have lambertw imported elsewhere

    # ---------- Forward diode ----------
    # 2 a v + b + I_s (exp((v - v_off)/vT) - 1) = 0
    def lambert_single_forward(
        self, a, b, I_s, v_off, vT,
        z_thresh=1e10, polish=True,
        abs_tol=1e-10, rel_tol=1e-10,
        exp_clip=10000.0, a_min=1e-30, max_inner_iter=32
    ):
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)

        Is64  = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64  = torch.as_tensor(vT,  dtype=torch.float64, device=device)
        voff64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("Is64", Is64)
        self._debug_tensor("vT64", vT64)
        self._debug_tensor("voff64", voff64)

        # Ashift = (b - I_s)/(2a)
        Ashift = (b64 - Is64) / (2.0 * a64)
        self._debug_tensor("Ashift_fwd", Ashift)

        # z = (I_s/(2 a vT)) * exp(-(v_off + Ashift)/vT)
        z = (Is64 / (2.0 * a64 * vT64)) * torch.exp(-(Ashift + voff64) / vT64)
        self._debug_tensor("z_fwd", z)

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)
        self._debug_tensor("W0_fwd", W0)

        # v* = -Ashift - vT * W0
        x = -Ashift - vT64 * W0
        self._debug_tensor("x_fwd", x)

        if polish:
            for _ in range(max_inner_iter):
                arg = (x - voff64) / vT64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f  = 2.0*a64*x + b64 + Is64*(e - 1.0)
                df = 2.0*a64   + (Is64/vT64)*e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_fwd", f)
                    self._debug_tensor("df_fwd", df)
                    break

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0*torch.abs(a64)*torch.abs(x) + torch.abs(b64) + torch.abs(Is64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f/(scale + 1e-30) < rel_tol):
                    break

        return x.to(out_dtype)

    # ---------- Reversed diode (your corrected equation) ----------
    # 2 a v + b - I_s (exp((-(v + v_off))/vT) - 1) = 0
    # <=> 2 a v + (b + I_s) - I_s exp(-(v + v_off)/vT) = 0
    def lambert_single_reverse(
        self, a, b, I_s, v_off, vT,
        z_thresh=1e10, polish=True,
        abs_tol=1e-10, rel_tol=1e-10,
        exp_clip=10000.0, a_min=1e-30, max_inner_iter=32
    ):
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)

        Is64  = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64  = torch.as_tensor(vT,  dtype=torch.float64, device=device)
        voff64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("Is64", Is64)
        self._debug_tensor("vT64", vT64)
        self._debug_tensor("voff64", voff64)

        # Ashift = (b + I_s)/(2a)
        Ashift = (b64 + Is64) / (2.0 * a64)
        self._debug_tensor("Ashift_rev", Ashift)

        # z = (I_s/(2 a vT)) * exp( -v_off/vT + Ashift/vT )
        z = (Is64 / (2.0 * a64 * vT64)) * torch.exp((-voff64 + Ashift) / vT64)
        self._debug_tensor("z_rev", z)

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)
        self._debug_tensor("W0_rev", W0)

        # v* = -Ashift + vT * W0
        x = -Ashift + vT64 * W0
        self._debug_tensor("x_rev", x)

        if polish:
            for _ in range(max_inner_iter):
                arg = -(x + voff64) / vT64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                # f = 2 a v + b + I_s - I_s e^{-(v+v_off)/vT}
                f  = 2.0*a64*x + b64 + Is64 - Is64*e
                # df = 2 a + (I_s/vT) e^{-(v+v_off)/vT}
                df = 2.0*a64   + (Is64/vT64)*e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_rev", f)
                    self._debug_tensor("df_rev", df)
                    break

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0*torch.abs(a64)*torch.abs(x) + torch.abs(b64) + torch.abs(Is64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f/(scale + 1e-30) < rel_tol):
                    break

        return x.to(out_dtype)


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

class QuadraticMinimizer(Minimizer):
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
        voltage_amp,
        current_amp,
        hard_sigmoid_param=None,
    ):
        quadratic_params = dict(quadratic_diode_param)
        exponential_params = dict(exponential_diode_param)
        hard_sigmoid_params = dict(hard_sigmoid_param or {})
        if not hard_sigmoid_params:
            # Derive sensible defaults from quadratic params when not provided explicitly.
            if "diode_conductance" in quadratic_params:
                hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
                hard_sigmoid_params["g_off"] = quadratic_params["diode_conductance"]
        # Ensure conductance keys exist even if caller omitted them (fallback to symmetric clamp)
        if "g_on" not in hard_sigmoid_params and "diode_conductance" in quadratic_params:
            hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
        if "g_off" not in hard_sigmoid_params and "g_on" in hard_sigmoid_params:
            hard_sigmoid_params["g_off"] = hard_sigmoid_params["g_on"]
        if "v_min" not in hard_sigmoid_params and "v_min" in quadratic_params:
            hard_sigmoid_params["v_min"] = quadratic_params["v_min"]
        if "v_max" not in hard_sigmoid_params and "v_max" in quadratic_params:
            hard_sigmoid_params["v_max"] = quadratic_params["v_max"]

        if non_linearity == 'perfect_diode':
            updaters = [QuadraticUpdater(layer, fn) for layer in free_layers]
        elif non_linearity == 'lpw_diode':
            updaters = [AdaptiveQuadraticUpdater(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == 'double_diode_quadratic':
            updaters = [QuadraticDoubleDiodeUpdaterOffset(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == 'double_diode_exponential':
            updaters = [ExponentialDoubleDiodeUpdater(layer, fn, exponential_params) for layer in free_layers]
        elif non_linearity == 'single_diode_exponential':
            updaters = [ExponentialSingleDiodeUpdater(layer, fn, exponential_params) for layer in free_layers]
        elif non_linearity == 'hard_sigmoid':
            updaters = [HardSigmoidUpdater(layer, fn, hard_sigmoid_params) for layer in free_layers]
        elif non_linearity == 'linear':
            updaters = [QuadraticUpdater(layer, fn) for layer in free_layers]
        else:
            raise ValueError(
                "non_linearity must be 'perfect_diode', 'double_diode_quadratic', "
                "'lpw_diode', 'double_diode_exponential', 'hard_sigmoid', or 'linear'; got {}".format(non_linearity)
            )

        Minimizer.__init__(self, fn, updaters, num_iterations, mode, voltage_amp, current_amp)

        for updater in self._updaters:
            updater.voltage_amp = self.voltage_amp
            updater.current_amp = self.current_amp

        self._non_linearity = non_linearity
        self._quadratic_params = quadratic_params
        self._exponential_params = exponential_params
        self._hard_sigmoid_params = hard_sigmoid_params

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
