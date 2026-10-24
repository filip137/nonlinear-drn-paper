"""custom_lambert_halley: the principal branch W0(z), z >= 0, in PyTorch by Halley's method.

W0 solves w e^w = z. Starting from Winitzki's approximation, Halley steps converge cubically;
every z from 0 through the float64 maximum converges to within 4 ulp in at most 4 steps.

Unlike torchlambertw, which casts to float32, stops at an absolute tolerance of 1e-6 and
loops in Python until every element has converged, this module:
- works in the input's floating dtype (float64 results are accurate to a few ulp);
- has no data-dependent Python control flow and never reads tensor values on the host, so
  it runs batched on CPU or GPU without synchronization and compiles with
  torch.compile(fullgraph=True);
- keeps a per-element tolerance: an element stops changing once its step is within `ulps`
  units in the last place; `max_steps` only caps the work.

Only the principal branch for z >= 0 is provided (the Shockley coordinate solves never need
more). z < 0 returns NaN, +inf returns +inf.
"""

import torch

__all__ = ["DEFAULT_MAX_STEPS", "halley_step", "lambertw", "winitzki"]

DEFAULT_MAX_STEPS = 5


def winitzki(z):
    """Winitzki's approximation to W0(z), the starting point of the iteration.

    Vectorized torch.log1p returns inf near the top of the float range (from about 1e307 in
    float64), so log(1 + z) is taken as log(z) above 1e6, where the two agree to far better
    than a starting point needs.
    """
    lp = torch.where(z > 1e6, torch.log(z), torch.log1p(z.clamp(max=1e6)))
    return lp * (1.0 - torch.log1p(lp) / (2.0 + lp))


def halley_step(w, z):
    """One Halley step for w e^w = z, divided through by e^w so it cannot overflow.

    With f = w - z e^-w (equal to (w e^w - z) e^-w), the step is
    f / (w + 1 - (w + 2) f / (2 (w + 1))), the same as Halley on w e^w - z.
    """
    f = w - z * torch.exp(-w)
    wp1 = w + 1.0
    return f / (wp1 - (w + 2.0) * f / (2.0 * wp1))


def lambertw(z, *, max_steps=DEFAULT_MAX_STEPS, ulps=4.0):
    """W0(z) for a floating tensor z >= 0, elementwise, in z's dtype.

    Each element stops changing once its Halley step is within `ulps` units in the last
    place of w; `max_steps` caps the number of steps (4 suffice everywhere).
    """
    if not torch.is_floating_point(z):
        raise TypeError(f"Expected a floating-point tensor. Provided dtype: {z.dtype}.")
    eps = torch.finfo(z.dtype).eps
    finite = torch.isfinite(z) & (z >= 0.0)
    # The seed of +inf (and NaN) lanes would poison the arithmetic; iterate on 0 instead.
    x = torch.where(finite, z, 0.0)
    w = winitzki(x)
    done = x == 0.0
    for _ in range(max_steps):
        step = halley_step(w, x)
        w = torch.where(done, w, w - step)
        done = done | ~(step.abs() > ulps * eps * w.abs())
    w = torch.where(torch.isposinf(z), torch.inf, w)
    return torch.where(finite | torch.isposinf(z), w, torch.nan)
