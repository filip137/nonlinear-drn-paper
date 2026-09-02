# Simulator profiles

Simulator profiles are the only source of the nonlinear device law,
inter-layer amplification, and coordinate-updater numerical policy. All four
bundled profiles satisfy
[`../schema/simulator-v2.schema.json`](../schema/simulator-v2.schema.json).

| File | Nonlinearity | Active updater |
|---|---|---|
| `default_single_shockley.json` | single Shockley diode | float64 Lambert-W, relaxation 1.2 |
| `default_double_shockley.json` | antiparallel double Shockley diode | float64 Lambert-W, relaxation 1.2 |
| `default_mnist_double_shockley.json` | paper MNIST double Shockley diode | float64 Lambert-W, no relaxation |
| `default_pwl.json` | measured piecewise-linear I-V curve | damped Newton, clamped extrapolation |

The profile shape is a typed union. A Shockley profile contains only Shockley
physical and Lambert-W fields. A measured/PWL profile contains only its curve
reference, extrapolation, nonconvergence policy, damping, iteration limit,
voltage tolerance, and relaxation. `nonconvergence_policy` is exactly
`accept_last` or `error`; the bundled migrated profile explicitly uses
`accept_last`. Inactive settings from other device families are invalid rather
than silently ignored.

The numerical choices that were previously hidden in code or process
environment are now explicit: Lambert-W backend and work dtype, coefficient
clamps, exponent clip, asymptotic threshold and term count, optional Newton
polish, and PWL extrapolation/nonconvergence behavior. The measured curve
itself is a repository-relative NPZ path. Its contents are validated before
the model is built and fingerprinted automatically in each run receipt; bundled
curve bytes are also covered by the release checksum index.

The deterministic migration records its tool/schema transition under
`provenance.migration` and lists only the active formerly implicit fields under
`provenance.materialized_defaults`. The list is family-specific: Shockley
profiles name their hidden Lambert-W safeguards, single Shockley additionally
names `quadratic_coefficient_min`, and PWL names extrapolation and
nonconvergence policy. Provenance is descriptive and is not runtime input.

To use another measured curve, pass its repository-local path with
`scripts/train_drn.py --iv-curve`. For other simulator changes, copy the
relevant profile into `configs/local/`, edit only fields active for its family,
and validate it. Then update a copied training source's `simulation_ref` with
both the new repository-relative path and the SHA-256 of the exact profile
bytes. Runtime aliases such as `custom`,
`overrelated`, and `overrelaxed` are not part of version 2; behavior is stated
directly through `method`, `dtype`, and `relaxation`.
