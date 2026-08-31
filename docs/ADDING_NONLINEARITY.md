# Adding a nonlinearity

There are two extension routes. If a device can be represented by sampled
current-voltage data, use the measured/PWL route; it requires no simulator code
changes. Add an analytic implementation only when the current law needs an
exact formula, a specialized coordinate solve, or parameters that cannot be
represented by one fixed sampled curve.

## Use a sampled I-V curve

Store the curve in an uncompressed or compressed NumPy `.npz` file using one
of these layouts:

- one-dimensional `i` and `v` arrays of equal length; or
- one `iv` array with shape `(2, N)`, with **current in row 0** and voltage in
  row 1.

Both layouts must contain at least two real numeric, finite samples. Voltages
must be strictly increasing and currents must be nondecreasing. These ordering
requirements keep every interpolated segment single-valued and passive; the
loader rejects duplicate voltages and negative-slope segments rather than
sorting or modifying the measurement silently.

For example:

```python
import numpy as np


voltage = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
current = np.array([-2.0e-3, -2.0e-4, 0.0, 2.0e-4, 2.0e-3])
np.savez("data/assets/my_curve.npz", i=current, v=voltage)
```

Pass the file to the compact runner while selecting the measured/PWL
nonlinearity and an explicit PWL parameter source:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity pwl \
  --parameter-set paper-digits \
  --iv-data-path data/assets/my_curve.npz \
  --dry-run
```

`--iv-data-path` replaces only the parameter source's curve. Damping, Newton
limits, amplification, learning rates, and all other settings still come from
the selected source unless their own flags override them. The flag is valid
only with `pwl`, `measured`, or canonical `experimental`; using it with a
Shockley model is an error. Relative paths are resolved from the repository
root, and `--dry-run` validates the file before any output directory or dataset
is created. The expanded configuration records whether the curve came from the
parameter source or the user as `runner.iv_data_source`. Paths inside the
repository are serialized relative to its root; external paths remain
absolute.

The equivalent Python field is `DRNRunSpec.iv_data_path`. The lower-level
`python scripts/reproduce.py train` interface does not have a separate curve
flag: set `iv_data_path` in its JSON configuration instead. A JSON-relative
path is also interpreted relative to the repository root.

The measured updater linearly interpolates `I(v)` and solves the coordinate
current balance

```text
2 a v + b + I(v) = 0,
```

where `a v^2 + b v` is the contribution from the rest of the network. The
measured/PWL Newton settings and range behavior are described in
[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md#measuredpwl-newton-settings).

## Implement an analytic current law

An analytic grounded element needs a consistent current, energy, and coordinate
update. Define the current law `I(v; theta)` and an energy primitive

```text
U(v; theta) = integral from 0 to v of I(u; theta) du,
```

so that `dU/dv = I(v)`. Adding `U` to the network energy is what keeps the
free and nudged phases, equilibrium-propagation gradients, and reported energy
consistent with the coordinate solver. For the usual coordinate-descent
guarantees, use a nondecreasing current law: then `U` is convex and, because
`a > 0`, `2 a v + b + I(v)` is strictly increasing and has at most one root.
A nonmonotone or active device needs an explicit root-selection/stability rule
and is outside the assumptions of the bundled updater.

### 1. Add the energy interaction

In `repro/vendor/model/function/interaction.py`, add a `Function` subclass for
the grounded element. Its `eval()` method must sum `U(v)` over all nodes and
return one value per batch item; `grad_layer_fn()` must return the matching
`I(v)`. Follow the existing Shockley interactions for tensor dtype/device
handling and amplifier scaling. Because the specialized updater supplies the
nonlinear part of the coordinate equation, its interaction should expose zero
`a_coef_fn()` and `b_coef_fn()` contributions, as the existing Shockley
interactions do; otherwise `I(v)` would be counted twice. In particular,
physical current parameters are scaled for the layer depth with the same
`voltage_amp`/`current_amp` convention; do not bake one layer's scale into the
base parameter.

Register that interaction in `DeepResistiveEnergy` in
`repro/vendor/model/resistive/network.py`, and add the canonical nonlinearity
name to the no-clipping branch of `NonlinearResistiveLayer.activate()` in
`repro/vendor/model/resistive/layer.py`. The analytic primitive and its
gradient should be tested against finite differences before adding a solver.

### 2. Add the coordinate updater

In `repro/vendor/model/resistive/minimizer.py`, add a `LayerUpdater` that obtains
`a` and `b` from `fn.a_coef_fn(layer)` and `fn.b_coef_fn(layer)`. For nonlinear
hidden layers, solve

```text
F(v) = 2 a v + b + I(v) = 0.
```

Use an exact inverse when one is stable; otherwise use a bracketed method or a
safeguarded Newton iteration with derivative `2 a + dI/dv`. Keep operations
vectorized, preserve the input tensor's dtype and device, stop on an explicit
residual or voltage-change tolerance, and reject non-finite updates. Linear
output or pooling layers must continue to use the quadratic result
`v = -b / (2a)`. If overrelaxation is exposed, apply it only after obtaining a
valid coordinate solution and document any voltage bounds.

Add the canonical name and any human-friendly aliases to the minimizer's
selector. Validate every physical and numerical parameter at construction;
unknown updater names or incomplete parameter dictionaries must fail with an
error that names the expected values.

### 3. Wire configuration, training, and replay

Add one explicit parameter object to the training schema and dataclasses, then
thread it through `repro.train._build_minimizer` and the corresponding builder
in `repro.digits_validate`. Register the canonical name in the training loader,
compact runner aliases and parameter-source map, output encoding choice, and
the schema's `non_linearity` enumeration. A compact runner source must still
provide a complete configuration; do not infer physical parameters from the
alias alone. Add an entry to a bundled parameter-set map only if the repository
also ships and audits a preset for the new law; otherwise use
`--parameter-config`.

Training and checkpoint replay must construct the same energy interaction,
parameter scaling, updater, and numerical settings. Store every new setting in
`config.generated.json`/`config.resolved.json`; archived replay configurations
must remain readable, with a migration or explicit default if the new field is
not conditionally optional. If the new law changes output orientation or
pairing, update the cost selection and document checkpoint incompatibility.

### 4. Verify the extension

At minimum, add tests for:

- `dU/dv = I(v)` by autograd or finite differences over the intended voltage
  range, including branch boundaries;
- the analytic derivative `dI/dv`, if Newton uses one;
- current-balance residuals after a coordinate update on CPU and, when
  supported, CUDA;
- finite outputs at extreme valid parameters and clear rejection of invalid
  parameters;
- alias, schema, config-loading, dry-run, and provenance behavior;
- a one-batch training smoke test with finite equilibrium-propagation
  gradients; and
- deterministic checkpoint replay using the resolved configuration.

Changing a current law or its root solver defines a new experiment. Keep the
new configuration and numerical results separate from the checked paper
presets unless they reproduce the same physical and solver contract exactly.
