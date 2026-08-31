# Training configuration reference

This document defines the complete JSON contract used by `configs/train/`,
`repro.train`, and the compact runner in `repro.runner`. The machine-readable
counterpart is [`configs/train/schema.json`](../configs/train/schema.json).

The checked paper configurations are scientific provenance records as well as
executable inputs. Preserve their values when reproducing the paper. For a new
experiment, use the matching default parameter set or copy a default template
to `configs/local/`, then inspect the expanded JSON with `--dry-run` before
training.

## Default templates and local changes

The compact runner resolves `--parameter-set default` according to the
requested nonlinearity:

| Nonlinearity | Default template |
|---|---|
| `single_diode_exponential` | `configs/train/default_single_shockley.json` |
| `double_diode_exponential` | `configs/train/default_double_shockley.json` |
| `experimental` (measured/PWL) | `configs/train/default_custom_iv.json` |

These sources are valid starting points for either Digits or MNIST, but their
values were validated on Digits. Using them for MNIST creates a new experiment;
it does not establish another paper-MNIST configuration or result. The
`paper-mnist-xs` set remains specific to the reported double-Shockley DRN-XS
run.

Keep versioned templates unchanged. Put local variants under the git-ignored
`configs/local/` directory and select the edited copy with
`--parameter-config`:

```bash
mkdir -p configs/local
cp configs/train/default_single_shockley.json \
  configs/local/my_single_shockley.json
# Edit configs/local/my_single_shockley.json before running:
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-config configs/local/my_single_shockley.json \
  --dry-run
```

The JSON `non_linearity` must match the runner selection. For measured/PWL
sources, `iv_data_path` is the curve path stored in JSON; `--iv-data-path` is
the compact runner override for one invocation. Relative curve paths resolve
from the repository root.

## Numerical settings at a glance

The common schema deliberately records every supported solver family. A field
can therefore be present but inactive for a particular nonlinearity.

| Setting | Single Shockley | Double Shockley | Measured/PWL | Fixed-sweep training |
|---|---:|---:|---:|---:|
| `exponential_diode_param` | active | active | inactive | — |
| `iv_data_path` | inactive | inactive | active | — |
| `z_thresh` | active for `custom`/`overrelaxed` | active for float32/float64 variants | inactive | — |
| `exp_clip` | active for `custom`/`overrelaxed` | active for float32/float64 variants | inactive | — |
| `use_polish` | active for `custom`/`overrelaxed` | active for float32/float64 variants | inactive | — |
| `max_newton_iters` | active only if `use_polish=true` | active only if `use_polish=true` | inactive | — |
| `damping` | inactive | inactive | active | — |
| `experimental_newton_max_steps` | inactive | inactive | active | — |
| `experimental_newton_tol` | inactive | inactive | active | — |
| `overrelaxation_factor` | active for `overrelaxed` | active for `*_overrelaxed` | active for `overrelaxed` | — |
| `rel_tol`, `vn_tol` | — | — | — | inactive because `adaptive_equilibrium=false` |

The inactive fields remain explicit so one configuration can be expanded by
the high-level runner without silently inventing solver or device parameters.
They do not change the selected updater's computation.

## Implementation map

The solver code is included in this repository rather than imported from the
larger research workspace. Following the original library layout, every
coordinate updater and the minimizer selector live in the single canonical
`model/resistive/minimizer.py` module:

- [`_load_lambertw`](../repro/vendor/model/resistive/minimizer.py#L19) selects
  `torch.special.lambertw` when available and otherwise the pinned
  `torchlambertw` backend.
- [`Float64ExponentialDoubleDiodeUpdater`](../repro/vendor/model/resistive/minimizer.py#L1055)
  implements the paper's float64 antiparallel double-Shockley update.
- [`Float32ExponentialDoubleDiodeUpdater`](../repro/vendor/model/resistive/minimizer.py#L1158)
  implements the mixed-precision alternative.
- [`ConfigurableExponentialSingleDiodeUpdater`](../repro/vendor/model/resistive/minimizer.py#L1298)
  implements the forward/reverse single-Shockley split.
- [`ExperimentalIVCurveUpdater`](../repro/vendor/model/resistive/minimizer.py#L1527)
  implements the measured/PWL damped-Newton solve.
- [`QuadraticMinimizer`](../repro/vendor/model/resistive/minimizer.py#L1645)
  selects the updater from `non_linearity` and the updater-name fields.
- [`load_iv_data`](../repro/iv_data.py#L12) keeps measured-I-V file loading at
  the reproduction boundary rather than coupling it to the model module.
- [`_build_minimizer`](../repro/train.py#L542) passes the checked JSON values
  into those implementations for training.

Class and function names are the stable navigation points if later edits move
the linked lines.

## Lambert W, `z`, and `z_thresh`

The Shockley coordinate subproblem has a closed-form solution involving the
principal real branch of the Lambert W function,

```text
W0(z) exp(W0(z)) = z .
```

For each node, the updater forms a positive, dimensionless argument of the
general form

```text
z = I_s / (2 a V_t) * exp(s),
```

where `a > 0` is the local quadratic coefficient and `s` depends on the linear
coefficient, diode orientation, `V_t`, and `V_off`. The exact forward,
reverse, and antiparallel expressions are implemented in
`repro/vendor/model/resistive/minimizer.py`.

`z_thresh` is an **algorithmic branch threshold**, not a physical diode
parameter and not a clamp on `z`:

- for `z <= z_thresh`, the updater evaluates `W0(z)` with the configured
  Lambert-W backend in float64;
- for `z > z_thresh`, it evaluates the four-term large-`z` expansion

```text
L1 = log(z)
L2 = log(L1)
W0(z) ≈ L1 - L2
        + L2 / L1
        + L2(-2 + L2) / (2 L1²)
        + L2(6 - 9 L2 + 2 L2²) / (6 L1³).
```

The asymptotic path avoids asking a general Lambert-W implementation to handle
extreme arguments while retaining increasing accuracy as `z` grows. Relative
errors of this four-term expression, evaluated against SciPy's principal
branch, are approximately:

| `z` | Relative error |
|---:|---:|
| `1e4` | `2.3e-5` |
| `1e8` | `2.7e-6` |
| `1e10` | `1.1e-6` |
| `1e12` | `5.2e-7` |

The paper configurations use `z_thresh=1e10`. Lowering the threshold sends
more nodes through the faster asymptotic formula but uses it where it is less
accurate. Raising it sends more nodes through the backend and should be treated
as a numerical-method change. The runtime requires `z_thresh > 1`; values near
one are mathematically legal for `W0` but unsuitable for this large-argument
series.

## `exp_clip`: a separate overflow guard

Before constructing `z`, the updater limits the dimensionless exponential
argument `s` to `[-exp_clip, exp_clip]` where both signs are needed. Thus,
unlike `z_thresh`, this value can alter the computed current-balance equation
when the guard is reached:

```text
z = I_s / (2 a V_t) * exp(clip(s, -exp_clip, exp_clip)).
```

It protects `exp(s)` from overflow and non-finite coordinate updates. Useful
reference values are `exp(100) ≈ 2.69e43` and `exp(165) ≈ 4.56e71`. Float64
overflows near an exponent of 709.8, whereas float32 overflows near 88.7.
Consequently, a double-Shockley `float32` updater should use a substantially
smaller clip, commonly around 80, unless its full operating range has been
validated. The paper's double-Shockley training presets use float64 and
`exp_clip=165`.

Increasing `exp_clip` preserves a wider range of the nominal Shockley equation
but reduces the overflow margin. Decreasing it improves robustness but can
clip physically meaningful currents. A changed value is therefore part of the
experiment definition and must be reported with results.

## Optional Newton polish

The Lambert-W result is already the closed-form coordinate solution. Setting
`use_polish=true` additionally evaluates the original current-balance residual
and applies up to `max_newton_iters` Newton corrections. This can reduce
residual error from the asymptotic branch or finite precision, but adds work
and can be less robust when an exponent is already near its clip.

The paper training presets use:

```json
"use_polish": false,
"max_newton_iters": 32
```

With polishing disabled, `max_newton_iters` is retained as provenance but is
inactive. Setting it to zero is valid and disables correction even if
`use_polish=true`.

## Measured/PWL Newton settings

The measured nonlinearity loads voltage and current samples from
`iv_data_path`, interpolates each segment linearly, and solves the local
current-balance equation iteratively.

Set `iv_data_path` in a copied `default_custom_iv.json` when the curve belongs
to that local parameter source. To leave the JSON unchanged or try another
curve for one run, pass `--iv-data-path PATH`; the command-line value overrides
the JSON field.

The `.npz` file must contain either one-dimensional `i` and `v` arrays of equal
length, or one `iv` array shaped `(2, N)` with current in row 0 and voltage in
row 1. Both layouts require at least two real numeric, finite samples. Voltage
must be strictly increasing and current must be nondecreasing; the loader
rejects invalid ordering rather than sorting or changing the curve. The compact
runner can replace the source curve with `--iv-data-path PATH` (or
`DRNRunSpec.iv_data_path`) only when the selected nonlinearity is measured/PWL.
Relative paths are resolved from the repository root. Paths inside the
repository are serialized relative to its root, external paths remain
absolute, and the generated metadata records `runner.iv_data_source` as `user`
or `parameter-source`.

- `damping` multiplies the Newton step. `1` is a full step; values between zero
  and one damp potentially unstable steps.
- `experimental_newton_max_steps` caps work per coordinate update.
- `experimental_newton_tol` stops when the absolute voltage change between
  consecutive steps falls below the configured value.
- `overrelaxation_factor` is applied after the PWL coordinate solve only when
  `double_diode_updater` is `overrelaxed`.

These settings are unrelated to Lambert W. Conversely, `z_thresh`,
`exp_clip`, `use_polish`, and `max_newton_iters` do not affect measured/PWL
runs.

## Equilibrium iteration settings

Training requires `adaptive_equilibrium=false`. Every free and nudged phase
therefore performs exactly `num_iterations` coordinate-descent sweeps. This is
important for reproducing the paper's computational budget and gradient
estimator.

When `num_iterations` is omitted from the compact runner, Digits uses four
sweeps for one hidden layer, eight for two or three, and the selected parameter
source's value for four or more. An explicit value always wins. The checked
JSON configurations remain literal records and do not apply these runner-only
depth defaults.

`rel_tol` and `vn_tol` define the adaptive infinity-norm stopping condition

```text
max_abs_voltage_change <= rel_tol * max_abs_previous_voltage + vn_tol.
```

They are retained in training configurations for compatibility with numerical
replay but are inactive during training. They must not be interpreted as a
claim that the fixed-sweep training phases converged to those tolerances.

## Updater and ordering fields

| Field | Meaning |
|---|---|
| `minimizer_impl` | Archived compatibility selector. The value `custom` resolves to the canonical `model.resistive.minimizer.QuadraticMinimizer`; it no longer names a separate module or class. |
| `mode` | Layer-update ordering. Paper presets use asynchronous odd-even block updates. |
| `single_diode_updater` | Selects `custom`, `standard`, or `overrelaxed` when the single-Shockley nonlinearity is active. |
| `double_diode_updater` | Selects float precision/overrelaxation for double Shockley; for measured/PWL it selects `standard` or `overrelaxed`. |
| `overrelaxation_factor` | SOR-style factor `omega` in `v_new = v_old + omega * (v_cd - v_old)`. It is inactive for a non-overrelaxed updater. |

`omega=1` is the ordinary coordinate update. Values above one extrapolate past
it and can accelerate or destabilize convergence depending on the network.
The checked paper values should be used for reproduction rather than assumed
to transfer to arbitrary architectures.

## Architecture and electrical fields

| Field | Meaning |
|---|---|
| `layer_shapes` | Input, hidden, and output tensor shapes. Inputs are duplicated into positive/negative channels. |
| `weight_gains` | One conductance-initialization gain per connection between adjacent layers. |
| `weight_min`, `weight_max` | Conductance bounds enforced after every optimizer update. |
| `weight_init_mode` | Conductance initialization rule. |
| `input_gain` | Input multiplier after preprocessing and signed duplication. |
| `voltage_amp`, `current_amp` | Inter-layer amplifier factors used by the electrical model. |
| `quadratic_diode_param` | Explicit compatibility parameters for finite-conductance/idealized diode models. |
| `exponential_diode_param.I_s` | Shockley reverse saturation current in simulator electrical units. |
| `exponential_diode_param.V_t` | Shockley thermal-voltage scale. |
| `exponential_diode_param.V_off` | Voltage offset or turn-on shift. |
| `hard_sigmoid_param` | Explicit compatibility parameters for hard-sigmoid models. |
| `iv_data_path` | Measured/PWL `.npz` source containing 1-D `i` and `v`, or `(2, N)` `iv` in current-then-voltage order; relative paths resolve from the repository root. |

All three physical parameter dictionaries are mandatory even when inactive.
This fail-closed contract prevents a configuration from silently acquiring
invented diode values when it is reused with another nonlinearity.

## Training fields

| Field | Meaning |
|---|---|
| `dataset` | Digits or MNIST selection, subset controls, and normalization. |
| `batch_size` | Mini-batch size. |
| `num_epochs` | Complete passes over the training split. |
| `learning_rates` | Optimizer rates in conductance-tensor order followed by hidden-bias order. |
| `scheduler_gamma` | Multiplicative rate decay after each epoch; `1` disables decay. |
| `nudging` | Magnitude of the equilibrium-propagation perturbation. |
| `ep_variant` | Positive, negative, or centered equilibrium-propagation estimator. |
| `seed` | Initialization, split, and loader-order seed. |

For `H` hidden layers, the dense runner expects `H+1` conductance rates followed
by `H` hidden-bias rates, for `2H+1` values total. The MNIST paper preset's
effective ordering is documented separately in `docs/MNIST_REPRODUCTION.md`.

## Validation and archived configurations

The runtime rejects non-positive tolerances and clip values, `z_thresh <= 1`,
negative Shockley polish counts, non-positive measured/PWL step limits,
malformed or nonmonotone I-V data, and training configurations that enable
adaptive equilibrium. Error messages state the expected format before
reporting the provided value.

The JSON files below `data/error_vs_iter/configs/` and
`data/timing/configs/` are archived numerical-replay records rather than
editable training templates. They may contain historical instrumentation keys
such as `dynamic_polish` or Anderson-acceleration settings. Only fields loaded
by `repro.config.RuntimeConfig` affect standalone replay; the extra keys are
retained for provenance.
