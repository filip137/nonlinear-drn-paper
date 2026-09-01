# Strict version-2 configuration reference

This document describes the scientific configuration contract shared by
training, checkpoint replay, the public demo, and hand-specified small-network
simulation. The authoritative machine-readable schemas live in
[`configs/schema/`](../configs/schema/).

## Design rules

Version 2 has four invariants:

1. Every scientific value has one owner.
2. Referenced sources and assets are identified by path and exact SHA-256.
3. Execution uses a fully expanded, revalidated snapshot.
4. Runtime APIs do not invent numerical defaults or read scientific values
   from process environment.

JSON parsing rejects duplicate object keys and non-finite constants before
schema validation. The schemas reject unknown keys, missing required keys,
wrong types (including numeric strings), and fields belonging to another
branch of a typed union. JSON does not perform string-to-number or
integer-to-boolean coercion.

Every top-level document carries `schema_version: 2` and a `kind` discriminator
except the replay manifest, whose schema fixes those values without an editor
`$schema` field. Bundled training, simulator, execution, and small-network
documents include `$schema` for editor support.

## Schema map and ownership

| Schema | Kind | Owns |
|---|---|---|
| [`training-v2.schema.json`](../configs/schema/training-v2.schema.json) | `training` | data, model, training algorithm, equilibrium |
| [`simulator-v2.schema.json`](../configs/schema/simulator-v2.schema.json) | `simulator_profile` | device law, amplification, updater |
| [`execution-v2.schema.json`](../configs/schema/execution-v2.schema.json) | `execution_profile` | device, dtype, determinism, threads, loader workers, seeding |
| [`replay-v2.schema.json`](../configs/schema/replay-v2.schema.json) | `replay` | replay data/model/simulation/equilibrium |
| [`manifest-v2.schema.json`](../configs/schema/manifest-v2.schema.json) | `replay_manifest` | job selection, asset refs, execution ref, job overrides |
| [`small-network-v2.schema.json`](../configs/schema/small-network-v2.schema.json) | `small_network` | literal network plus complete simulation/equilibrium/execution |
| [`common-v2.schema.json`](../configs/schema/common-v2.schema.json) | definitions only | shared typed blocks and references |

The training source deliberately does not contain physical or execution
values. A simulator profile deliberately does not contain equilibrium policy,
data, or training values. Composition fails if a base and a reference both
claim the same top-level section.

## Path-and-hash references

A configuration or scientific-asset reference is exactly:

```json
{
  "path": "configs/simulator/default_double_shockley.json",
  "sha256": "6415fbebbf2f3c751f012f03577fee23f92c0a5c95daeb8b861325441c0fdd06"
}
```

Paths are repository-relative, may not escape with `..`, and must name an
existing file inside the repository. The resolver hashes the exact file bytes
before parsing it. A renamed, reformatted, or edited file therefore requires a
new digest; a matching path alone is insufficient.

Training sources have `simulation_ref` and `execution_ref`. A measured/PWL
updater has `curve`, also a path-and-hash reference. The replay manifest hashes
every base config, weight checkpoint, optional SPICE reference, and its
execution profile. `data/checksums.sha256` is the separate, artifact-wide
integrity index; checksums are not duplicated inside the manifest.

## Source, expanded snapshot, and receipt

A training source contains the reference fields and omits inline `simulation`
and `execution`. Resolution performs this sequence:

1. parse and validate the source;
2. resolve and schema-validate both referenced profiles;
3. verify their exact hashes;
4. insert the owned `simulation` and `execution` sections;
5. add the source records to `provenance.config_sources`; and
6. validate the expanded document against the training schema again.

The executable snapshot contains inline `simulation` and `execution` and no
reference fields. It is self-contained: reloading it does not consult the
original profiles. Training writes it as `config.resolved.json` before loading
data or constructing the model.

The snapshot's canonical JSON hash appears in `run_receipt.json`. The receipt
also records source and asset hashes, data and split fingerprints, git commit
and dirty-state hash, Python/platform details, all installed package versions,
the effective execution profile, native threadpools, PyTorch determinism, and
CUDA runtime/device details when applicable. Receipts describe an execution;
they never supply values back to it.

## Explicit overrides

Scientific CLI changes have the form:

```text
--override JSON_POINTER=JSON_VALUE
```

The pointer follows RFC 6901 and the right-hand side is parsed as JSON. For
example:

```bash
--override '/training/epochs=1'
--override '/model/input_gain=20.0'
--override '/model/layer_shapes=[[128],[64],[20]]'
--override '/training/loader/train_shuffle=false'
```

An override may replace only an existing target. Duplicate pointers,
parent/child pointer pairs, malformed escapes, non-finite values, generated
provenance fields, and a result that fails the full schema are rejected. All
accepted changes are copied to `provenance.generation_overrides` in the
expanded snapshot. Overrides cannot be stacked on a snapshot that already has
that generated record; create a new source when another scientific change is
needed.

Use `scripts/train_drn.py --write-config` to materialize and inspect the
complete document without loading data or training.

## Canonical vocabulary

Version 2 accepts exact canonical values rather than spelling normalization or
human-friendly aliases.

| Concept | Canonical values |
|---|---|
| Nonlinearity | `single_diode_exponential`, `double_diode_exponential`, `experimental` |
| Shockley updater | `lambert_w_v1` |
| Measured/PWL updater | `piecewise_linear_newton_v1` |
| Equilibrium | `fixed_sweeps`, `voltage_change` |
| Update order | `asynchronous`, `synchronous`, `forward`, `backward` |
| State/work dtype | `float32`, `float64` |
| Output encoding | `single_ended`, `differential_pair` |
| Training estimator | `positive`, `negative`, `centered` |
| Bias scale mode | `legacy`, `constant` |
| Bias interaction | `linear`, `quadratic` |

Precision and overrelaxation are data, not updater names: `updater.dtype`
chooses the work precision and `updater.relaxation` gives the numerical factor.
`relaxation: 1.0` is the ordinary coordinate update; values above one
extrapolate past it and can accelerate or destabilize convergence.

## Training source

A composed training source has these top-level fields:

```text
$schema, schema_version, kind, description,
data, model, training, equilibrium,
simulation_ref, execution_ref, provenance
```

An expanded training snapshot replaces the two references with `simulation`
and `execution`.

### `data`

Digits uses:

- `source: sklearn_digits`;
- an explicit `float32` or `float64` affine preprocessing block (`divisor`,
  `multiplier`, `offset`); bundled sources use `float32`;
- `subset`, either `all` or seeded random `count`;
- a seeded-random split with `train_fraction` and explicit `rounding`;
  bundled sources use `rounding: floor`; and
- `seed`, shared by initialization, split, and loader order.

MNIST uses:

- `source: torchvision_mnist` and a repository-relative storage `path`;
- either `float32` or `float64` `to_tensor`/`normalized_tensor` preprocessing
  with `mean`, `standard_deviation`, and final `scale`; bundled sources use
  `float32`;
- `subset`, either `all` or explicit train/evaluation prefix counts;
- the official train/evaluation split; and
- `seed`.

The paper MNIST transform is
`0.3 * (pixel - 0.1307) / 0.3081`. Dataset download permission is operational
and remains the explicit `--download` CLI switch.

### `model`

`model` makes the previously implicit construction policy explicit:

- `layer_shapes` includes input, every hidden layer, and output;
- `state_dtype` must match `execution.backend.default_dtype`;
- `weight_gains` has one initialization gain per adjacent-layer connection;
- `weight_bounds` contains `minimum` and `maximum` conductance;
- `weight_initialization` is `kaiming_uniform` for the bundled sources;
- `input_gain` is applied after preprocessing and signed input duplication;
- `bias` fixes whether bias is enabled, scale mode (`legacy` or `constant`),
  interaction type, initialization, and bounds;
- `topology: dense`, `input_encoding: signed_pair`, and
  `amplification_learning` state the construction and trainability policy;
- `signed_weights` selects one conductance or a positive/negative pair;
- `output` fixes encoding and class count; and
- `loss` is the explicit `squared_error`/`mean` policy.

For `H` dense hidden layers with unsigned weights, there are `H+1`
conductance tensors followed by `H` hidden-bias tensors. The optimizer therefore
expects `2H+1` learning rates in that parameter order. A zero rate freezes a
parameter without removing it.

### `training`

`training` contains:

- `epochs`;
- `batch_limits.train` and `.evaluation`, where `null` means every batch;
- loader batch size, train/evaluation shuffle, and `drop_last`;
- equilibrium-propagation variant, nudging magnitude, cost-nudging mode, and
  standard/alternative gradient formula;
- the complete SGD policy, including learning rates, momentum, dampening,
  weight decay, zeroing, Nesterov, maximize, foreach, differentiable, and fused
  behavior;
- exponential scheduler `gamma`, initial epoch, and exact `after_epoch`
  timing; and
- final/best checkpoint policy, metric/mode, and deterministic tie break.

For example, the bundled one-hidden-layer policies spell out every field
consumed by PyTorch and the checkpoint selector:

```json
{
  "optimizer": {
    "method": "sgd",
    "learning_rates": [0.01, 0.01, 0.01],
    "momentum": 0.0,
    "dampening": 0.0,
    "weight_decay": 0.0,
    "zero_grad_set_to_none": true,
    "nesterov": false,
    "maximize": false,
    "foreach": false,
    "differentiable": false,
    "fused": false
  },
  "scheduler": {
    "method": "exponential",
    "gamma": 1.0,
    "step_timing": "after_epoch",
    "initial_epoch": -1
  },
  "checkpoint": {
    "save_final": true,
    "save_best": {
      "metric": "evaluation_accuracy",
      "mode": "max",
      "tie_break": "first"
    }
  }
}
```

`tie_break: first` means that an equal best metric retains the earliest
checkpoint rather than replacing it.

Data-loader worker count, persistence, and pinned-memory policy live in the
execution profile, not the training block.

### `equilibrium`

The fixed-budget form is:

```json
{
  "initial_state": "zeros",
  "method": "fixed_sweeps",
  "update_order": "asynchronous",
  "sweeps": 4
}
```

The adaptive form is:

```json
{
  "initial_state": "zeros",
  "method": "voltage_change",
  "update_order": "asynchronous",
  "max_sweeps": 128,
  "relative_tolerance": 1e-5,
  "absolute_tolerance": 1e-6
}
```

The adaptive stop condition is

```text
max_abs_voltage_change
    <= relative_tolerance * max_abs_previous_voltage
       + absolute_tolerance.
```

Both branches declare `initial_state`; version 2 currently supports the exact
value `zeros`. Training sources use fixed sweeps so every free and nudged phase
has the same declared coordinate-update budget. Replay and small-network
configs may use either branch. `fixed_sweeps` supports all four declared update orders;
`voltage_change` requires `asynchronous`, because the other minimizer orders do
not implement adaptive stopping. Inactive tolerances do not appear in a
fixed-sweep block.

## Simulator profile

A simulator profile contains only:

```text
$schema, schema_version, kind, description, simulation, provenance
```

`simulation` is a strict union discriminated by `nonlinearity`. Every branch
contains `amplification.voltage_factor` and `.current_factor`, plus exactly one
physical and updater block.

### Shockley physical block

Single and double Shockley use:

```json
{
  "saturation_current": 1e-6,
  "thermal_voltage": 0.05,
  "offset_voltage": 0.8
}
```

These are the physical `I_s`, `V_t`, and voltage offset expressed with
unambiguous names. The updater has:

- `method: lambert_w_v1`;
- explicit `backend` and work `dtype`;
- `relaxation`;
- `linear_coefficient_clamp`;
- `exponent_clip`;
- `asymptotic_threshold` and fixed `asymptotic_terms: 4`; and
- `polish`, either `false` or the family-specific typed Newton policy.

Single Shockley uses its implemented canonical float64 path and additionally
declares `quadratic_coefficient_min`. Its polish block has absolute and relative
residual tolerances. Double Shockley permits float32 or float64 work and its
polish uses an explicit batch-scaled absolute residual rule and coefficient.
Those fields cannot appear in the other family.

### Lambert W threshold and exponent clip

The coordinate solution uses the principal real Lambert-W branch,

```text
W0(z) exp(W0(z)) = z,
z = I_s / (2 a V_t) * exp(s).
```

`asymptotic_threshold` is a branch threshold, not a clamp on `z`. At or below
the threshold, the configured backend evaluates `W0`; above it, the updater
uses the declared four-term large-argument expansion:

```text
L1 = log(z)
L2 = log(L1)
W0(z) ≈ L1 - L2
        + L2 / L1
        + L2(-2 + L2) / (2 L1²)
        + L2(6 - 9 L2 + 2 L2²) / (6 L1³).
```

`exponent_clip` separately limits the argument supplied to `exp`. It can alter
the current-balance equation when reached and is therefore a scientific
setting. The paper float64 double-Shockley profiles use 165; the single profile
uses 100. Float32 has much less exponent headroom and must be validated at its
intended operating range.

The Lambert-W value is already a closed-form coordinate solution. A configured
polish performs bounded Newton residual corrections; `polish: false` omits all
polish-only fields and work.

### Measured/PWL branch

The physical block is exactly:

```json
{"representation": "measured_piecewise_linear"}
```

Its `piecewise_linear_newton_v1` updater contains:

- a hashed `curve` reference;
- `extrapolation`, either `clamp` or `linear`;
- `nonconvergence_policy`, either `accept_last` or `error`;
- Newton `damping`, `max_steps`, and `voltage_tolerance`; and
- post-solve `relaxation`.

The asset must contain equal-length 1-D `i` and `v` arrays or one `(2, N)` `iv`
array with current first. It needs at least two real finite samples, strictly
increasing voltage, and nondecreasing current. The loader never sorts or
repairs a curve. The updater interpolates `I(v)` and solves

```text
2 a v + b + I(v) = 0.
```

If Newton reaches `max_steps` without meeting `voltage_tolerance`, `error`
raises immediately; `accept_last` continues with the last finite iterate. The
bundled migrated profile explicitly preserves the historical `accept_last`
behavior.

Lambert-W, Shockley-polish, and exponent fields are invalid in this branch.

## Execution profiles

Bundled profiles are documented under
[`configs/execution/`](../configs/execution/README.md). Both fix:

- backend device, CPU construction device, and default dtype;
- deterministic-algorithm policy and whether warnings are allowed;
- PyTorch intra/inter-op and OMP thread counts plus one `blas` count applied to
  both MKL and OpenBLAS;
- data-loader workers, persistence, pinned memory, prefetch factor, and
  timeout; and
- Python, NumPy, and PyTorch seeding from the config seed.

The CUDA branch additionally fixes device index, cuDNN deterministic and
benchmark flags, TF32, and `CUBLAS_WORKSPACE_CONFIG`. A deterministic operation
that is unavailable fails rather than selecting another algorithm silently.

## Migration provenance

Migration metadata is descriptive and is never read as scientific input. Each
bundled migrated simulator or training source records
`provenance.migration.tool`, `.from_schema_version`, and `.to_schema_version`.
Its sorted `provenance.materialized_defaults` array names the exact active
fields whose effective version-1 values had lived in code, runtime fallbacks,
or implicit construction policy. The list is family- and dataset-specific: for
example, only Digits sources name affine preprocessing and split rounding, only
single-Shockley profiles name `quadratic_coefficient_min`, and only the PWL
profile names extrapolation and nonconvergence policy.

`scripts/migrate_training_configs.py` rewrites the four simulator profiles
first, embeds their resulting exact hashes in all ten training sources, checks
that every provenance path exists, validates each source and nested reference,
and is byte-idempotent. Any future migration must preserve that order so
downstream hashes describe the final profile bytes.

## Replay and manifest

All archived replay bases are explicit version-2 `replay` documents. Historical
keys that were never consumed by science code live only under
`provenance.historical_unused`. Values that were formerly implicit are listed
under `provenance.materialized_defaults`.

`data/manifest.json` names each base config and artifact by hash. A job override
owns only the swept equilibrium limit, eliminating duplicated family, depth,
tolerance, and relaxation labels. The manifest also declares the demo job,
demo batch size, and execution-profile reference. The runtime validates the
base, applies the typed job change, inserts execution, and validates the
expanded replay snapshot before loading weights.

## Small-network document

A `small_network` document is already fully expanded. Its `network` block
contains literal `layer_sizes`, conductance matrices, input-voltage rows,
`input_gain`, explicit `bias: {"method": "none"}`, `seed`, and `state_dtype`;
it then embeds complete `simulation`, `equilibrium`, and `execution` blocks.
There are no numerical function arguments and no fallback defaults.
`simulate_small_network` accepts only a validated mapping or JSON path and
returns voltages, convergence information, and an in-memory receipt without
writing files.

See the checked
[`configs/small_network/example.json`](../configs/small_network/example.json)
for a complete document.

## Validation and local variants

Keep bundled sources unchanged. Put generated experiments under the
git-ignored `configs/local/` directory. When a referenced file changes, update
both path and digest. A convenient audit is:

```bash
sha256sum configs/local/my_profile.json
python scripts/train_drn.py \
  --config configs/local/my_training.json \
  --write-config configs/local/my_training.resolved.json
```

The second command validates the source and every reference, writes a complete
snapshot, and exits before scientific work. It is the recommended boundary
between configuration generation and execution.
