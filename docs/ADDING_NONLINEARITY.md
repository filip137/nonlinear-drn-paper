# Adding a nonlinearity

There are two extension routes. Use a sampled measured/PWL curve when the
device can be represented by one passive current-voltage relation; this needs
no simulator-code change. Add an analytic family only when an exact law,
specialized coordinate solve, or additional physical parameters are essential.

Both routes use the strict version-2 configuration boundary. New scientific
values must have one typed owner and may not be introduced as an environment
variable, implicit Python default, free-form provenance key, or runtime-only
flag.

## Use a sampled I–V curve

Store the curve inside the repository in one of two NPZ layouts:

- equal-length one-dimensional `i` and `v` arrays; or
- one `iv` array shaped `(2, N)`, with current in row 0 and voltage in row 1.

Both layouts need at least two real, finite samples. Voltage must be strictly
increasing and current must be nondecreasing. These constraints make every
interpolated segment single-valued and passive. The loader rejects duplicate
voltages and negative slopes; it never sorts, clips, or repairs measurements.

For a local experiment:

```bash
mkdir -p configs/local
python - <<'PY'
import numpy as np

voltage = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
current = np.array([-2.0e-3, -2.0e-4, 0.0, 2.0e-4, 2.0e-3])
np.savez("configs/local/my_curve.npz", i=current, v=voltage)
PY

sha256sum configs/local/my_curve.npz
cp configs/simulator/default_pwl.json configs/local/my_pwl.json
```

Edit `configs/local/my_pwl.json` so
`simulation.updater.curve` contains the path and printed digest:

```json
{
  "path": "configs/local/my_curve.npz",
  "sha256": "<exact digest printed above>"
}
```

The same updater block explicitly owns the range and numerical policy:

```json
{
  "method": "piecewise_linear_newton_v1",
  "curve": {
    "path": "configs/local/my_curve.npz",
    "sha256": "<exact curve digest>"
  },
  "extrapolation": "clamp",
  "nonconvergence_policy": "accept_last",
  "damping": 1.0,
  "max_steps": 32,
  "voltage_tolerance": 0.01,
  "relaxation": 1.0
}
```

`extrapolation` accepts only `clamp` or `linear`. If Newton exhausts
`max_steps`, `nonconvergence_policy: error` aborts the solve, while
`accept_last` continues with the last finite iterate. Choose this explicitly
for a new profile; `accept_last` above matches the bundled migrated profile.

Hash the completed profile, copy a matching training source, and update its
`simulation_ref`:

```bash
sha256sum configs/local/my_pwl.json
cp configs/train/default_custom_iv.json configs/local/my_pwl_training.json
# Put the my_pwl.json path and printed hash in simulation_ref, then resolve:
python scripts/train_drn.py \
  --config configs/local/my_pwl_training.json \
  --write-config configs/local/my_pwl_training.resolved.json
```

On MNIST, start from `configs/train/default_mnist_custom_iv.json`. It is a
Digits-derived starting point, not an audited paper-MNIST result.

There is intentionally no separate curve CLI parameter or Python numerical
argument. Selecting another curve changes the simulator profile and therefore
its hash. The resolved snapshot embeds the verified curve reference, and the
run receipt records the curve bytes again as an executed asset.

The measured updater interpolates `I(v)` and solves the coordinate current
balance

```text
2 a v + b + I(v) = 0,
```

where `a v² + b v` is the contribution from the rest of the network. See the
[configuration reference](CONFIG_REFERENCE.md#measuredpwl-branch) for the
Newton and extrapolation semantics.

## Implement an analytic current law

An analytic grounded element needs a mutually consistent current, energy, and
coordinate update. Define the current law `I(v; θ)` and an energy primitive

```text
U(v; θ) = integral from 0 to v of I(u; θ) du,
```

so that `dU/dv = I(v)`. Adding `U` to the network energy keeps the free and
nudged phases, equilibrium-propagation gradients, and reported energy
consistent with the coordinate solver.

For the usual coordinate-descent guarantees, choose a nondecreasing current
law. Then `U` is convex and, because the network contributes `a > 0`,
`2 a v + b + I(v)` is strictly increasing and has at most one root. A
nonmonotone or active device needs an explicit root-selection and stability
contract and lies outside the bundled updater assumptions.

### 1. Add the energy interaction

In `repro/vendor/model/function/interaction.py`, add a `Function` subclass for
the grounded element. Its `eval()` must sum `U(v)` over nodes and return one
value per batch item; `grad_layer_fn()` must return the matching `I(v)`. Follow
the existing Shockley interactions for dtype/device and layer-depth amplifier
scaling.

The specialized updater supplies the nonlinear part of the coordinate
equation, so the interaction's quadratic coefficient hooks must not count
`I(v)` a second time. Register the interaction in `DeepResistiveEnergy` in
`repro/vendor/model/resistive/network.py`, and allow the canonical family in
the non-clipping branch of `repro/vendor/model/resistive/layer.py`.

Test the primitive and current with finite differences before writing the
coordinate solver.

### 2. Add the coordinate updater

In `repro/vendor/model/resistive/minimizer.py`, add a `LayerUpdater` that reads
`a` and `b` from the function and solves, for nonlinear hidden layers,

```text
F(v) = 2 a v + b + I(v) = 0.
```

Use an exact inverse when stable; otherwise use a bracketed method or
safeguarded Newton with derivative `2 a + dI/dv`. Keep it vectorized, preserve
the input tensor's dtype/device, use config-owned stopping criteria, reject
non-finite states, and make output/pooling layers retain the quadratic result
`-b/(2a)`. Apply configured relaxation only after a valid coordinate solution.

The vendored minimizer is an implementation layer, not public configuration
vocabulary. Public selection belongs at the single translation boundary in
`repro/minimizer_factory.py`. Add a new exact method/family branch there; do not
add spelling aliases or a fallback selector.

### 3. Add a typed simulation branch

Extend `configs/schema/common-v2.schema.json` with:

- one physical block containing every required physical parameter;
- one updater block containing every active numerical parameter;
- one simulation object with a unique canonical `nonlinearity`; and
- the new object as one branch of the shared `simulation` union.

Use `additionalProperties: false` and require every field. Conditional or
optional blocks should represent a real algorithmic branch, as `polish: false`
versus a typed Newton object does. Do not put fields from existing families in
the new block merely to satisfy an aggregate Python object.

The simulator-profile, training-expanded, replay-expanded, and small-network
schemas all reuse the shared union and therefore gain the same vocabulary.
Update model/output relations if the family requires another orientation or
encoding. A checkpoint whose topology or physical law changes is a different
artifact and needs a separate hash and provenance record.

### 4. Thread every scientific entry point

Training, replay, demo, and small-network simulation must construct the same
energy interaction and call `repro.minimizer_factory.build_minimizer` with the
same typed block. Update:

- the energy builder in `repro/train.py`;
- replay construction in `repro/digits_validate.py`;
- the literal-network builder in `repro/small_network.py`;
- cost/output validation when encoding changes; and
- receipt asset discovery when the new family references files.

Add a simulator profile and a training source only after the runtime accepts a
fully expanded document. The training source must reference the profile by
path and SHA rather than copying its fields. Replay jobs similarly reference a
hashed base config and checkpoint through `data/manifest.json`.

If existing archived documents need a new active value, write a deterministic
migration that materializes the previously effective value, records migration
facts and exact active field paths under `provenance.migration` and
`provenance.materialized_defaults`, updates all dependent hashes, and validates
every output. Run the migration twice and require identical bytes on the second
pass. Do not add a runtime compatibility adapter that silently restores a
default.

### 5. Verify the extension

At minimum, add tests for:

- `dU/dv = I(v)` by autograd or finite differences over the intended range and
  branch boundaries;
- `dI/dv` when Newton uses an analytic derivative;
- current-balance residuals after a coordinate update on CPU and supported
  CUDA;
- finite results at extreme valid parameters and clear rejection of invalid
  parameters;
- strict schema behavior: missing, extra, inactive-family, alias, numeric-string,
  duplicate-key, and non-finite cases;
- exact path-and-hash verification for referenced assets and profiles;
- deterministic hostile-environment tests proving that process variables cannot
  change a resolved config's behavior;
- one-batch training with finite equilibrium-propagation gradients;
- one replay and one small-network solve with a receipt; and
- equivalence between the pre-migration effective settings and the new resolved
  snapshot when extending an existing family.

Changing a current law, physical parameterization, coordinate solver, or
numerical policy defines a new experiment. Keep its configs, checkpoints, and
results separate from the checked paper records unless they satisfy exactly the
same physical and solver contract.
