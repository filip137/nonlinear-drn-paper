# Resolve-then-train guide

`scripts/train_drn.py` is a thin interface around strict version-2 training
configurations. It does not assemble an experiment from independent dataset,
device, architecture, or solver flags. Instead it validates one source,
verifies its simulator and execution hashes, expands those references, applies
explicit recorded overrides, validates the complete snapshot, writes that
snapshot, and only then starts scientific work.

The same rules are used by `python scripts/reproduce.py train` and the public
Python functions in `repro`.

## Run a bundled source

Digits is bundled with scikit-learn and requires no download:

```bash
python scripts/train_drn.py \
  --config configs/train/digits_double_shockley.json
```

The source declares the dataset transform and split, width-32 architecture,
batch size 32, 15 epochs, four equilibrium sweeps, centered equilibrium
propagation, learning rates, scheduler, double-Shockley device, and deterministic
CPU execution policy. None of those values comes from the command line.

The other checked Digits sources are:

```text
configs/train/digits_single_shockley.json
configs/train/digits_double_shockley.json
configs/train/digits_pwl.json
```

The six `default_*.json` files under `configs/train/` are editable-starting-point
sources. Exact paper MNIST reproduction uses
`configs/train/mnist_paper_double_shockley.json`. Only that double-Shockley
MNIST source has audited paper parameters; the single-Shockley and PWL MNIST
defaults adapt Digits settings and define new experiments.

## Inspect or generate without training

Use `--write-config` to make the resolve/generate boundary visible:

```bash
mkdir -p configs/local
python scripts/train_drn.py \
  --config configs/train/digits_single_shockley.json \
  --write-config configs/local/digits_single.resolved.json
```

This command:

1. strictly parses and validates the training source;
2. verifies the exact simulator and execution profile hashes;
3. inserts their owned sections;
4. records both sources under `provenance.config_sources`;
5. validates cross-field relations and the expanded schema; and
6. writes a self-contained snapshot without loading Digits or constructing a
   network.

The resulting expanded document has inline `simulation` and `execution` and no
reference fields. It can be executed directly:

```bash
python scripts/train_drn.py \
  --config configs/local/digits_single.resolved.json
```

## Explicit scientific changes

One-off changes use repeatable JSON-Pointer replacements:

```bash
python scripts/train_drn.py \
  --config configs/train/digits_double_shockley.json \
  --override '/training/epochs=1' \
  --override '/equilibrium/sweeps=2' \
  --override '/training/batch_limits/train=1' \
  --override '/training/batch_limits/evaluation=1' \
  --write-config configs/local/digits_double_smoke.json
```

The syntax is `JSON_POINTER=JSON_VALUE`. The right side is parsed as JSON:

```bash
--override '/model/input_gain=20.0'
--override '/training/loader/train_shuffle=false'
--override '/model/layer_shapes=[[128],[128],[64],[20]]'
--override '/training/optimizer/learning_rates=[0.01,0.01,0.01,0.005,0.005]'
```

Quote the whole shell argument, and include JSON quotes inside it when the new
value is a string. Replacements must target existing fields. Duplicate or
overlapping pointers are order-dependent and rejected. The final document must
still satisfy the strict schema and training relations.

Every accepted replacement is stored in
`provenance.generation_overrides`. A snapshot that already contains this record
can be run as-is but cannot receive another layer of overrides; make a new
source instead of stacking provenance.

### Architecture and learning-rate changes

Architecture is never inferred from a width flag. Replace or edit the complete
`model.layer_shapes` array, then keep all dependent arrays consistent:

- `model.weight_gains` needs one value per adjacent-layer matrix;
- `model.output` determines whether output width is `classes` or `2 * classes`;
- a single-Shockley model requires every hidden width to be even; and
- `training.optimizer.learning_rates` needs one value per trainable parameter.

For `H` dense hidden layers with unsigned weights, parameter order is `H+1`
conductance tensors followed by `H` hidden biases, for `2H+1` rates. For two
hidden layers, a valid role-aware list might be:

```json
[0.01, 0.01, 0.005, 0.002, 0.001]
```

Zero freezes a selected tensor. Changing depth, width, gains, rates, or sweep
budget defines a new experiment; there is no depth-dependent hidden fallback.

## Deterministic CPU and CUDA profiles

Every bundled training source contains a hashed `execution_ref`. The default
checked sources use `configs/execution/reference_cpu.json`, which fixes one
thread, zero loader workers, float32 model state, deterministic algorithms, and
config-derived Python/NumPy/PyTorch seeds.

For CUDA, make a new source rather than passing a device flag:

```bash
mkdir -p configs/local
cp configs/train/mnist_paper_double_shockley.json \
  configs/local/mnist_paper_double_cuda.json
sha256sum configs/execution/reference_cuda.json
```

In the copy, replace `execution_ref` with:

```json
{
  "path": "configs/execution/reference_cuda.json",
  "sha256": "e75ae3c2223163618a469ba8737ed3d2ccf162da6502a93832f72c856005cbb8"
}
```

Verify that the printed digest matches before using the snippet. Then resolve
and launch:

```bash
python scripts/train_drn.py \
  --config configs/local/mnist_paper_double_cuda.json \
  --write-config configs/local/mnist_paper_double_cuda.resolved.json

python scripts/train_drn.py \
  --config configs/local/mnist_paper_double_cuda.resolved.json \
  --download
```

Install `requirements-cuda.txt` and run
`python scripts/reproduce.py verify --device cuda` first. The CUDA profile fixes
device index 0, deterministic algorithms, cuDNN determinism/benchmark behavior,
TF32, cuBLAS workspace, threads, workers, and seeding. Unsupported deterministic
operations fail closed.

`--download` grants torchvision permission to populate the configured MNIST
path. It is operational rather than scientific and therefore is not stored as
an override.

## Dataset and preset matrix

| Dataset | Family | Training source | Hidden width | Simulator profile |
|---|---|---|---:|---|
| Digits | single Shockley | `default_single_shockley.json` | 32 | `default_single_shockley.json` |
| Digits | double Shockley | `default_double_shockley.json` | 32 | `default_double_shockley.json` |
| Digits | measured/PWL | `default_custom_iv.json` | 32 | `default_pwl.json` |
| MNIST | single Shockley | `default_mnist_single_shockley.json` | 100 | `default_single_shockley.json` |
| MNIST | double Shockley | `default_mnist_double_shockley.json` | 100 | `default_mnist_double_shockley.json` |
| MNIST | measured/PWL | `default_mnist_custom_iv.json` | 100 | `default_pwl.json` |

All six editable defaults use batch size 10. The paper-specific Digits files
use widths 64/32/32 for single/double/PWL respectively. The paper MNIST source
uses width 100, batch size 16, 100 epochs, four sweeps, input gain 50, centered
equilibrium propagation, rates `[0.15, 0.08, 0.05]`, and scheduler gamma 0.99.

Digits sources explicitly record float32 affine preprocessing, subset policy,
seeded split, `rounding: floor`, and seed. MNIST sources explicitly record the
official split, storage path, subset policy, and transform. The paper transform is
`0.3 * (pixel - 0.1307) / 0.3081`.

Every source includes the complete SGD flag set, scheduler method/timing and
initial epoch, and checkpoint metric/mode/tie-break policy. No omitted PyTorch
or checkpoint-library default is part of the experiment contract; see the
[complete policy blocks](CONFIG_REFERENCE.md#training).

## Local reusable variants and hashes

Use `configs/local/`, which is git-ignored, for generated experiments. A
training-only edit needs no simulator copy. A physical or updater edit does:

```bash
cp configs/train/default_mnist_single_shockley.json \
  configs/local/my_mnist_single.json
cp configs/simulator/default_single_shockley.json \
  configs/local/my_single_simulator.json
# Edit the simulator, then calculate its exact byte hash:
sha256sum configs/local/my_single_simulator.json
```

Set `simulation_ref.path` in the training copy to
`configs/local/my_single_simulator.json` and set `simulation_ref.sha256` to the
printed digest. Reformatting the profile changes the digest; this is intentional.
The resolver verifies the profile before reading its scientific values.

Do the same for a different execution profile. A source with a correct path and
stale hash is invalid rather than silently following changed defaults.

## Custom measured I–V data

There is no standalone curve override. A curve is a hashed scientific asset
owned by the measured/PWL simulator profile:

1. put the NPZ inside the repository, for example
   `configs/local/my_curve.npz`;
2. copy `configs/simulator/default_pwl.json` into `configs/local/`;
3. replace `simulation.updater.curve.path` and `.sha256`;
4. hash the copied simulator profile; and
5. update a copied training source's `simulation_ref`.

The NPZ may contain equal-length 1-D `i` and `v`, or `(2, N)` `iv` in
current-then-voltage order. Samples must be real and finite; voltage must be
strictly increasing and current nondecreasing. The profile also explicitly
owns extrapolation, the `accept_last` or `error` nonconvergence policy, Newton
damping, maximum steps, voltage tolerance, and post-solve relaxation. The
bundled migrated profile uses `accept_last` to preserve its historical
effective behavior.

See [Adding a nonlinearity](ADDING_NONLINEARITY.md) for a complete example.

## Python API

The public API uses the same source/snapshot boundary:

```python
from repro import build_training_config, run_drn, write_training_config

overrides = (
    "/training/epochs=1",
    "/training/batch_limits/train=1",
    "/training/batch_limits/evaluation=1",
)

document = build_training_config(
    "configs/train/digits_double_shockley.json",
    overrides=overrides,
)
print(document["provenance"]["generation_overrides"])

snapshot = write_training_config(
    "configs/train/digits_double_shockley.json",
    "configs/local/python_smoke.json",
    overrides=overrides,
)
result = run_drn(snapshot)
print(result.output_dir)
print(result.receipt_path)
```

`build_training_config` returns the expanded mapping without writing.
`write_training_config` writes one complete mapping and returns its path.
`run_drn` accepts a config path and has only operational destination/download
arguments plus the same recorded override mechanism. It has no granular
numerical parameters.

## Output contract

Unless `--output` is supplied, a timestamped directory is created under
`outputs/training/`. The resolved snapshot is written first. A completed run
contains:

- `config.resolved.json`: exact executable scientific configuration;
- `history.json`: epoch-wise training/evaluation loss and accuracy;
- `model_best.pt`: checkpoint selected by configured metric/mode;
- `model.pt` when `training.checkpoint.save_final` is true;
- `run_receipt.json`: source/config/asset/data/split/environment execution
  receipt; and
- `run_metadata.json`: compact hashes and final/best metrics pointing to the
  full receipt.

Interactive terminals refresh progress for each batch. Redirected output logs
at roughly ten-percent intervals and prints a complete epoch summary. Progress
formatting and `--output` placement are operational; batch limits, loader
behavior, epochs, and checkpoint policy remain explicit scientific config.
