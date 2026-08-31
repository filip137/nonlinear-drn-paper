# Training runner

`scripts/train_drn.py` and `repro.runner.run_drn` provide a compact interface
for defining dense nonlinear-DRN experiments. The design follows the original
Scellier fast-DRN examples: the important experiment choices are visible
together, and one call builds and launches the complete run.

The runner does not replace the checked JSON configurations used for exact
paper reproduction. It expands the compact definition into
`config.generated.json`, and the training code additionally writes
`config.resolved.json`. Editable defaults compose a training template with its
referenced simulator profile. Both saved configurations contain the fully
expanded, self-contained settings and retain the simulator profile's source
path and SHA-256 hash.

## Command-line examples

### Digits

This trains the compact double-Shockley anchor. Digits is bundled with
scikit-learn and requires no download.

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity double \
  --parameter-set configs/train/digits_double_shockley.json \
  --epochs 10 \
  --batch-size 32 \
  --device cpu
```

Omitting `--hidden-sizes` inherits the parameter source's one-hidden-layer
architecture (width 32 for this double-Shockley source). The corresponding
Digits solver default is four coordinate-descent iterations.

To define a two-hidden-layer run, supply both widths:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --hidden-sizes 128 64 \
  --non-linearity double \
  --parameter-set configs/train/digits_double_shockley.json \
  --epochs 10 \
  --device cpu
```

With two or three hidden layers and no `--num-iterations` override, the runner
uses eight iterations.

For a quick end-to-end check, add:

```bash
--epochs 1 --max-batches 1 --max-eval-batches 1
```

During a run, an interactive terminal refreshes one line after every training
and evaluation batch with the running loss, running accuracy, and elapsed time.
When output is redirected to a file, progress is printed at roughly 10%
intervals instead, followed by the usual complete summary at the end of each
epoch.

### MNIST

The exact paper DRN-XS electrical and solver parameters are available for the
double-Shockley model. Before using `--device cuda`, install
`requirements-cuda.txt` in the repository `.venv` and run
`python scripts/reproduce.py verify --device cuda` as described in the
[main README](../README.md#clone-and-install).

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --hidden-sizes 100 \
  --non-linearity double \
  --parameter-set configs/train/mnist_paper_double_shockley.json \
  --epochs 100 \
  --num-iterations 4 \
  --device cuda \
  --download
```

`--download` is explicit: without it, the runner requires MNIST to exist under
`data/external/mnist/` (or the directory supplied through `--mnist-root`).

Single-Shockley and measured/PWL models can also be trained on MNIST. Their
bundled defaults are derived from the checked Digits sources. For example,
launch a single-Shockley MNIST experiment with:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-set configs/train/default_mnist_single_shockley.json \
  --device cuda \
  --download \
  --seed 0
```

This is a new MNIST experiment, not a claimed paper configuration or result.
The audited `configs/train/mnist_paper_double_shockley.json` source cannot be
substituted here because it supports only the double-Shockley nonlinearity.

The corresponding measured/PWL starting point is also dataset-specific:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity pwl \
  --parameter-set configs/train/default_mnist_custom_iv.json \
  --device cuda \
  --download \
  --seed 0
```

Both the MNIST single-Shockley and PWL templates adapt validated Digits
settings; neither is a paper-MNIST result. Only double Shockley has audited
MNIST paper parameters.

### Nonlinearity names

The short names are:

| CLI name | Stored canonical name | Output encoding |
|---|---|---|
| `single` or `single-shockley` | `single_diode_exponential` | 10 linear outputs |
| `double` or `double-shockley` | `double_diode_exponential` | 20 paired outputs |
| `pwl` or `measured` | `experimental` | 20 paired outputs |

Canonical names are accepted directly. Single-Shockley hidden widths must be
even because each layer is divided into forward- and reverse-oriented nodes.

## Dataset-specific defaults

Learning rate, input gain, architecture, batch size, and seed come from the
explicitly selected training template when their flags are omitted. Each
template references a simulator profile for its physical and solver settings:

| Dataset | Nonlinearity | Training template | Hidden width | Simulator profile |
|---|---|---|---:|---|
| Digits | single Shockley | `default_single_shockley.json` | 32 | `default_single_shockley.json` |
| Digits | double Shockley | `default_double_shockley.json` | 32 | `default_double_shockley.json` |
| Digits | measured/PWL | `default_custom_iv.json` | 32 | `default_pwl.json` |
| MNIST | single Shockley | `default_mnist_single_shockley.json` | 100 | `default_single_shockley.json` |
| MNIST | double Shockley | `default_mnist_double_shockley.json` | 100 | `default_mnist_double_shockley.json` |
| MNIST | measured/PWL | `default_mnist_custom_iv.json` | 100 | `default_pwl.json` |

Training templates are under `configs/train/`; profiles are under
`configs/simulator/`. All six editable defaults use batch size 10. The MNIST
single-Shockley and PWL rows deliberately reuse Digits-derived simulator
profiles and are new-experiment starting points. The double-Shockley row is the
only MNIST default with audited MNIST physical settings; exact paper
reproduction still uses `configs/train/mnist_paper_double_shockley.json`.

The `default` alias chooses the correct row from both `dataset` and
`non_linearity`. Explicit paths are preferred because they expose the selected
source directly:

```bash
--parameter-set configs/train/default_mnist_single_shockley.json
```

Architecture remains optional. When `--hidden-sizes` is omitted, the selected
training template supplies its architecture. Omitted Digits solver iterations
depend on the resulting depth: four for one hidden layer, eight for two or
three hidden layers, and the training template's value for four or more. An
explicit `--num-iterations` always takes precedence.

The default templates are starting points rather than a claim that one rate is
optimal for every depth and width. On the deterministic Digits split
(`seed=0`), the checked width-64/32/32 paper-Digits sources reached 93.6% test
accuracy for single Shockley and 93.1% for both double Shockley and measured/PWL
after 15 epochs and four sweeps on the Python 3.12 CPU reference stack. These
checks are not additional paper results.

A uniform source default is expanded to every conductance and bias tensor when
the architecture is resized. The single-Shockley source instead preserves the
Scellier training policy: its common conductance rate is expanded to all weight
tensors and every hidden bias remains frozen. The MNIST source preserves the
audited parameter roles: 0.15 for the first conductance, 0.08 for every later
conductance, and 0.05 for hidden biases. For substantially deeper or wider
models, or substantially narrower ones, inspect the generated config with
`--dry-run` and tune from this starting point. Only the one-hidden-layer MNIST
DRN-XS setting is a reported paper configuration.

The MNIST preset also supplies the paper's exponential scheduler
(`gamma=0.99`) and exact input transform
`0.3 * (pixel - 0.1307) / 0.3081`. See
[`MNIST_REPRODUCTION.md`](MNIST_REPRODUCTION.md) for the learning-rate ordering
audit, the gain-20 PCA distinction, and the original-seed limitation.

Use `--learning-rate` or `--input-gain` to override either default:

```bash
--learning-rate 0.005 --input-gain 20
```

Adaptive equilibrium is always set to `false` for training. A custom parameter
source that enables it is rejected, so training performs the requested fixed
number of coordinate-descent iterations.

### Layerwise learning rates

One number applies the same rate to every trainable parameter:

```bash
--learning-rate 0.01
```

With `H` hidden layers, a dense model has `H + 1` conductance tensors followed
by `H` hidden-layer bias tensors. Supply `2H + 1` values to control them
individually. For two hidden layers, for example:

```bash
--learning-rate 0.01 0.01 0.005 0.002 0.001
```

Zero can freeze selected tensors, but at least one supplied rate must be
positive.

### Inspect before launching

`--dry-run` validates the definition and prints the fully expanded JSON without
creating an output directory, loading a dataset, or starting training:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity single \
  --parameter-set configs/train/digits_single_shockley.json \
  --dry-run
```

## Python runner

The same experiment can be defined in a short Python file:

```python
from repro import DRNRunSpec, run_drn


experiment = DRNRunSpec(
    dataset="digits",
    non_linearity="double",
    parameter_set="configs/train/digits_double_shockley.json",
    epochs=10,
    batch_size=32,
    seed=0,
)

result = run_drn(experiment, device="cpu")
print(result.output_dir)
```

This inherits the double-Shockley one-hidden-layer width and therefore uses
four iterations. Set `hidden_sizes=(128, 64)` or
`hidden_sizes=(128, 64, 32)` to define two or three hidden layers; if
`num_iterations` remains omitted, either depth uses eight iterations.

Useful optional fields on `DRNRunSpec` include `nudging`, `input_gain`,
`learning_rate`, `iv_data_path`, `voltage_amp`, `current_amp`, `weight_gain`,
`digits_num_samples`, `mnist_train_samples`, and `mnist_test_samples`.

## Custom measured I-V curves

Select the measured/PWL nonlinearity and pass a custom curve independently of
the physical and solver parameter source:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity pwl \
  --parameter-set configs/train/default_custom_iv.json \
  --iv-data-path data/assets/my_curve.npz \
  --dry-run
```

The command-line curve overrides the referenced simulator profile's
`iv_data_path`. It does not replace that profile's damping, Newton, or
amplification settings, and optimizer settings continue to come from the
training template.

The NPZ must contain either equal-length, one-dimensional `i` and `v` arrays or
an `iv` array shaped exactly `(2, N)` in current-then-voltage order. It must
contain at least two real numeric, finite samples, with strictly increasing
voltage and nondecreasing current. Relative paths resolve from the repository
root, and validation also runs during `--dry-run`. Paths inside the repository
are stored relative to its root; external paths remain absolute. The generated
runner metadata records `iv_data_source` as `user` or `parameter-source`.
Supplying `--iv-data-path` with a non-PWL nonlinearity is rejected.

Use `iv_data_path="data/assets/my_curve.npz"` in `DRNRunSpec` for the Python
interface. See [`ADDING_NONLINEARITY.md`](ADDING_NONLINEARITY.md) for a file
creation example and for the code path required by a new analytic current law.

## Training sources and simulator profiles

The runner requires one explicit training JSON source. Pass its
repository-relative or absolute path through `--parameter-set` (or
`parameter_set` in `DRNRunSpec`). Use the `default_*.json` file matching both
dataset and nonlinearity for a new experiment, one of the three `digits_*.json`
files for the checked paper-Digits settings, or
`mnist_paper_double_shockley.json` for the reported MNIST DRN-XS settings.

The alias `default` is dataset- and nonlinearity-aware. The names
`paper-digits` and `paper-mnist-xs` also remain accepted as bundled aliases.
`--parameter-config` and `DRNRunSpec.parameter_config` remain legacy path
aliases; new commands and code should use an explicit path through
`parameter_set`.

An editable training JSON has the training schema used by the defaults in
`configs/train/` and must declare `simulator_profile`. The referenced profile
must contain the physical and solver fields and its `non_linearity` must match
the requested run. Shockley profiles provide `exponential_diode_param`; a
measured/PWL profile provides `iv_data_path`. The compact run's
`iv_data_path`/`--iv-data-path` can replace that curve. A profile reference is
repository-relative and must resolve inside the repository.

### Edit a default template

Keep the bundled templates unchanged. Copy the one matching both the dataset
and nonlinearity into the git-ignored `configs/local/` directory and edit that
copy. For example:

```bash
mkdir -p configs/local
cp configs/train/default_mnist_single_shockley.json \
  configs/local/my_mnist_single.json
# Edit configs/local/my_mnist_single.json, then run:
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-set configs/local/my_mnist_single.json \
  --device cuda \
  --download
```

This is sufficient for training, model, and data changes. `seed` intentionally
remains here because it governs parameter initialization, data splitting, and
loader order.

To change the physical or coordinate-solver settings too, copy the referenced
profile and update the local training file's `simulator_profile` value:

```bash
cp configs/simulator/default_single_shockley.json \
  configs/local/my_single_simulator.json
```

Set `simulator_profile` in `configs/local/my_mnist_single.json` to
`configs/local/my_single_simulator.json`. For a custom I-V source, start from
`default_mnist_custom_iv.json` on MNIST or `default_custom_iv.json` on Digits,
copy `configs/simulator/default_pwl.json`, and edit `iv_data_path` in the copied
profile. Alternatively, pass `--iv-data-path PATH` for a one-run override.

For example:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --hidden-sizes 256 \
  --learning-rate 0.005 \
  --non-linearity double \
  --parameter-set configs/train/digits_double_shockley.json \
  --epochs 20
```

## Generated outputs

Unless `--output` is supplied, runs are placed under
`outputs/training/runner/`. Each run contains:

- `config.generated.json`: the compact runner definition expanded to the full
  training and simulator settings, including `simulator_profile_source` and
  `simulator_profile_sha256`;
- `config.resolved.json`: runtime overrides and resolved paths;
- `history.json`: epoch-wise losses and accuracies;
- `model.pt` and `model_best.pt`: final and best checkpoints; and
- `run_metadata.json`: runtime, environment, accuracy, and checkpoint hashes.

An explicitly supplied output directory must be absent or empty so an existing
scientific run cannot be overwritten accidentally.
