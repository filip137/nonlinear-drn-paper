# Training runner

`scripts/train_drn.py` and `repro.runner.run_drn` provide a compact interface
for defining dense nonlinear-DRN experiments. The design follows the original
Scellier fast-DRN examples: the important experiment choices are visible
together, and one call builds and launches the complete run.

The runner does not replace the checked JSON configurations used for exact
paper reproduction. It expands the compact definition into
`config.generated.json`, and the training code additionally writes
`config.resolved.json`. Consequently, every result retains all physical,
solver, dataset, architecture, and runtime settings.

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
  --parameter-set configs/train/default_single_shockley.json \
  --device cuda \
  --download \
  --seed 0
```

This is a new MNIST experiment, not a claimed paper configuration or result.
The audited `configs/train/mnist_paper_double_shockley.json` source cannot be
substituted here because it supports only the double-Shockley nonlinearity.

### Nonlinearity names

The short names are:

| CLI name | Stored canonical name | Output encoding |
|---|---|---|
| `single` or `single-shockley` | `single_diode_exponential` | 10 linear outputs |
| `double` or `double-shockley` | `double_diode_exponential` | 20 paired outputs |
| `pwl` or `measured` | `experimental` | 20 paired outputs |

Canonical names are accepted directly. Single-Shockley hidden widths must be
even because each layer is divided into forward- and reverse-oriented nodes.

## Known-working defaults

Learning rate and input gain are optional. They come from the explicitly
selected JSON source when their flags are omitted. The reusable starting
templates are:

| Nonlinearity | Source | Hidden width |
|---|---|---:|
| single Shockley | `configs/train/default_single_shockley.json` | 100 |
| double Shockley | `configs/train/default_double_shockley.json` | 32 |
| measured/PWL | `configs/train/default_custom_iv.json` | 100 |

All three can be used with either Digits or MNIST. Their settings are
Digits-derived starting points, not newly reported paper-MNIST settings. The
learning-rate, input-gain, and architecture anchors are:

| Parameter source | Nonlinearity | Default learning rate | Default input gain | Anchor |
|---|---|---:|---:|---|
| `default_single_shockley.json` | single Shockley | 0.005 for weights; 0 for bias | 10 | Reusable Digits-derived start, width 100 |
| `default_double_shockley.json` | double Shockley | 0.01 | 10 | Reusable Digits-derived start, width 32 |
| `default_custom_iv.json` | measured/PWL | 0.01 | 10 | Reusable Digits-derived start, width 100 |
| `digits_single_shockley.json` | single Shockley | 0.005 for weights; 0 for bias | 10 | Checked paper-Digits source, width 64 |
| `digits_double_shockley.json` | double Shockley | 0.01 | 10 | Checked paper-Digits source, width 32 |
| `digits_pwl.json` | measured/PWL | 0.01 | 10 | Checked paper-Digits source, width 32 |
| `mnist_paper_double_shockley.json` | double Shockley | 0.15 / 0.08 / 0.05 (weight / weight / bias) | 50 | MNIST accuracy-panel DRN-XS, width 100 |

Architecture is optional for Digits as well. When `--hidden-sizes` is omitted,
the selected source supplies its architecture. The default single-Shockley and
PWL templates use width 100, while the default double-Shockley template uses
width 32. The checked paper-Digits sources retain widths 64, 32, and 32,
respectively. Omitted Digits solver iterations depend on the resulting depth:
four for one hidden layer, eight for two or three hidden layers, and the
parameter source's value for four or more. An explicit `--num-iterations`
always takes precedence.

The default templates are starting points rather than a claim that one rate is
optimal for every depth and width. On the deterministic Digits split
(`seed=0`), the checked width-64/32/32 paper-Digits sources reached 93.6% test
accuracy for single Shockley and 93.1% for both double Shockley and measured/PWL
after 15 epochs and four sweeps on the Python 3.12 CPU reference stack. These checks are
not additional paper results.

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

The custom file overrides the parameter source's `iv_data_path`; it does not
replace that source's damping, Newton, amplification, or optimizer settings.
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

## Physical parameter sources

The runner requires one explicit JSON source. Pass its repository-relative or
absolute path through `--parameter-set` (or `parameter_set` in
`DRNRunSpec`). Use a matching `default_*.json` template for a new experiment,
one of the three `digits_*.json` files for the checked paper-Digits settings,
or `mnist_paper_double_shockley.json` for the reported MNIST DRN-XS settings.

The names `default`, `paper-digits`, and `paper-mnist-xs` remain accepted as
legacy bundled aliases. `--parameter-config` and `DRNRunSpec.parameter_config`
remain legacy path aliases; new commands and code should use `parameter_set`.

A custom parameter JSON must have the same complete schema as the examples in
`configs/train/`, including non-empty `quadratic_diode_param`,
`exponential_diode_param`, and `hard_sigmoid_param` dictionaries. Its
`non_linearity` must match the requested run. A measured/PWL source must also
provide `iv_data_path`; the compact run's `iv_data_path`/`--iv-data-path` can
then replace that curve. Relative paths are resolved from the repository root.

### Edit a default template

Keep the bundled templates unchanged. Copy the one matching the requested
nonlinearity into the git-ignored `configs/local/` directory and edit that
copy. For example:

```bash
mkdir -p configs/local
cp configs/train/default_single_shockley.json \
  configs/local/my_single_shockley.json
# Edit configs/local/my_single_shockley.json, then run:
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-set configs/local/my_single_shockley.json \
  --device cuda \
  --download
```

Equivalent starting copies for the other nonlinearities are
`default_double_shockley.json` and `default_custom_iv.json`. A copied source's
`non_linearity` must continue to match the requested run. In the custom-I-V
copy, change the JSON field `iv_data_path`, or pass `--iv-data-path PATH` to
override it for one run.

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
  training schema;
- `config.resolved.json`: runtime overrides and resolved paths;
- `history.json`: epoch-wise losses and accuracies;
- `model.pt` and `model_best.pt`: final and best checkpoints; and
- `run_metadata.json`: runtime, environment, accuracy, and checkpoint hashes.

An explicitly supplied output directory must be absent or empty so an existing
scientific run cannot be overwritten accidentally.
