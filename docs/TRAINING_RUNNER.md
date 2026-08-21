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
  --parameter-set paper-digits \
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
  --parameter-set paper-digits \
  --epochs 10 \
  --device cpu
```

With two hidden layers and no `--num-iterations` override, the runner uses
eight iterations.

For a quick end-to-end check, add:

```bash
--epochs 1 --max-batches 1 --max-eval-batches 1
```

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
  --parameter-set paper-mnist-xs \
  --epochs 100 \
  --num-iterations 4 \
  --device cuda \
  --download
```

`--download` is explicit: without it, the runner requires MNIST to exist under
`data/external/mnist/` (or the directory supplied through `--mnist-root`).

Single-Shockley and measured/PWL models can also be trained on MNIST. Their
paper measurements were made in the Digits experiments, so select those
electrical and updater settings explicitly:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --hidden-sizes 256 128 \
  --learning-rate 0.01 \
  --non-linearity pwl \
  --parameter-set paper-digits \
  --epochs 20 \
  --device cuda \
  --download
```

This is a new MNIST experiment, not a claimed paper configuration. The output
configuration records that its physical parameter source is `paper-digits`.

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
selected parameter source when their flags are omitted:

| Parameter set | Nonlinearity | Default learning rate | Default input gain | Anchor |
|---|---|---:|---:|---|
| `paper-digits` | single Shockley | 0.005 for weights; 0 for bias | 10 | Digits, one hidden layer of width 64 |
| `paper-digits` | double Shockley | 0.01 | 10 | Digits, one hidden layer of width 32 |
| `paper-digits` | measured/PWL | 0.01 | 10 | Digits, one hidden layer of width 32 |
| `paper-mnist-xs` | double Shockley | 0.15 / 0.08 / 0.05 (weight / weight / bias) | 50 | MNIST accuracy-panel DRN-XS, one hidden layer of width 100 |

Architecture is optional for Digits as well. When `--hidden-sizes` is omitted,
the selected source supplies its one-hidden-layer anchor: width 64 for single
Shockley and width 32 for double Shockley or measured/PWL. Omitted Digits
solver iterations depend on the resulting depth: four for one hidden layer and
eight for two hidden layers. An explicit `--hidden-sizes` or
`--num-iterations` value takes precedence. The bundled Digits JSON
configurations all use one hidden layer and four fixed iterations.
For three or more hidden layers, the runner retains the selected source's
iteration count; set `--num-iterations` explicitly when exploring those
depths.

These are known-working anchors rather than a claim that one rate is optimal
for every depth and width. On the deterministic Digits split (`seed=0`), the
15-epoch, four-sweep anchors reached 93.6% test accuracy for single Shockley
and 93.1% for both double Shockley and measured/PWL on the Python 3.12 CPU
reference stack. These checks validate the runner and its defaults; they are
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
  --parameter-set paper-digits \
  --dry-run
```

## Python runner

The same experiment can be defined in a short Python file:

```python
from repro import DRNRunSpec, run_drn


experiment = DRNRunSpec(
    dataset="digits",
    non_linearity="double",
    parameter_set="paper-digits",
    epochs=10,
    batch_size=32,
    seed=0,
)

result = run_drn(experiment, device="cpu")
print(result.output_dir)
```

This inherits the double-Shockley one-hidden-layer width and therefore uses
four iterations. Set `hidden_sizes=(128, 64)` to define two hidden layers; if
`num_iterations` remains omitted, that depth uses eight iterations.

Useful optional fields on `DRNRunSpec` include `nudging`, `input_gain`,
`learning_rate`, `voltage_amp`, `current_amp`, `weight_gain`,
`digits_num_samples`, `mnist_train_samples`, and `mnist_test_samples`.

## Physical parameter sources

The runner requires exactly one explicit source:

- `parameter_set="paper-digits"` supports all three nonlinearities and uses
  the physical and updater settings of the bundled Digits configurations.
- `parameter_set="paper-mnist-xs"` preserves the reported MNIST DRN-XS
  double-Shockley accuracy-run settings.
- `parameter_config="path/to/config.json"` loads a custom source.

A custom parameter JSON must have the same complete schema as the examples in
`configs/train/`, including non-empty `quadratic_diode_param`,
`exponential_diode_param`, and `hard_sigmoid_param` dictionaries. Its
`non_linearity` must match the requested run. A measured/PWL source must also
provide `iv_data_path`. Relative paths are resolved from the repository root.

For example:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --hidden-sizes 256 \
  --learning-rate 0.005 \
  --non-linearity double \
  --parameter-config configs/train/digits_double_shockley.json \
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
