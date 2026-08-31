# Training configurations

The JSON files in this directory include reusable starting-point templates and
the checked parameter sources used for paper reproduction. They share the
editor-aware [`schema.json`](schema.json). Editors that understand JSON Schema
display a field's definition and allowed values when hovering over it.

For equations, units, numerical tradeoffs, field applicability, and guidance
for changing solver values, read
[`docs/CONFIG_REFERENCE.md`](../../docs/CONFIG_REFERENCE.md).

## Default starting points

The compact runner's `--parameter-set default` selects a template from the
requested nonlinearity:

| CLI nonlinearity | Template | Validated starting point |
|---|---|---|
| `single` | `default_single_shockley.json` | Digits, one hidden layer of width 64 |
| `double` | `default_double_shockley.json` | Digits, one hidden layer of width 32 |
| `pwl` | `default_custom_iv.json` | Digits, one hidden layer of width 32 |

The runner can use each template with either Digits or MNIST. Their values are
validated Digits-derived starting points; using one with MNIST defines a new
experiment and is not a paper-MNIST claim.

Do not edit a bundled template in place. Copy the matching file into the
git-ignored `configs/local/` directory, edit that copy, and pass it through
`--parameter-config`. For example:

```bash
mkdir -p configs/local
cp configs/train/default_single_shockley.json \
  configs/local/my_single_shockley.json
cp configs/train/default_double_shockley.json \
  configs/local/my_double_shockley.json
cp configs/train/default_custom_iv.json \
  configs/local/my_custom_iv.json
```

Copy only the template or templates needed. After editing one, select it
explicitly:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-config configs/local/my_single_shockley.json \
  --device cuda \
  --download
```

The copied JSON's `non_linearity` must match the command. For measured/PWL,
set `iv_data_path` in `configs/local/my_custom_iv.json`, or override that field
for an individual run with `--iv-data-path path/to/curve.npz`.

## Paper parameter sources

| File | Dataset | Architecture / fixed sweeps | Active nonlinear updater |
|---|---|---|---|
| `digits_single_shockley.json` | Digits | one hidden layer / 4 | single-Shockley Lambert-W |
| `digits_double_shockley.json` | Digits | one hidden layer / 4 | double-Shockley float64 Lambert-W with overrelaxation |
| `digits_pwl.json` | Digits | one hidden layer / 4 | measured/PWL damped Newton |
| `mnist_paper_double_shockley.json` | MNIST | one hidden layer / 4 | paper DRN-XS float64 Lambert-W |

These files use a complete common schema. Consequently, they retain some
compatibility fields that are inactive for the chosen nonlinearity. In
particular, Lambert-W settings do not affect measured/PWL runs, and measured
Newton settings do not affect Shockley runs. The applicability table in the
configuration reference is authoritative.

JSON does not support comments. Keep explanatory text in the shared schema and
reference rather than adding ad hoc comment keys to individual experiment
files. Use `python scripts/train_drn.py ... --dry-run` to inspect the fully
expanded configuration before launching a run.

For the compact runner, omitting `--hidden-sizes` inherits the selected
parameter source's architecture. Digits then defaults to four iterations for
one hidden layer and eight for two or three hidden layers. With four or more it
retains the parameter source's iteration count. An explicit
`--num-iterations` always takes precedence.

For a custom measured curve, keep `--non-linearity pwl` and a PWL parameter
source, then add `--iv-data-path path/to/curve.npz`. This overrides only the
source's curve. The NPZ must contain equal-length, one-dimensional `i` and `v`
arrays, or an `iv` array shaped exactly `(2, N)` with current first. At least
two real numeric, finite samples are required; voltage must be strictly
increasing and current must be nondecreasing. Relative paths resolve from the
repository root. See
[`docs/ADDING_NONLINEARITY.md`](../../docs/ADDING_NONLINEARITY.md) for the full
sampled-data contract and analytic-extension guide.
