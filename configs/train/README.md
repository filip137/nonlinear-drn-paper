# Training configurations

The JSON files in this directory include reusable starting-point templates and
the checked parameter sources used for paper reproduction. They share the
editor-aware [`schema.json`](schema.json). Editors that understand JSON Schema
display a field's definition and allowed values when hovering over it.

For equations, units, numerical tradeoffs, field applicability, and guidance
for changing solver values, read
[`docs/CONFIG_REFERENCE.md`](../../docs/CONFIG_REFERENCE.md).

## Training templates and simulator profiles

The six editable defaults deliberately separate two kinds of choices:

- `configs/train/default*.json` contains dataset preprocessing, architecture,
  weight initialization and bounds, optimizer settings, fixed-sweep training
  controls, and `seed`.
- Its `simulator_profile` field points to a file under `configs/simulator/`
  containing the nonlinearity, electrical parameters, amplifier settings, and
  coordinate-updater controls.

The seed stays in the training template because it controls parameter
initialization, the data split, and loader order rather than the deterministic
device law. Loading a template merges its simulator profile before validation.
The generated and resolved configurations saved with a run are fully expanded
and self-contained, and record both `simulator_profile_source` and
`simulator_profile_sha256`.

## Dataset-specific defaults

Pass the template matching both the dataset and nonlinearity directly to
`--parameter-set`:

| Dataset | CLI nonlinearity | Training template | Hidden width | Referenced simulator profile |
|---|---|---|---:|---|
| Digits | `single` | `default_single_shockley.json` | 32 | `configs/simulator/default_single_shockley.json` |
| Digits | `double` | `default_double_shockley.json` | 32 | `configs/simulator/default_double_shockley.json` |
| Digits | `pwl` | `default_custom_iv.json` | 32 | `configs/simulator/default_pwl.json` |
| MNIST | `single` | `default_mnist_single_shockley.json` | 100 | `configs/simulator/default_single_shockley.json` |
| MNIST | `double` | `default_mnist_double_shockley.json` | 100 | `configs/simulator/default_mnist_double_shockley.json` |
| MNIST | `pwl` | `default_mnist_custom_iv.json` | 100 | `configs/simulator/default_pwl.json` |

All six defaults use batch size 10; `--batch-size` overrides it for one run.
Only the double-Shockley MNIST settings have an audited paper basis. The MNIST
single-Shockley and PWL templates adapt validated Digits settings and are
starting points for new experiments, not paper-MNIST results. Use
`mnist_paper_double_shockley.json`, not the editable MNIST double default, for
the exact reported DRN-XS protocol.

The alias `--parameter-set default` resolves by both dataset and nonlinearity,
but explicit JSON paths are preferred because they make the selected training
source visible in the command. Relative paths resolve from the repository
root. The `paper-digits` and `paper-mnist-xs` aliases and the
`--parameter-config` path flag remain available for compatibility.

## Make local changes

Keep the bundled defaults unchanged. Copy the matching training template into
the git-ignored `configs/local/` directory, edit the copy, and pass its path to
`--parameter-set`:

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

Changing architecture, learning rate, batch size, dataset preprocessing, or
seed requires only the copied training file. To change physical or solver
settings, also copy its referenced simulator profile:

```bash
cp configs/simulator/default_single_shockley.json \
  configs/local/my_single_simulator.json
```

Then set `simulator_profile` in `configs/local/my_mnist_single.json` to
`configs/local/my_single_simulator.json`. Profile references must be
repository-relative paths that stay inside the repository.

For a reusable measured/PWL setup, copy `default_mnist_custom_iv.json` (or the
Digits `default_custom_iv.json`) and `configs/simulator/default_pwl.json`, set
the copied training file's `simulator_profile` to the copied profile, and edit
`iv_data_path` in that profile. For a one-run curve override, leave the profile
unchanged and add `--iv-data-path path/to/curve.npz`.

## Checked paper configurations

| File | Dataset | Architecture / fixed sweeps | Active nonlinear updater |
|---|---|---|---|
| `digits_single_shockley.json` | Digits | width 64 / 4 | single-Shockley Lambert-W |
| `digits_double_shockley.json` | Digits | width 32 / 4 | double-Shockley float64 Lambert-W with overrelaxation |
| `digits_pwl.json` | Digits | width 32 / 4 | measured/PWL damped Newton |
| `mnist_paper_double_shockley.json` | MNIST | width 100 / 4 | paper DRN-XS float64 Lambert-W |

These are reproduction records rather than editable defaults. Preserve their
values when reproducing the paper.

JSON does not support comments. Keep explanatory text in the schemas and
reference rather than adding ad hoc keys. Use
`python scripts/train_drn.py ... --dry-run` to inspect the fully expanded
configuration before launching a run.

For the compact runner, omitting `--hidden-sizes` inherits the selected
training source's architecture. Digits then defaults to four iterations for one
hidden layer and eight for two or three hidden layers. With four or more it
retains the training source's iteration count. An explicit `--num-iterations`
always takes precedence.

A measured curve NPZ must contain equal-length, one-dimensional `i` and `v`
arrays, or an `iv` array shaped exactly `(2, N)` with current first. At least
two real numeric, finite samples are required; voltage must be strictly
increasing and current must be nondecreasing. See
[`docs/ADDING_NONLINEARITY.md`](../../docs/ADDING_NONLINEARITY.md) for the full
sampled-data contract and analytic-extension guide.
