# Nonlinear DRN paper

This repository contains the code and data needed to train models and
reproduce the results in
**“Fast simulation of nonlinear deep resistive networks for energy-based
computation.”**

It supports the three nonlinearities evaluated in the paper:

- single Shockley diode (`single_diode_exponential`),
- double Shockley diode (`double_diode_exponential`), and
- the piecewise-linear I–V curve (`experimental` in the code).

It serves three purposes:

1. **Reproducing figures and tables:** regenerate every data-driven paper asset from the
   bundled curated inputs; copy the six source schematics into the same output
   tree.
2. **Reproducing numerical results:** rerun selected Digits checkpoints with coordinate
   descent and compare node voltages against bundled SPICE reference states.
3. **Training:** train dense DRNs with EqProp using any of
   the three paper nonlinearities.

## Prerequisites

- Git, which is needed both to clone this repository and to install the pinned
  Lambert-W dependency.
- Python 3.12 or 3.13 with `venv` and `pip`. Python 3.12 is the paper's
  reference environment; Python 3.13 uses a separately pinned compatible
  stack and can have small numerical differences.
- A CPU is sufficient for installation checks, plotting, Digits experiments,
  and short training runs. An NVIDIA GPU is recommended for full MNIST
  training and numerical replay; the repository provides a separately pinned
  CUDA 12.4 environment for that path.

The repository is self-contained at the source and artifact level: it does not
need a parent checkout or a manually configured `PYTHONPATH`. Installing the
Python dependencies requires internet access. MNIST is the only dataset that
is downloaded separately, and only when `--download` is supplied.

## Clone and install

### CPU

```bash
git clone https://github.com/filip137/nonlinear-drn-paper.git
cd nonlinear-drn-paper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/reproduce.py verify
```

`verify` checks both the installed scientific environment and the integrity of
the bundled inputs. No dataset is downloaded and no training is started. If
`pip` prints an `ERROR`, the installation is incomplete; recreate the virtual
environment before continuing.

`requirements.txt` selects matched CPU builds automatically: PyTorch 2.5.1
with Torchvision 0.20.1 on Python 3.12, and PyTorch 2.6.0 with Torchvision
0.21.0 on Python 3.13. Use Python 3.12 when reproducing the paper's reference
numerical results.

### NVIDIA GPU

Create and activate the same repository-local virtual environment, but install
the complete CUDA 12.4 requirements instead of `requirements.txt`:

```bash
source .venv/bin/activate
python -m pip install -r requirements-cuda.txt
python scripts/reproduce.py verify --device cuda
```

Run `python -c "import sys; print(sys.executable)"` if there is any doubt about
the active environment: the path should end in
`nonlinear-drn-paper/.venv/bin/python`. CUDA verification fails explicitly when
the PyTorch build, NVIDIA driver, or device is unavailable; CUDA training never
silently falls back to CPU.

## Run a first simulation

Pass the bundled scikit-learn Digits test split through a trained
double-Shockley DRN:

```bash
python scripts/reproduce.py demo
```

The demo loads the bundled one-hidden-layer, width-64 checkpoint and its
audited solver configuration, evaluates all 360 held-out examples, and prints
the resulting accuracy (95.0% on the supported reference environments). It
does not train, download data, or create output files.

## Train a model

### Digits

Train the compact double-Shockley anchor on CPU:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity double \
  --parameter-set paper-digits \
  --epochs 10 \
  --device cpu
```

Omitting `--hidden-sizes` uses the selected parameter source's audited
one-hidden-layer anchor (width 32 here). Supply one width per hidden layer to
change the architecture, for example `--hidden-sizes 128 64`.

When `--num-iterations` is omitted, Digits uses four coordinate-descent sweeps
for one hidden layer and eight for two or three hidden layers. For four or more
hidden layers, set `--num-iterations` explicitly; otherwise the selected
parameter source's value is retained. An explicit value always wins.

Add `--max-batches 1 --max-eval-batches 1 --epochs 1` for a quick end-to-end
training check.

While training, an interactive terminal refreshes one progress line after every
training and evaluation batch with the running loss, running accuracy, and
elapsed time. Redirected output reports the same metrics at roughly 10%
intervals so log files stay compact. The complete summary is still printed at
the end of every epoch.

### MNIST

Launch the paper's double-Shockley DRN-XS configuration on CUDA:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity double \
  --parameter-set paper-mnist-xs \
  --device cuda \
  --download \
  --seed 0
```

This preset supplies the one-hidden-layer width of 100, 100 epochs, four
coordinate-descent sweeps, paper learning rates, input gain, and electrical
parameters. `--download` is required unless MNIST already exists under
`data/external/mnist/`. See
[docs/MNIST_REPRODUCTION.md](docs/MNIST_REPRODUCTION.md) for the exact protocol,
published-aggregate provenance, and seed limitation.

For a single-Shockley MNIST experiment, start from the bundled default instead:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-set default \
  --device cuda \
  --download \
  --seed 0
```

`default` is nonlinearity-aware: it selects
`configs/train/default_single_shockley.json` for `single`,
`default_double_shockley.json` for `double`, and `default_custom_iv.json` for
`pwl`. These parameter sources can be used with either Digits or MNIST, but
their values are validated Digits-derived starting points. The command above
therefore defines a new MNIST experiment, not a paper-MNIST configuration or
result. The paper-MNIST parameter set supports only `double`, which is why
combining `--parameter-set paper-mnist-xs` with `--non-linearity single` is
rejected.

Choose `single`, `double`, or `pwl` for the three paper nonlinearities. The
selected parameter set supplies
known-working learning-rate and input-gain defaults; `--learning-rate` and
`--input-gain` override them. The parameter set is required because it also
supplies explicit physical diode parameters, which are never silently
invented. Adaptive equilibrium is disabled during training. Every run saves
the fully expanded configuration beside its checkpoints.

To change a default's physical or solver settings, copy the matching template
to the git-ignored `configs/local/` directory, edit the copy, and select it with
`--parameter-config`:

```bash
mkdir -p configs/local
cp configs/train/default_single_shockley.json \
  configs/local/mnist_single.json
# Edit configs/local/mnist_single.json, then run:
python scripts/train_drn.py \
  --dataset mnist \
  --non-linearity single \
  --parameter-config configs/local/mnist_single.json \
  --device cuda \
  --download \
  --seed 0
```

Use either `--parameter-set` or `--parameter-config`, never both. For a custom
measured/PWL template, set `iv_data_path` in the copied JSON or override it at
the command line with `--iv-data-path`.

The same interface is available as the Python function `repro.run_drn`.
See [docs/TRAINING_RUNNER.md](docs/TRAINING_RUNNER.md) for Digits, MNIST,
layerwise-learning-rate, dry-run, and custom-parameter examples. To inspect the
fully expanded configuration without training or writing files, add
`--dry-run`.

Every checked training JSON points to an editor-aware schema. See
[configs/train/README.md](configs/train/README.md) for the available parameter
sources and [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) for the full
field reference, including the Lambert-W large-argument threshold,
exponent-clipping guard, Newton-polish settings, and which values are inactive
for each nonlinearity.

Run a checked JSON configuration directly, or smoke-test all three paper
nonlinearities, with:

```bash
python scripts/reproduce.py train \
  --config configs/train/digits_double_shockley.json \
  --device cpu \
  --epochs 15
python scripts/reproduce.py train-smoke --device cpu
```

Training runs write the expanded configuration, history, final and best
checkpoints, and runtime metadata under `outputs/training/` unless `--output`
selects another directory.

## Use a custom I–V curve

The measured/PWL model accepts a sampled passive current–voltage curve in an
NPZ file. The recommended layout uses one-dimensional `i` and `v` arrays:

```python
import numpy as np

v = np.linspace(-1.5, 1.5, 301)
i = 1e-3 * v + 2e-3 * v**3
np.savez("my_curve.npz", i=i, v=v)
```

Both arrays must contain at least two finite values; voltage must be strictly
increasing and current must be nondecreasing. An `iv` array shaped `(2, N)` is
also accepted, with current in the first row and voltage in the second.

Use the curve in the compact runner with:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity pwl \
  --parameter-set paper-digits \
  --iv-data-path my_curve.npz \
  --epochs 10 \
  --device cpu
```

Here `paper-digits` still supplies the electrical and PWL solver settings; only
the sampled curve is replaced. Relative curve paths are resolved from the
repository root. Curves should span the intended operating-voltage range
because the updater clamps outside the sampled range by default.

For an analytic device law or another coordinate-update rule, see
[docs/ADDING_NONLINEARITY.md](docs/ADDING_NONLINEARITY.md). It describes the
energy interaction, coordinate solver, registration, configuration, and tests
needed to add a new nonlinearity.

## Reproduce the paper figures

Regenerate every data-driven manuscript asset from the bundled inputs:

```bash
python scripts/reproduce.py figures
```

Generated assets are written under `outputs/paper/` with the same relative
paths and filenames used by the manuscript. The checked-in originals under
`paper/reference/` are comparison targets, not inputs to the generated plots.
To regenerate only the two MNIST panels, run
`python scripts/reproduce.py mnist-figures`; the audited gain, learning-rate,
normalization, and solver provenance is in
[docs/MNIST_REPRODUCTION.md](docs/MNIST_REPRODUCTION.md).

## Numerical replay

List the 195 bundled validation jobs and run a one-job CPU check:

```bash
python scripts/reproduce.py list
python scripts/reproduce.py validate --group timing --device cpu --limit 1
python scripts/reproduce.py compare --group timing --limit 1
```

Full replay is much more expensive and is normally run on CUDA:

```bash
python scripts/reproduce.py all --device cuda
```

Wall-clock plots always use the bundled measurements because timings are
machine-dependent. SPICE netlist generation is not part of this artifact;
the required SPICE state arrays and timing measurements are bundled.

## Acknowledgments and provenance

This repository extends the coordinate-descent formulation and DRN simulation
framework introduced by Benjamin Scellier in
[“A fast algorithm to simulate nonlinear resistive networks”](https://proceedings.mlr.press/v235/scellier24a.html),
ICML 2024, PMLR 235:43477–43503. The original work establishes the fast exact
coordinate-descent method for ideal-diode networks; this artifact extends that
framework to the paper's non-ideal Shockley and measured/PWL nonlinearities.

The original MIT copyright for Benjamin Scellier, Maxence Ernoult, and Rain
Neuromorphics Inc. is retained in [LICENSE](LICENSE). See [NOTICE](NOTICE) for
the code and scientific provenance statement.

## Repository map

- `configs/train/`: ready-to-run training configurations.
- `data/`: selected weights, reference NPZ files, figure inputs, and checksums.
- `paper/reference/`: the exact manuscript assets used as visual references.
- `repro/vendor/model/resistive/minimizer.py`: canonical coordinate updaters
  and minimizer selection, in the original model layout.
- `repro/`: training, validation, measured-data loading, comparison,
  aggregation, and plotting code.
- `scripts/reproduce.py`: single command-line entry point.
- `docs/`: data provenance and reproduction notes.
- `tests/`: fast integrity, configuration, plotting, and training checks.

See `paper/figure_manifest.json` for the one-to-one paper asset map and
`docs/DATA_PROVENANCE.md` for the selection policy and limitations.
