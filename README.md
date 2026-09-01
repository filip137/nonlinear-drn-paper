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
   bundled curated inputs.
2. **Reproducing numerical results:** rerun selected networks with the coordinate
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

## Simulate a hand-specified small network

Use the small-network helper when you want node voltages for literal physical
conductances rather than a trained checkpoint. The input width is the number
of voltage-source nodes: the helper applies those voltages directly and does
not create positive and negative input channels.

Run the editable example:

```bash
python scripts/small_network.py
```

Or define a network in Python:

```python
import numpy as np

from repro import simulate_small_network

result = simulate_small_network(
    layer_sizes=[2, 4, 2],
    conductances=[
        np.array([
            [1.0, 0.2, 0.8, 0.4],
            [0.3, 1.1, 0.5, 0.9],
        ]),
        np.array([
            [0.8, 0.2],
            [0.4, 1.0],
            [1.1, 0.3],
            [0.2, 0.7],
        ]),
    ],
    input_voltages=np.array([
        [0.2, -0.1],
        [0.7, 0.3],
    ]),
    non_linearity="double",
    shockley_parameters={"I_s": 1e-6, "V_t": 0.05, "V_off": 0.8},
    adaptive_equilibrium=True,
)

print(result.hidden_voltages[0])
print(result.output_voltages)
print(result.converged, result.sweeps)
```

Each conductance matrix has shape `(previous_layer, next_layer)`, all entries
must be finite and non-negative, and each row of `input_voltages` is simulated
independently. The hidden layers use the selected grounded nonlinearity while
the output layer is linear. The accepted selectors are `single`, `double`, and
`pwl`. Single Shockley requires even hidden widths: the first half of each
hidden layer uses forward-oriented diodes and the second half reverse-oriented
diodes.

The default Shockley dictionary is `{"I_s": 1e-6, "V_t": 0.05, "V_off":
0.8}`. PWL uses
`data/assets/experimental_curve_voff_0.8_200_points.npz` unless
`iv_data_path` is replaced. Adaptive settling stops once the voltage-change
criterion is met, up to `max_sweeps=128`; set `adaptive_equilibrium=False` to
run exactly 128 sweeps. Both modes report convergence in the returned result
and warn if the final voltages do not meet the configured tolerance.

## Train a model

### Digits

Train the compact double-Shockley anchor on CPU:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity double \
  --parameter-set configs/train/digits_double_shockley.json \
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
  --parameter-set configs/train/mnist_paper_double_shockley.json \
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
  --parameter-set configs/train/default_mnist_single_shockley.json \
  --device cuda \
  --download \
  --seed 0
```

This MNIST template adapts the validated Digits single-Shockley settings, so
the command defines a new experiment rather than a paper-MNIST result. Only
the double-Shockley MNIST configuration has audited paper parameters. The six
dataset-specific defaults all use batch size 10; see
[configs/train/README.md](configs/train/README.md) for the default table and
the training-template/simulator-profile copy-and-edit workflow.

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
  --dataset mnist \
  --non-linearity pwl \
  --parameter-set configs/train/default_mnist_custom_iv.json \
  --iv-data-path my_curve.npz \
  --epochs 10 \
  --device cuda \
  --download
```

For Digits, use `configs/train/default_custom_iv.json` instead. The training
template references `configs/simulator/default_pwl.json`; `--iv-data-path`
overrides that profile's `iv_data_path` for this run. Relative curve paths
resolve from the repository root, and curves should span the intended operating
range. The MNIST PWL template is a Digits-derived starter, not a paper-MNIST
result; see [configs/train/README.md](configs/train/README.md) for reusable
profile edits.

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
