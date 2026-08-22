# Nonlinear DRN paper reproducibility artifact

This  repository contains code and
data needed train and reproduce the results in
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
2. **Reproduce numerical results:** rerun selected Digits checkpoints with coordinate
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

```bash
git clone https://github.com/filip137/nonlinear-drn-paper.git
cd nonlinear-drn-paper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU environment
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

For an NVIDIA GPU, install the complete CUDA 12.4 environment instead of the
CPU requirements, then require CUDA during verification:

```bash
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements-cuda.txt
python scripts/reproduce.py verify --device cuda
```

The printed interpreter must end in
`nonlinear-drn-paper/.venv/bin/python`. Do not launch from the Conda `base`
interpreter: an unrelated PyTorch build there may require a newer CUDA driver
than the machine provides. The CUDA verification deliberately fails when the
GPU stack is unavailable; training requested with `--device cuda` never
silently falls back to CPU.

## Run a first simulation

First inspect a fully expanded one-hidden-layer Digits experiment without
creating output or starting training:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity double \
  --parameter-set paper-digits \
  --dry-run
```

Then run one training and evaluation batch on CPU:

```bash
python scripts/train_drn.py \
  --dataset digits \
  --non-linearity double \
  --parameter-set paper-digits \
  --epochs 1 \
  --max-batches 1 \
  --max-eval-batches 1 \
  --device cpu
```

## Define a training run

For exploratory training, `scripts/train_drn.py` specify the dataset, hidden-layer
sizes, and nonlinearity.

```bash
python scripts/train_drn.py \
  --dataset digits \
  --hidden-sizes 128 64 \
  --non-linearity double \
  --parameter-set paper-digits \
  --epochs 10 \
  --device cpu
```

This explicit architecture has two hidden layers, so an omitted
`--num-iterations` resolves to eight. Digits uses four iterations for one
hidden layer and eight for two; explicit `--hidden-sizes` and
`--num-iterations` values always win.

Use `--dataset mnist --download` for MNIST, and choose `single`, `double`, or
`pwl` for the three paper nonlinearities. The selected parameter set supplies
known-working learning-rate and input-gain defaults; `--learning-rate` and
`--input-gain` override them. The parameter set is required because it also
supplies explicit physical diode parameters, which are never silently
invented. Adaptive equilibrium is disabled during training. Every run saves
the fully expanded configuration beside its checkpoints.

The same interface is available as the Python function `repro.run_drn`.
See [docs/TRAINING_RUNNER.md](docs/TRAINING_RUNNER.md) for Digits, MNIST,
layerwise-learning-rate, dry-run, and custom-parameter examples.

Every checked training JSON points to an editor-aware schema. See
[configs/train/README.md](configs/train/README.md) for the available parameter
sources and [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) for the full
field reference, including the Lambert-W large-argument threshold,
exponent-clipping guard, Newton-polish settings, and which values are inactive
for each nonlinearity.

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

## Training

Run a short CPU smoke test for all three nonlinearities:

```bash
python scripts/reproduce.py train-smoke --device cpu
```

Train one configuration (command-line values override the JSON):

```bash
python scripts/reproduce.py train \
  --config configs/train/digits_double_shockley.json \
  --device cpu \
  --epochs 15
```

All three bundled Digits training configurations use one hidden layer and
four fixed coordinate-descent iterations.

The paper MNIST setup is in
`configs/train/mnist_paper_double_shockley.json`. MNIST itself is not
redistributed; allow torchvision to download it explicitly:

```bash
python scripts/reproduce.py train \
  --config configs/train/mnist_paper_double_shockley.json \
  --device cuda \
  --download
```

Each run writes `config.resolved.json`, `history.json`, `model.pt`, and
`run_metadata.json` to `outputs/training/` (or to `--output`).

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
