# Nonlinear DRN paper reproducibility artifact

This standalone repository contains the smallest practical set of code and
data needed to inspect, train, and reproduce the computational results in
**“Fast simulation of nonlinear deep resistive networks for energy-based
computation.”** It is deliberately separate from the larger research
workspace so it can be linked directly from the paper.

The artifact supports the three nonlinearities evaluated in the paper:

- single Shockley diode (`single_diode_exponential`),
- double Shockley diode (`double_diode_exponential`), and
- the measured piecewise-linear I–V curve (`experimental` in the code).

It provides three levels of reproduction:

1. **Figures and tables:** regenerate every data-driven paper asset from the
   bundled curated inputs; copy the six source schematics into the same output
   tree.
2. **Numerical replay:** rerun selected Digits checkpoints with coordinate
   descent and compare node voltages against bundled SPICE reference states.
3. **Training:** train dense DRNs with equilibrium propagation using any of
   the three paper nonlinearities. A no-download Digits path is included for
   smoke tests, together with the exact MNIST double-Shockley configuration
   used for the paper experiment.

## Quick start

Python 3.12 is the reference environment. A CPU-only installation is enough
for figures and smoke tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/reproduce.py verify
python scripts/reproduce.py figures
```

Generated assets are written under `outputs/paper/` with the same relative
paths and filenames used by the manuscript. The checked-in originals under
`paper/reference/` are comparison targets, not inputs to the generated plots.
To regenerate only the two MNIST panels, run
`python scripts/reproduce.py mnist-figures`; the audited gain, learning-rate,
normalization, and solver provenance is in
[docs/MNIST_REPRODUCTION.md](docs/MNIST_REPRODUCTION.md).

## Simple experiment runner

For exploratory training, `scripts/train_drn.py` follows the compact style of
the original Scellier fast-DRN examples: specify the dataset, hidden-layer
sizes, and nonlinearity together, with optional overrides for the learning rate
and other training parameters.

```bash
python scripts/train_drn.py \
  --dataset digits \
  --hidden-sizes 128 64 \
  --non-linearity double \
  --parameter-set paper-digits \
  --epochs 10 \
  --device cpu
```

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
`run_metadata.json` to `outputs/training/` (or to `--output`). Diode parameter
dictionaries are mandatory in every configuration; no physical parameters are
silently invented.

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
machine-dependent. SPICE netlist generation is not part of this minimal
artifact; the required SPICE state arrays and timing measurements are bundled.

## Repository map

- `configs/train/`: ready-to-run training configurations.
- `data/`: selected weights, reference NPZ files, figure inputs, and checksums.
- `paper/reference/`: the exact manuscript assets used as visual references.
- `repro/`: training, validation, comparison, aggregation, and plotting code.
- `scripts/reproduce.py`: single command-line entry point.
- `docs/`: data provenance and reproduction notes.
- `tests/`: fast integrity, configuration, plotting, and training checks.

See `paper/figure_manifest.json` for the one-to-one paper asset map and
`docs/DATA_PROVENANCE.md` for the selection policy and limitations.
