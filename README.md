# Nonlinear DRN paper

This repository contains the code and data needed to train models and
reproduce the results in **“Fast simulation of nonlinear deep resistive
networks for energy-based computation.”**

It supports the three device families evaluated in the paper:

- single Shockley diode (`single_diode_exponential`);
- antiparallel double Shockley diode (`double_diode_exponential`); and
- measured piecewise-linear I–V data (`experimental`).

The artifact can regenerate every data-driven paper asset from bundled curated
inputs, replay selected Digits networks with coordinate descent and compare
their node voltages against bundled SPICE reference states, train dense DRNs
with equilibrium propagation, and settle a hand-specified physical network.

## Prerequisites

- Git, including for installation of the pinned Lambert-W dependency.
- Python 3.12 or 3.13 with `venv` and `pip`. Python 3.12 is the paper reference
  environment; Python 3.13 has a separately pinned compatible stack and can
  exhibit small numerical differences.
- A CPU for verification, plotting, Digits replay, and short training runs.
  An NVIDIA GPU is recommended for full MNIST training.

The source and bundled artifacts are self-contained; no parent checkout or
manual `PYTHONPATH` is needed. Dependency installation requires internet
access. MNIST is downloaded only when `--download` is explicitly supplied.

## Clone and install

Use Python 3.12 to reproduce the paper's reference numerical results. Python
3.13 is also supported but may produce small numerical differences.

First, clone the repository and create a local virtual environment:

```bash
git clone https://github.com/filip137/nonlinear-drn-paper.git
cd nonlinear-drn-paper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install **one** of the following dependency sets.

### CPU

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py verify
```

### NVIDIA GPU (CUDA 12.4)

```bash
python -m pip install -r requirements-cuda.txt
python scripts/reproduce.py verify --device cuda
```

`verify` checks the installed scientific packages and the integrity of the
bundled inputs. It does not download a dataset or start training. With
`--device cuda`, it also checks that PyTorch can initialize CUDA. CUDA runs
fail explicitly instead of silently falling back to CPU.

The requirement files select matched versions automatically: PyTorch
2.5.1/Torchvision 0.20.1 for Python 3.12, or PyTorch 2.6.0/Torchvision 0.21.0
for Python 3.13.

## Quick demo: simulate a small network

Use `simulate_small_network` to simulate a toy network defined directly in
JSON. The user provides the conductance matrices, input voltages, and diode
parameters; no training or model checkpoint is involved. Start from the bundled
editable example and run:

```bash
python scripts/small_network.py \
  --config configs/small_network/example.json
```

The command prints the settled hidden and output voltages together with the
convergence information.

## Train a model

### Digits on CPU

Train the bundled double-Shockley Digits configuration on its referenced CPU
execution profile:

```bash
python scripts/train_drn.py \
  --config configs/train/digits_double_shockley.json
```

For a quick, auditable end-to-end check, first generate a complete snapshot
with explicit overrides, then run that snapshot:

```bash
mkdir -p configs/local
python scripts/train_drn.py \
  --config configs/train/digits_double_shockley.json \
  --override '/training/epochs=1' \
  --override '/equilibrium/sweeps=2' \
  --override '/training/batch_limits/train=1' \
  --override '/training/batch_limits/evaluation=1' \
  --write-config configs/local/digits_double_smoke.json

python scripts/train_drn.py \
  --config configs/local/digits_double_smoke.json
```

Each override is `JSON_POINTER=JSON_VALUE`; the value is parsed as JSON, so
strings need JSON quotes. Targets must already exist, and duplicate,
overlapping, unknown, or schema-invalid changes are rejected. The generated
snapshot records every change under `provenance.generation_overrides`.

### MNIST and CUDA

The exact reported DRN-XS training source is
`configs/train/mnist_paper_double_shockley.json`. The checked source references
the portable CPU profile. For a CUDA run, copy it under `configs/local/` and
replace `execution_ref` with the path and digest of
`configs/execution/reference_cuda.json` (verify it with `sha256sum`). The
[training guide](docs/TRAINING_RUNNER.md#deterministic-cpu-and-cuda-profiles)
gives the complete source-edit and resolve workflow. Then run:

```bash
python scripts/train_drn.py \
  --config configs/local/mnist_paper_double_cuda.json \
  --download
```

There is no scientific `--device`, `--epochs`, `--seed`, or learning-rate flag.
Those values belong in the source or an explicitly generated snapshot. The
`--download` switch is operational permission to fetch MNIST into the configured
data path; it does not alter the scientific config.

Only double Shockley has audited MNIST paper settings. The bundled MNIST
single-Shockley and measured/PWL files are Digits-derived starting points for
new experiments. See [MNIST reproduction notes](docs/MNIST_REPRODUCTION.md)
for the reported aggregate and original-seed limitation.

## Use a custom measured I–V curve

The measured/PWL family accepts an NPZ containing either equal-length 1-D `i`
and `v` arrays or one `(2, N)` `iv` array in current-then-voltage order. Values
must be real and finite, voltage strictly increasing, and current
nondecreasing.

Create `configs/local/` before writing the local asset:

```bash
mkdir -p configs/local
```

```python
import numpy as np

voltage = np.linspace(-1.5, 1.5, 301)
current = 1e-3 * voltage + 2e-3 * voltage**3
np.savez("configs/local/my_curve.npz", i=current, v=voltage)
```

A curve is never selected by an environment variable or an unrecorded runtime
flag. Copy `configs/simulator/default_pwl.json`, replace
`simulation.updater.curve` with the new repository-relative path and SHA-256,
choose `extrapolation` (`clamp` or `linear`) and `nonconvergence_policy`
(`accept_last` or `error`) explicitly, then update a copied training source's
`simulation_ref` with the copied profile's path and SHA-256. See
[Adding a nonlinearity](docs/ADDING_NONLINEARITY.md) for the complete workflow.

## Run outputs and receipts

Training and replay output directories contain the complete
`config.resolved.json` before numerical work begins. Training additionally
writes `history.json`, best/final checkpoints according to config policy,
`run_metadata.json`, and `run_receipt.json`. The receipt, rather than mutable
process state, records what ran. The demo and small-network API intentionally
write nothing; their result objects carry the same receipt information in
memory.

## Repository map

- `configs/schema/`: shared strict v2 JSON Schemas.
- `configs/train/`: data/model/training/equilibrium sources with hashed refs.
- `configs/simulator/`: typed physical and coordinate-updater profiles.
- `configs/execution/`: deterministic CPU and CUDA profiles.
- `configs/small_network/`: complete hand-specified network examples.
- `data/manifest.json`: strict replay jobs and hashed artifact references.
- `data/checksums.sha256`: the repository artifact integrity index.
- `repro/strict_config.py`: strict parsing, validation, composition, hashing,
  and JSON-Pointer overrides.
- `repro/minimizer_factory.py`: the single v2-config-to-model translation
  boundary.
- `repro/vendor/model/resistive/minimizer.py`: coordinate updater
  implementations in the original model layout.
- `scripts/reproduce.py`: paper reproduction entry point.
- `scripts/train_drn.py`: resolve-then-train entry point.
- `scripts/small_network.py`: config-only physical-network entry point.
- `tests/`: schema, integrity, numerical, plotting, and training checks.

See [data provenance](docs/DATA_PROVENANCE.md) and
`paper/figure_manifest.json` for the source-selection and paper-asset maps.

## Reproduce the scientific results

The following commands progress from a quick checkpoint evaluation to the
complete numerical replay. They use the bundled inputs and do not retrain a
model.

### Evaluate a bundled checkpoint

Evaluate the one-hidden-layer, width-64 double-Shockley checkpoint on the
complete held-out Digits split:

```bash
python scripts/reproduce.py demo
```

The demo prints 95.0% accuracy on supported reference environments. The
manifest fixes the job, batch size, assets, solver snapshot, and execution
profile. No dataset is downloaded and no files are written.

### Regenerate the paper figures

Regenerate every data-driven manuscript asset from bundled inputs:

```bash
python scripts/reproduce.py figures
```

Outputs use the manuscript's relative paths under `outputs/paper/`. Checked
files under `paper/reference/` are comparison targets, not plot inputs. To
regenerate only the two MNIST panels, run:

```bash
python scripts/reproduce.py mnist-figures
```

### Replay the numerical results

Choose either the quick spot check or the complete replay below. Validation
jobs require new or empty output directories so stale artifacts cannot be
reused.

For the spot check, list the 195 manifest jobs, run one CPU validation, and
compare it with its bundled SPICE reference:

```bash
python scripts/reproduce.py list
python scripts/reproduce.py validate --group timing --limit 1
python scripts/reproduce.py compare --group timing --limit 1
```

Alternatively, with no existing validation outputs, run the complete manifest
with:

```bash
python scripts/reproduce.py all
```

The manifest references the deterministic CPU execution profile by path and
hash. It also fixes each base configuration, checkpoint, optional SPICE state,
and job-specific equilibrium override. Wall-clock plots use bundled
measurements because timings are machine-dependent. SPICE netlist generation
is outside this artifact; the required reference arrays are included.

## Acknowledgments and provenance

This repository extends the coordinate-descent formulation and DRN simulation
framework introduced by Benjamin Scellier in
[“A fast algorithm to simulate nonlinear resistive networks”](https://proceedings.mlr.press/v235/scellier24a.html),
ICML 2024, PMLR 235:43477–43503. The original work establishes the fast exact
coordinate-descent method for ideal-diode networks; this artifact extends it
to non-ideal Shockley and measured/PWL devices.

The original MIT copyright for Benjamin Scellier, Maxence Ernoult, and Rain
Neuromorphics Inc. is retained in [LICENSE](LICENSE). See [NOTICE](NOTICE) for
the code and scientific provenance statement.
