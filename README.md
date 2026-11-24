# Fast simulation of nonlinear deep resistive networks for energy-based computation

Use this repository to simulate and train deep resistive networks (DRNs) and
reproduce the paper’s results with the included code and data.

The simulator supports three non-linearities described in the paper:

- Single Shockley diode (`single_diode_exponential`)
- Double Shockley diode (`double_diode_exponential`)
- Piecewise-linear approximation of the measured MOSFET I–V curve (`experimental`)

We implemented two coordinate-descent (CD) solvers:

- `FastCD` uses compiled C kernels.
- `TorchCD` uses PyTorch.

With this repository, you can:

- Define an electrical network and compute its steady-state voltages.
- Train DRNs using EqProp.

## Prerequisites

- **Python 3.12 or 3.13** with `venv` and `pip`. Use Python 3.12 to reproduce the paper’s reference results.
- **Git** to clone the repository and install the pinned Lambert-W dependency.

## Clone and install

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/filip137/nonlinear-drn-paper.git
cd nonlinear-drn-paper
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Choose the CPU or NVIDIA GPU instructions below.

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

## Quick demo: simulate a small network

Use `simulate_small_network` to simulate a toy network defined in
[configs/small_network/example.json](configs/small_network/example.json).
You provide the conductance matrices, input voltages, and diode parameters.
Start from the editable example and run:

```bash
python scripts/small_network.py \
  --config configs/small_network/example.json \
  --solver fast_cd --vol_tol 1e-8
```

Choose `fast_cd` or `torch_cd` with `--solver`. Use `--vol_tol` for the maximum
node-voltage change per sweep (V) or `--cur_tol` for the maximum node current imbalance (A).

### Compare against ngspice

With `ngspice` on your `PATH`, add `--compare-ngspice` to the command:

```bash
python scripts/small_network.py --solver torch_cd --vol_tol 1e-8 --cur_tol 1e-9 --compare-ngspice
```

ngspice computes an independent DC operating point for each input. The command
reports the RMS voltage difference across all inputs and free nodes, then shows
the input, node, and two voltages with the largest absolute difference.

## Train on MNIST

To train the model on MNIST, run:

```bash
python scripts/train_drn.py \
  --config configs/train/mnist_paper_double_shockley.json \
  --download
```

The training configuration is stored in [configs/train/mnist_paper_double_shockley.json](configs/train/mnist_paper_double_shockley.json).

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
