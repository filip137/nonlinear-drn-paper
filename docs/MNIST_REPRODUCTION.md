# MNIST reproduction

The paper's MNIST figure has two panels with different provenance. The top
panel is a training-accuracy aggregate. The bottom panel is a CD–SPICE PCA
validation of a separate historical checkpoint. Their input gains are
intentionally different.

The machine-readable audit is
[`data/paper/mnist/training_protocol.json`](../data/paper/mnist/training_protocol.json).

## Reproduce the published panels

This requires no MNIST download or retraining:

```bash
python scripts/reproduce.py verify
python scripts/reproduce.py mnist-figures
```

It writes:

- `outputs/paper/figures/supplementary/mnist/mean_test_accuracy_selected_runs_with_perfect_diode.png`;
- `outputs/paper/figures/supplementary/mnist/pca_sweep_rel_l1_blue_dots_x1e3_log_scale.png`;
  and
- `outputs/paper/mnist_asset_manifest.json`, recording every numerical input
  and output digest.

The accuracy plot uses bundled 100-epoch mean/min/max curves for four selected
double-Shockley runs and four ideal-diode baselines. The PCA plot uses bundled
coordinate-descent and SPICE state arrays. Checked PNGs are never plot inputs.

## Accuracy-panel training protocol

| Setting | Paper value |
|---|---:|
| Architecture | `1568 × 100 × 20` (`(2,28,28)`, 100, differential-pair output) |
| Input gain | 50 |
| Initial optimizer rates | `0.15, 0.08, 0.05` for `DenseWeight_0`, `DenseWeight_1`, `Bias_0` |
| LR schedule | exponential, `gamma=0.99` after every epoch |
| Batch size / epochs | 16 / 100 |
| Coordinate sweeps per phase | 4 |
| EP variant / nudging | centered / 0.05 |
| Equilibrium | zero initial state; fixed sweeps; asynchronous update order |
| Diode parameters | `I_s=1e-6`, `V_t=0.05`, `V_off=1.0` |
| Updater | float64 `lambert_w_v1`, relaxation 1, no polish |

Pixels are transformed as

```text
0.3 * (pixel - 0.1307) / 0.3081
```

and concatenated with their negatives before input gain 50. The historical
name `normalize_std=0.3` denoted the final multiplier; treating 0.3 as the
normalization denominator is not equivalent. Version 2 records the unambiguous
`mean`, `standard_deviation`, and `scale` values separately.

The checked source also spells out the effective zero-momentum/zero-decay SGD
flags, exponential scheduler `initial_epoch: -1` and `after_epoch` timing, and
the `evaluation_accuracy`/`max` checkpoint rule with `tie_break: first`. These
materialized policies prevent installed-library defaults from changing an
independent replication.

### Why the rates differ from historical source labels

The archived source labels put weight rates at `[0.08, 0.05]` and the bias rate
at `[0.15]`. The historical runner concatenated the bias list first, producing
`[0.15, 0.08, 0.05]`, while optimizer parameter order was `DenseWeight_0`,
`DenseWeight_1`, `Bias_0`. The checked v2 source records the effective optimizer
mapping, preserving what ran rather than the misleading labels.

The four selected runs finish at 96.88%, 97.00%, 96.97%, and 97.23% test
accuracy. Their final mean is 97.02%; the best mean-curve value is 97.0725%.

## Launch an independent replication

Full training should use the pinned CUDA 12.4 environment:

```bash
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements-cuda.txt
python scripts/reproduce.py verify --device cuda
```

The interpreter path should end in
`nonlinear-drn-paper/.venv/bin/python`. CUDA verification and training fail
explicitly if PyTorch, the NVIDIA driver, or the device is incompatible.

The checked scientific source is
`configs/train/mnist_paper_double_shockley.json`. It contains every audited
data/model/training/equilibrium value and references the portable CPU execution
profile. To run on CUDA, create a new source whose `execution_ref` names the
hashed deterministic CUDA profile:

```bash
mkdir -p configs/local
cp configs/train/mnist_paper_double_shockley.json \
  configs/local/mnist_paper_double_cuda.json
sha256sum configs/execution/reference_cuda.json
```

Replace `execution_ref` in the copy with:

```json
{
  "path": "configs/execution/reference_cuda.json",
  "sha256": "e75ae3c2223163618a469ba8737ed3d2ccf162da6502a93832f72c856005cbb8"
}
```

Confirm that the printed digest matches, resolve the source, then run the
immutable snapshot:

```bash
python scripts/train_drn.py \
  --config configs/local/mnist_paper_double_cuda.json \
  --write-config configs/local/mnist_paper_double_cuda.resolved.json

python scripts/train_drn.py \
  --config configs/local/mnist_paper_double_cuda.resolved.json \
  --download
```

The original four run directories did not record random seeds. Therefore seeds
0–3 are independent replication seeds, not claimed identities for those
trajectories. To generate another seed from the CUDA source, replace the
existing value explicitly and write a distinct snapshot:

```bash
python scripts/train_drn.py \
  --config configs/local/mnist_paper_double_cuda.json \
  --override '/data/seed=1' \
  --write-config configs/local/mnist_paper_double_cuda_seed1.json
```

Architecture, rates, input gain, epoch count, and sweep budget are not inferred
from flags. For another depth, edit or override the complete
`model.layer_shapes`, `model.weight_gains`, and optimizer learning-rate arrays.
Using 0.15 for the first conductance, 0.08 for later conductances, and 0.05 for
hidden biases is a documented starting policy, not evidence that the paper
tested additional architectures.

Each run writes `config.resolved.json` before loading MNIST and finishes with a
receipt containing source/profile hashes, config hash, data/split fingerprints,
seed, git state, packages, platform, CUDA device/runtime, and deterministic
execution settings.

## PCA-panel distinction

The PCA panel uses a separate historical checkpoint with input gain 20,
`V_off=0.5`, and uniform initial rate 0.1. Its training config used eight
sweeps; the plotted PCA comparison evaluates coordinate descent with four.
These values must not be substituted into the gain-50 accuracy source. Raw
coordinate-descent and SPICE arrays are bundled, so regenerating the panel does
not require the proprietary simulator or original checkpoint.
