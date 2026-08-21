# MNIST reproduction

The paper's MNIST figure has two panels with different provenance. The top
panel is a training-accuracy aggregate. The bottom panel is a CD--SPICE PCA
validation of a separate historical checkpoint. Their input gains are
therefore intentionally different.

The machine-readable audit is
[`data/paper/mnist/training_protocol.json`](../data/paper/mnist/training_protocol.json).

## Reproduce the published panels

This command requires no MNIST download and no retraining:

```bash
python scripts/reproduce.py verify
python scripts/reproduce.py mnist-figures
```

It writes:

- `outputs/paper/figures/supplementary/mnist/mean_test_accuracy_selected_runs_with_perfect_diode.png`;
- `outputs/paper/figures/supplementary/mnist/pca_sweep_rel_l1_blue_dots_x1e3_log_scale.png`; and
- `outputs/paper/mnist_asset_manifest.json`, which records each numerical
  input and output digest.

The accuracy plot is regenerated from the bundled 100-epoch mean/min/max
curves for four selected double-Shockley runs and four ideal-diode baselines.
The PCA plot is regenerated from the bundled coordinate-descent and SPICE
state arrays. Checked reference PNGs are never plotting inputs.

## Accuracy-panel training protocol

The selected nonlinear runs used the following settings:

| Setting | Paper value |
|---|---:|
| Architecture | `1568 × 100 × 20` (`(2,28,28)`, 100, paired 20-output) |
| Input gain | 50 |
| Initial optimizer rates | `0.15, 0.08, 0.05` for `DenseWeight_0`, `DenseWeight_1`, `Bias_0` |
| LR schedule | exponential, `gamma=0.99` after every epoch |
| Batch size / epochs | 16 / 100 |
| CD sweeps per phase | 4 |
| EP variant / nudging | centered / 0.05 |
| Adaptive equilibrium | `false` |
| Diode parameters | `I_s=1e-6`, `V_t=0.05`, `V_off=1.0` |
| Updater | float64 Lambert-W, no polishing |

MNIST pixels are transformed as

```text
0.3 * (pixel - 0.1307) / 0.3081
```

and then concatenated with their negatives before applying input gain 50. The
historical field `normalize_std=0.3` named the final multiplier; treating 0.3
as the normalization denominator is not equivalent.

### Why the rates differ from the source labels

The archived source config labels weight rates as `[0.08, 0.05]` and the bias
rate as `[0.15]`. The historical runner concatenated the bias list first,
producing `[0.15, 0.08, 0.05]`, while the optimizer parameter order was
`DenseWeight_0`, `DenseWeight_1`, `Bias_0`. The checked standalone config uses
the effective optimizer mapping. This preserves what the selected runs did,
not what their field labels appeared to mean.

The four runs finish at 96.88%, 97.00%, 96.97%, and 97.23% test accuracy. Their
final mean is 97.02%; the best mean curve value is 97.0725%.

## Launch a canonical replication

Full training should use the repository's pinned CUDA 12.4 environment. From
the repository root, activate the repository virtual environment and verify
that it is the active interpreter before launching:

```bash
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements-cuda.txt
python scripts/reproduce.py verify --device cuda
```

The interpreter path must end in
`nonlinear-drn-paper/.venv/bin/python`, not the Conda `base` interpreter. CUDA
verification and CUDA training fail explicitly if the installed PyTorch build,
NVIDIA driver, and device are incompatible; there is no automatic CPU
fallback.

The ready-to-run preset contains the audited values:

```bash
python scripts/train_drn.py \
  --dataset mnist \
  --hidden-sizes 100 \
  --non-linearity double \
  --parameter-set paper-mnist-xs \
  --device cuda \
  --download \
  --seed 0
```

Omitting `--epochs`, `--num-iterations`, `--learning-rate`, and `--input-gain`
selects the paper defaults: 100 epochs, four fixed sweeps, effective rates
`[0.15, 0.08, 0.05]`, and gain 50. The generated and resolved configs are
saved beside the checkpoints.

The original four run directories did not record their random seeds. For that
reason, seeds 0--3 are the canonical independent replication set, not claimed
identities for the four original trajectories. The exact published aggregate
curve remains reproducible from the bundled data.

For a different hidden depth, the runner preserves the paper rate roles: the
first conductance uses 0.15, later conductances use 0.08, and hidden biases use
0.05. This is a documented starting policy, not a claim that the paper tested
those additional architectures. Supply `--learning-rate` to override it.

## PCA-panel distinction

The PCA panel uses a separate historical checkpoint with input gain 20,
`V_off=0.5`, and uniform initial rate 0.1. That checkpoint's training config
used eight sweeps; the plotted PCA comparison evaluates CD with four sweeps.
These settings must not be substituted into the gain-50 accuracy-training
preset. Raw CD and SPICE arrays are bundled, so regenerating the PCA panel does
not require the proprietary simulator or its original checkpoint.
