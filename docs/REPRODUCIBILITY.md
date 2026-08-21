# Reproducibility notes

## Paper assets

`python scripts/reproduce.py figures` creates `outputs/paper/` and mirrors the
manuscript tree. A successful run has 26 assets plus
`outputs/paper/asset_manifest.json`:

- 16 plots generated with matplotlib from portable numerical inputs;
- 6 staged source schematics; and
- 4 tables, all byte-identical to their checked manuscript references.

`paper/figure_manifest.json` maps every output to its input and method. The
checked reference plots are not read by the plotting functions.

The two MNIST panels can be regenerated independently with:

```bash
python scripts/reproduce.py mnist-figures
```

See `docs/MNIST_REPRODUCTION.md` and the machine-readable
`data/paper/mnist/training_protocol.json` for the audited distinction between
the gain-50 accuracy runs and the gain-20 PCA checkpoint.

Raster bytes can differ across matplotlib, FreeType, or operating-system
versions even when curves and values are identical. The pinned Python
environment reduces that variability; scientific reproduction should compare
the plotted values and axes, not only the compressed PNG bytes.

## Numerical replay

The `validate` command loads a selected checkpoint, evaluates the deterministic
sklearn Digits test split, and writes layer states. `compare` computes the
node-weighted per-sample relative L1 error against its matching SPICE NPZ. Use
the same group and limit for a paired smoke run:

```bash
python scripts/reproduce.py validate --group error_vs_iter --device cpu --limit 1
python scripts/reproduce.py compare --group error_vs_iter --limit 1
```

`all` performs validation, comparison, summary aggregation, and figure/table
generation. It is intentionally expensive for the full 195-job matrix.

## Training

Training uses centered equilibrium propagation by default. Each mini-batch is
first relaxed with the unnudged energy and then evaluated at the two configured
nudging signs. Gradients are assigned directly to the conductance tensors and
other configured trainable tensors, and conductance bounds are enforced after
every optimizer step.
Adaptive equilibrium is disabled for training so each phase performs the
configured fixed number of coordinate-descent iterations.

The three Digits configurations are compact functional examples. The MNIST
configuration preserves the paper DRN-XS architecture and hyperparameters:

- input shape `(2, 28, 28)` after positive/negative input duplication;
- hidden width 100 and paired 20-node output;
- double Shockley diode with `I_s=1e-6`, `V_t=0.05`, and `V_off=1`;
- four coordinate-descent iterations;
- centered EP with nudging 0.05; and
- effective initial rates `[0.15, 0.08, 0.05]` in weight/weight/bias order;
- an exponential learning-rate scheduler with gamma 0.99; and
- 100 epochs with batch size 16.

MNIST normalization is `0.3 * (pixel - 0.1307) / 0.3081`, followed by signed
input duplication and input gain 50. The four original selected runs did not
record seeds; the bundled aggregate curves reproduce the published figure,
while standalone seeds 0--3 define independent replication runs.

Seeds fix parameter initialization, data splitting, and loader order. Floating
point reductions and nonlinear solver trajectories can still vary slightly
between CPU/GPU models and PyTorch builds.
