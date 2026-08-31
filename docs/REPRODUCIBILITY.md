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

The six editable defaults are split into dataset-specific training templates
and reusable simulator profiles. Training templates hold data, architecture,
initialization, optimizer, fixed-sweep, and seed settings; their
`simulator_profile` points to the physical and solver settings under
`configs/simulator/`. Seed intentionally remains training-side because it
controls initialization, the data split, and loader order. All six defaults use
batch size 10.

The `default` alias resolves from both dataset and nonlinearity, although
explicit training-template paths are preferred for provenance. At load time,
the simulator profile is merged into the training configuration. Saved
generated/resolved configurations are self-contained and record
`simulator_profile_source` and `simulator_profile_sha256`.

The three checked Digits paper configurations are compact functional examples.
Each uses one hidden layer and four fixed coordinate-descent iterations. In the
compact runner, omitting the Digits architecture inherits that one-hidden-layer
source anchor; omitted iterations resolve to four for one hidden layer, eight
for two or three, and the parameter source's value for four or more. Explicit
`--num-iterations` values take precedence at every depth.

A custom measured/PWL curve is a new experiment rather than a reproduction of
the bundled measured-device result. The compact runner accepts it through
`--iv-data-path`; the generated and resolved configurations retain the chosen
path and whether it overrode the referenced simulator profile's curve. For a
reusable change, copy `configs/simulator/default_pwl.json`, edit its
`iv_data_path`, and point a copied training template at it. Accepted NPZ files
contain equal-length 1-D `i`/`v` arrays or current-first `(2, N)` `iv` data
with at least two real numeric, finite samples, strictly increasing voltage,
and nondecreasing current.

The MNIST configuration preserves the paper DRN-XS architecture and
hyperparameters:

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

Only the double-Shockley MNIST settings have an audited paper basis. The
editable `default_mnist_single_shockley.json` and
`default_mnist_custom_iv.json` templates adapt validated Digits simulator
profiles and define new experiments, not paper-MNIST reproductions.
