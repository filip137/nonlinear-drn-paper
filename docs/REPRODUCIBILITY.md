# Reproducibility notes

## Paper assets

`python scripts/reproduce.py figures` creates `outputs/paper/` and mirrors the
manuscript tree. A successful run has 26 assets plus
`outputs/paper/asset_manifest.json`:

- 16 plots generated with matplotlib from portable numerical inputs;
- 6 staged source schematics; and
- 4 tables, byte-identical to their checked manuscript references.

`paper/figure_manifest.json` maps every output to its input and method. Checked
reference plots are not read by plotting functions.

Regenerate the two MNIST panels independently with:

```bash
python scripts/reproduce.py mnist-figures
```

See [MNIST reproduction](MNIST_REPRODUCTION.md) and the machine-readable
`data/paper/mnist/training_protocol.json` for the distinction between gain-50
accuracy runs and the separate gain-20 PCA checkpoint.

Raster bytes can differ across matplotlib, FreeType, and operating systems even
when plotted values are identical. The pinned environment reduces that
variation; scientific comparison should inspect data and axes, not only PNG
compression bytes.

## Numerical replay

Replay is driven entirely by the strict version-2 `data/manifest.json`. Each
job references a base config, checkpoint, optional SPICE state, and execution
profile by path and exact hash. The manifest owns the job-specific equilibrium
limit; family, depth, device law, and dataset policy come from the validated
base.

`validate` evaluates the deterministic sklearn Digits split and writes the
resolved snapshot, layer states, metadata, and a run receipt. `compare`
computes node-weighted per-sample relative L1 error against the matching SPICE
NPZ. Use the same group and limit for a paired smoke run:

```bash
python scripts/reproduce.py validate --group error_vs_iter --limit 1
python scripts/reproduce.py compare --group error_vs_iter --limit 1
```

The checked manifest references `configs/execution/reference_cpu.json`; there
is no replay device override. `all` performs validation, comparison, summary
aggregation, and figure/table generation for all 195 jobs and is intentionally
expensive.

Before loading weights, replay verifies every reference, validates the base,
applies the typed equilibrium change, inserts the execution profile, and
validates the complete snapshot. Historical values that science code never
used are isolated under `provenance.historical_unused`; they cannot affect a
run.

## Training

Training sources use the same resolve-then-run contract. A source owns `data`,
`model`, `training`, and fixed-sweep `equilibrium`; it references separately
owned simulator and execution profiles by path and SHA-256. Resolution expands
those profiles into a self-contained document and records the exact sources.
The document is written as `config.resolved.json` before data loading or model
construction.

Centered equilibrium propagation is selected explicitly by the checked
sources. Each mini-batch is relaxed at the unnudged energy and evaluated at the
two configured nudging signs. Gradients are assigned to the declared trainable
parameters, and conductance bounds are enforced after every optimizer step.
Training requires `equilibrium.method: fixed_sweeps`, so every free and nudged
phase uses the same declared coordinate-update budget. Bundled sources also
declare `initial_state: zeros`; adaptive stopping is available only to replay
and small-network simulation, not training.

Digits sources explicitly use floor rounding when converting the seeded 0.8
train fraction to a sample count. The complete SGD flags, exponential-scheduler
initial epoch and step timing, and first-on-tie checkpoint policy are likewise
stored in each source rather than inherited from PyTorch defaults.

There are no dataset/nonlinearity aliases or granular numerical defaults in the
runner. Launch a checked source directly:

```bash
python scripts/train_drn.py \
  --config configs/train/digits_double_shockley.json
```

Generate a changed experiment with explicit JSON-Pointer replacements:

```bash
mkdir -p configs/local
python scripts/train_drn.py \
  --config configs/train/digits_double_shockley.json \
  --override '/training/epochs=1' \
  --override '/equilibrium/sweeps=2' \
  --override '/training/batch_limits/train=1' \
  --override '/training/batch_limits/evaluation=1' \
  --write-config configs/local/digits_double_smoke.json
```

Every accepted change is stored in `provenance.generation_overrides`. See the
[training guide](TRAINING_RUNNER.md) and
[configuration reference](CONFIG_REFERENCE.md) for source ownership, CUDA
profiles, hashing, and Python APIs.

A custom measured/PWL curve is a new experiment. It must live inside the
repository and be referenced by path and SHA-256 from a copied PWL simulator
profile. The profile itself is then referenced by path and SHA-256 from a
copied training source. There is no environment or one-run curve override.
Accepted NPZ layouts and the complete procedure are documented in
[Adding a nonlinearity](ADDING_NONLINEARITY.md).

## MNIST protocol summary

The checked MNIST source preserves the paper DRN-XS contract:

- input shape `(2, 28, 28)` after positive/negative duplication;
- hidden width 100 and differential-pair 20-node output;
- double Shockley with `I_s=1e-6`, `V_t=0.05`, and `V_off=1`;
- four fixed coordinate-descent sweeps;
- centered equilibrium propagation with nudging 0.05;
- effective rates `[0.15, 0.08, 0.05]` in weight/weight/bias order;
- exponential scheduler gamma 0.99 after each epoch; and
- 100 epochs with batch size 16.

MNIST normalization is `0.3 * (pixel - 0.1307) / 0.3081`, followed by signed
input duplication and input gain 50. The four selected historical runs did not
record seeds. Bundled aggregate curves reproduce the published figure; seeds
0–3 define independent replications, not identities for the original
trajectories.

Seeds, dtype, deterministic algorithms, thread counts, loader workers, and
CUDA policy are all explicit configuration. Hardware and library changes can
still produce small floating-point differences; the receipt records those
facts rather than implying bitwise portability across devices.

Only double Shockley has audited MNIST paper settings. The editable MNIST
single-Shockley and PWL sources adapt validated Digits profiles and define new
experiments.
