# Data provenance and curation

This artifact was curated from the companion research workspace on 20 August
2026. The selection rule was: retain an artifact only when it is needed to
regenerate a paper asset, replay a reported numerical comparison, or exercise
one of the three paper nonlinearities in training.

## Bundled numerical data

### Digits coordinate-descent/SPICE core

`data/error_vs_iter/`, `data/vol_tol/`, and `data/timing/` contain 195 selected
jobs. They include:

- explicit JSON configurations;
- trained parameter tensors (`weights.pt`);
- SPICE/reference voltage arrays required for comparisons;
- curated error summaries and CPU timing CSVs used by the figures; and
- the three timing tables used by the manuscript.

The jobs cover one, two, and three hidden layers for the single-Shockley,
double-Shockley, and measured/PWL cases. A few points for which the source
campaign did not produce a valid SPICE artifact are listed without a reference
NPZ in `data/manifest.json`; the `compare` command reports these as skipped.

`data/assets/experimental_curve_voff_0.8_200_points.npz` is the 200-point I–V
curve used by the measured/PWL coordinate updater. It is also the default input
for the PWL training configuration.

### Supplementary paper inputs

`data/paper/` contains only the arrays or tabular values used by supplementary
assets:

- `mnist/`: mean/min/max test-accuracy series for four selected
  double-Shockley runs and four perfect-diode baseline runs;
- `pca_sweep/`: the raw 30×30 coordinate-descent grid, matching SPICE layer
  states, and portable grid metadata;
- `overrelaxation/`: P90 error values for the hidden-3, width-128 relaxation
  sweep;
- `conditioning/`: numerical conditioning/runtime metrics, with original
  machine-specific path columns removed;
- `component_timing/`: float32/float64 local-update timing components; and
- `accuracy_ladder/`: the minimal rows needed to regenerate the accuracy
  ladder table.

The PCA NPZ files are important: an earlier reviewer bundle kept only the
rendered PCA PNG, which was insufficient for reproduction. This repository
includes the underlying coordinate-descent and SPICE arrays.

### Source artwork and reference assets

`paper/reference/` is the exact 22-figure/4-table tree used by the manuscript.
The sixteen plots are reference outputs only; plotting code reads exclusively
from `data/`. Six circuit/device schematics are source artwork rather than
numerically generated plots, so `figures` stages byte-identical copies and
labels them `source-artwork:copy` in its output manifest.

## Deliberate exclusions

The following are not needed for the claims reproduced here and are therefore
not bundled:

- raw MNIST images (torchvision can download them with explicit `--download`);
- TensorBoard event files and complete training run directories;
- Optuna databases and exploratory sweeps;
- duplicate rendered plots;
- intermediate per-node diagnostics not used by the paper; and
- a SPICE installation, netlist generator, or proprietary simulator output
  workflow.

The selected SPICE state arrays and recorded SPICE timings are included.
Timings are intentionally replotted rather than rerun because wall-clock values
depend on hardware and simulator configuration.

## Integrity

`data/manifest.json` contains the replay job definitions and SHA256 digests for
all versioned artifact files. `data/checksums.sha256` provides the same digest
set in the conventional `sha256sum` format. Run:

```bash
python scripts/reproduce.py verify
```

Generated files under `outputs/` and Git internals are excluded from the
digest set.
