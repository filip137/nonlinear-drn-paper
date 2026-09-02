# Training configuration sources

Every JSON file in this directory is a complete, strict version-2 training
source. The machine-readable contract is
[`../schema/training-v2.schema.json`](../schema/training-v2.schema.json).
Unknown fields, numeric strings, non-finite numbers, missing values, and inline
simulator or execution settings are rejected.

## Single ownership

A training source owns exactly four scientific sections:

- `data`: dataset identity, preprocessing, subset/split policy, and seed;
- `model`: shapes, dtype, initialization, bounds, bias and signed-weight policy,
  output encoding, and loss;
- `training`: loader behavior, batch limits, equilibrium propagation, optimizer,
  scheduler timing, and checkpoint policy; and
- `equilibrium`: zero initialization, coordinate-update order, and the fixed
  sweep budget used for every free and nudged training phase.

It refers to, but does not duplicate, the other owners:

- `simulation_ref` names an exact file under `configs/simulator/`; and
- `execution_ref` names an exact deterministic profile under
  `configs/execution/`.

Both references contain a repository-relative path and the SHA-256 of the
referenced file bytes. Resolution fails if the file is missing or its bytes no
longer match. The resolver validates each source before composition, rejects
section collisions, validates the fully expanded configuration, and executes
that expanded snapshot.

## Bundled sources

| File | Dataset | Hidden width | Batch | Epochs | Simulator |
|---|---|---:|---:|---:|---|
| `default_single_shockley.json` | Digits | 32 | 10 | 15 | single Shockley |
| `default_double_shockley.json` | Digits | 32 | 10 | 15 | double Shockley |
| `default_custom_iv.json` | Digits | 32 | 10 | 15 | measured/PWL |
| `default_mnist_single_shockley.json` | MNIST | 100 | 10 | 15 | single Shockley |
| `default_mnist_double_shockley.json` | MNIST | 100 | 10 | 100 | double Shockley |
| `default_mnist_custom_iv.json` | MNIST | 100 | 10 | 15 | measured/PWL |
| `digits_single_shockley.json` | Digits | 64 | 10 | 15 | paper single Shockley |
| `digits_double_shockley.json` | Digits | 32 | 32 | 15 | paper double Shockley |
| `digits_pwl.json` | Digits | 32 | 32 | 15 | paper measured/PWL |
| `mnist_paper_double_shockley.json` | MNIST | 100 | 16 | 100 | paper DRN-XS |

The paper MNIST source references the deterministic CUDA execution profile;
the other bundled sources reference the portable CPU profile.

The four paper sources now use the same reference mechanism as the editable
defaults; simulator values are no longer copied inline. The paper MNIST outcome
summary remains under `provenance.expected_results` and is never read by the
scientific runtime.

## Making a variant

Do not edit a bundled source for a one-off experiment. Generate a new complete
configuration under the git-ignored `configs/local/` directory and record
changes as JSON-Pointer overrides. Scientific command-line overrides are a
configuration-generation step: the expanded file is written before training,
then training receives only that validated file.

If a simulator setting changes, copy the simulator profile too and update both
`simulation_ref.path` and `simulation_ref.sha256`. If the execution policy
changes, create or select a versioned execution profile and update both fields
of `execution_ref`. A path without its matching hash is deliberately invalid.
Measured curves are the simpler exception: use `--iv-curve` with a
repository-local NPZ path, and the generated snapshot records the change.

The migration used for the bundled records is reproducible and idempotent:

```bash
python scripts/migrate_training_configs.py
```

It migrates simulator profiles first, embeds their resulting hashes in the ten
training sources, validates every nested reference and all fourteen files
against the version-2 schemas, and produces identical bytes when rerun.
`provenance.migration` records the tool and schema transition;
`provenance.materialized_defaults` names the exact formerly implicit fields
materialized for that dataset and family. Scientific runtime code never reads
those provenance records.
