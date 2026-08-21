# Training configurations

The JSON files in this directory are ready-to-run, checked parameter sources
for the three paper nonlinearities. They share the editor-aware
[`schema.json`](schema.json). Editors that understand JSON Schema display a
field's definition and allowed values when hovering over it.

For equations, units, numerical tradeoffs, field applicability, and guidance
for changing solver values, read
[`docs/CONFIG_REFERENCE.md`](../../docs/CONFIG_REFERENCE.md).

## Bundled parameter sources

| File | Dataset | Active nonlinear updater |
|---|---|---|
| `digits_single_shockley.json` | Digits | single-Shockley Lambert-W |
| `digits_double_shockley.json` | Digits | double-Shockley float64 Lambert-W with overrelaxation |
| `digits_pwl.json` | Digits | measured/PWL damped Newton |
| `mnist_paper_double_shockley.json` | MNIST | paper DRN-XS float64 Lambert-W |

These files use a complete common schema. Consequently, they retain some
compatibility fields that are inactive for the chosen nonlinearity. In
particular, Lambert-W settings do not affect measured/PWL runs, and measured
Newton settings do not affect Shockley runs. The applicability table in the
configuration reference is authoritative.

JSON does not support comments. Keep explanatory text in the shared schema and
reference rather than adding ad hoc comment keys to individual experiment
files. Use `python scripts/train_drn.py ... --dry-run` to inspect the fully
expanded configuration before launching a run.
