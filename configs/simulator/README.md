# Simulator profiles

These JSON files contain the physical nonlinearity and equilibrium-solver
settings used by the editable training defaults in `configs/train/`:

- `default_single_shockley.json`: shared Digits/MNIST single-diode profile;
- `default_double_shockley.json`: Digits double-diode profile;
- `default_mnist_double_shockley.json`: audited MNIST double-diode settings; and
- `default_pwl.json`: shared sampled/PWL profile and default I-V data path.

To change a device or solver setting, copy the relevant profile to the
git-ignored `configs/local/` directory and change `simulator_profile` in a
copied training template to that repository-relative path. The runtime rejects
inline simulator overrides in a composed training source, so the ownership of
each setting stays unambiguous.

`seed` remains in the training template because it controls initialization,
data splitting, and loader order. Generated and resolved run configurations
contain the fully expanded simulator settings plus the source profile path and
SHA-256 hash.

The machine-readable contract is [`schema.json`](schema.json); the complete
field guide is [`docs/CONFIG_REFERENCE.md`](../../docs/CONFIG_REFERENCE.md).
