# Reference execution profiles

These version-2 profiles make process-wide numerical policy part of the
resolved experiment configuration. They specify the device backend, default
floating-point dtype, deterministic-algorithm policy, PyTorch/OMP thread counts,
one `blas` count applied to both MKL and OpenBLAS, data-loader worker/prefetch/
timeout policy, and random-generator seeding. The CUDA profile additionally
fixes cuDNN, TF32, and cuBLAS workspace behavior.

Scientific configuration sources reference a profile with both its
repository-relative `path` and the SHA-256 digest of its exact file bytes. The
resolver verifies that digest before composition and executes the expanded
snapshot, so editing a profile cannot silently change an archived run.

- `reference_cpu.json` is the portable reference and uses one thread and zero
  data-loader workers.
- `reference_cuda.json` is the deterministic CUDA reference for device index 0.

The CUDA profile can still fail closed when an operation has no deterministic
CUDA implementation. That is intentional: silently switching algorithms would
make two runs with the same resolved configuration incomparable.
