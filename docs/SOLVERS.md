# Solvers used in the paper revisions

The review implementations are available in `repro.solvers`. They accept arrays
and run independently of the local review folders, bundles, launchers, and
machine paths. The existing Torch minimizer remains the entry point for training
and archived manifest replay.

| Implementation | Device law | Update and stopping rule |
| --- | --- | --- |
| `FastCD` | Single or double Shockley | CPU float64, one input; red-black layer sweeps, fixed or adaptive relaxation; voltage-change stop |
| `TorchCD` | Single or double Shockley | CPU float64, fused Torch updates with adaptive relaxation, full-law current-residual stop |
| `newton` | Single/double Shockley or `TableLaw` | Joint KCL solve; dense LU or block Cholesky; Armijo damping by default |
| `newton_cached` | Same as `newton` | Block Cholesky with Jacobian reuse; refresh on slow progress or failed stale direction; current-residual stop |
| `PWLCoordinateDescent` | Monotone measured tables | CPU float64, one input; sequential layers, optional compiled sweep and adaptive relaxation; current-residual stop |
| `coordinate_descent` | Single/double Shockley | Independent full-law numerical control, using bracketed scalar Newton updates |

## Run a comparison

From the repository root, after installing the documented dependencies:

```bash
python scripts/compare_solvers.py --config configs/small_network/example.json
```

The command runs FastCD, Torch CD, dense-LU Newton, block-Cholesky Newton, and
cached Newton on each input row. Its JSON reports voltages, convergence,
iteration counts, and the full-law KCL residual. A failed solve gives a nonzero
exit status. Use repeated selectors to run a subset, for example
`--solver torch_cd --solver newton_cached`. Only FastCD loads the C kernel.

All five use the same conductances, clamped inputs, physical parameters, and
amplifier gains in CPU float64. FastCD uses `--rtol` and `--atol` for its
voltage-change stop. Torch CD and Newton use `--current-tolerance` in amperes.
The CD sweep budget is `--max-sweeps`, and Newton uses `--max-iterations`.
For config-driven replay use `scripts/small_network.py`.

In Python, construct `TorchCD(weights, simulation)` with a v2 simulation block
and call `solve(clamped_input, tolerance=1e-9)` for each row. Apply the input
gain before calling. Torch CD uses the repository's fused Shockley kernels,
the supplied updater's clamp and polish settings, and two-term adaptive
relaxation starting at one. Pass `compiled=True` at construction to enable
`torch.compile`. Each call returns a `SolveResult` and starts from zero.

FastCD needs a C compiler (`CC`, `cc`, or `gcc`). On first construction it builds
`fast_kernels.c` into `DRN_KERNEL_CACHE`, or
`~/.cache/nonlinear-drn/kernels`. The cache key includes source, compiler identity,
and flags. No generated library is written into the source tree. PWL sweep
compilation uses `torch.compile` and its normal cache; `compiled=False` runs the
same sweep eagerly. These requirements add no Python package dependencies.

## Shockley API

```python
import numpy as np
from repro.solvers import AdaptiveRelaxation, DoubleShockley, FastCD, LayeredDRN, newton

weights = [np.array([[0.1, 0.2], [0.3, 0.1]]), np.array([[0.2], [0.4]])]
circuit = LayeredDRN(weights, DoubleShockley(), voltage_gain=4.0, current_gain=1.0)
forcing = circuit.forcing(np.array([2.0, -1.0]))

cd = FastCD(weights, "double", av=4.0, ai=1.0)
sweeps = cd.solve(forcing[circuit.slices[0]], AdaptiveRelaxation())
assert cd.converged, cd.reason
cd_voltages = cd.state

reference = newton(circuit, forcing, backend="block_cholesky",
                   residual_kind="current", tolerance=1e-9)
assert reference.converged, reference.reason
```

Matrices are input-to-output: `weights[k]` has shape `(previous_width, next_width)`.
Conductances are nonnegative. The input is clamped and is not multiplied by the
inter-layer voltage gain. States concatenate hidden layers then the linear output
layer. `forcing()` accepts one input or a batch; each solver call handles one row.
Newton supports constant bias-current vectors through `LayeredDRN`. FastCD is the
bias-free review solver and takes only the first-layer forcing `input @ weights[0]`.

For oriented single diodes, use `SingleDiodeDRN(weights, physical, av, ai)` and
`FastCD(weights, "single", av, ai, physical=physical)`. Orientations are positive
on the first `width // 2` nodes of each hidden layer and negative on the rest.
The paper uses even hidden widths. `physical` contains `saturation_current`,
`thermal_voltage`, and `offset_voltage`; defaults are `1e-6 A`, `0.05 V`, `0.8 V`.
FastCD requires positive saturation current. Newton also supports a zero-current
linear device. Instances of CD own mutable buffers; use separate instances for
concurrent solves. Every `solve` starts from zero and clears its adaptive history.

FastCD preserves the review's small-argument series, warm Lambert-W starts, and
no-polish defaults (`small=True, warm=True, polish=False`). Its double-diode
coordinate kernel keeps the dominant exponential and drops the opposite branch.
The omitted current is at most `Is * exp(-Voff / Vt)` (about `1.13e-13 A` with
paper parameters). Newton and the independent CD control evaluate the full
antiparallel law. This approximation bound changes with the device parameters.
FastCD also retains the review's `|b| <= 1e6` coordinate forcing clamp.

A voltage-change stop does not certify a particular KCL residual. Inspect
`circuit.residual(cd.state, forcing)` when comparing to a current threshold.
Reaching `max_sweeps` is reported by `cd.converged == False` and `cd.reason`.
Newton's default tolerance is the conductance-scaled residual in volts; explicitly
select `residual_kind="current"` for a KCL threshold in amperes. The Newton result
also reports backtracks, residual evaluations and failure reason. Undamped/fixed
steps and voltage-change stopping remain available for reproducing the controls.

## Measured and transistor tables

```python
from repro.solvers import LayeredDRN, PWLCoordinateDescent, TableLaw, newton_cached

# Shared measured grid, signed physical currents in A, hidden widths [2].
voltage = np.array([-2.0, 0.0, 2.0])
nmos = np.array([-0.1, 0.0, 2.0]) * 1e-7
pmos = np.array([-2.0, 0.0, 0.1]) * 1e-7
law = TableLaw.paired(voltage, nmos, pmos, [2], current_unit_A=1e-7)
circuit = LayeredDRN(weights, law, voltage_gain=4.0, current_gain=1.0)
forcing = circuit.forcing(np.array([2.0, -1.0]))
cd = PWLCoordinateDescent(circuit, adaptive=True, compiled=True)
result = cd.solve(forcing, tolerance=1e-9)
assert result.converged, result.reason
cached = newton_cached(circuit, forcing, theta=0.9, tolerance=1e-10)
assert cached.converged, cached.reason
```

`TableLaw(voltage, currents, kinds, current_unit_A=...)` also supports one shared
measured law or several device types. Each entry in `kinds` selects a current-table
row for one hidden node. Paired tables assign NMOS then PMOS within each hidden
layer; PMOS currents are already signed and are not mirrored. Knots must increase,
currents must be finite and nondecreasing, and paired hidden widths must be even.

The review's compiled L2/L3 and Newton variants extend the first and last linear
segments outside the table. This behavior is preserved; extrapolation does not
establish physical validity beyond the characterized voltage range. The bounded
L1 experiment and its foundry-data loaders are not part of this port.

With `current_unit_A=1e-7`, conductances and residuals are in network units: a
numerical conductance of 1 means `1e-7 S`, and a tolerance of `1e-9` means
`1e-16 A`. Use the same scaling for weights, bias currents and the table law.
`newton_cached` retains its original default `theta=0.5`; the transistor review
used `theta=0.9`, set explicitly above.

## Source and validation

The numerical implementations come from the October–November 2026 review runs:
`solver.py`, `single_diode_solver.py`, `fast_cd_20261027/fast_cd.py` and its C
kernel, and the transistor bundle's `newton_cached.py` and `transistor.py`.
The port removes experiment I/O, exposes array-based table construction, shares
the adaptive policy, validates public inputs, and reports exhausted solves.
No remote runs or private datasets are needed by the package or tests.

```bash
python -m pytest tests/test_review_newton.py tests/test_review_fast_cd.py tests/test_review_pwl.py tests/test_solver_comparison.py
```

These checks cover independent scalar roots and finite-difference Jacobians,
dense/block linear solves, full-law equilibrium agreement, repeated-input reset,
table units and endpoint extrapolation, compiled/eager sweeps, failure handling,
and the public comparison command.
