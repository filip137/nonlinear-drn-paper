from __future__ import annotations

from pathlib import Path

import pytest
import torch

from repro.digits_validate import _build_energy_stack, _digits_loaders
from repro.execution import apply_execution_profile, seed_from_config
from repro.manifest import PackManifest
from repro.minimizer_factory import _updater_values, build_minimizer


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("family", "expected_selector_key", "expected_selector"),
    (
        (
            "single_diode_exponential",
            "single_selector",
            "overrelaxed",
        ),
        (
            "double_diode_exponential",
            "double_selector",
            "float64_overrelaxed",
        ),
        (
            "experimental",
            "double_selector",
            "overrelaxed",
        ),
    ),
)
def test_relaxation_is_an_exact_configured_choice(
    family: str,
    expected_selector_key: str,
    expected_selector: str,
) -> None:
    manifest = PackManifest.load(ROOT)
    job = next(
        job
        for job in manifest.jobs
        if job.job_id == f"timing/{family}/hidden_1/hidden_64"
    )
    updater = dict(manifest.resolved_job_config(ROOT, job).simulation["updater"])
    updater["relaxation"] = 1.0 + 1e-12

    values = _updater_values(
        family,
        updater,
        repo_root=ROOT,
    )

    assert values[expected_selector_key] == expected_selector


@pytest.mark.parametrize(
    ("family", "expected_updater"),
    (
        (
            "single_diode_exponential",
            "ConfigurableExponentialSingleDiodeUpdater",
        ),
        (
            "double_diode_exponential",
            "OverRelaxedFloat64DoubleDiodeUpdater",
        ),
        ("experimental", "ExperimentalIVCurveUpdater"),
    ),
)
def test_v2_manifest_executes_one_bundled_batch_per_family(
    family: str,
    expected_updater: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise reference resolution, composition, assets, and one equilibrium."""

    # Legacy environment knobs must not alter a fully resolved v2 job.
    monkeypatch.setenv("DRN_B_CLAMP", "7")
    monkeypatch.setenv("LABS_IV_CURVE_PATH", "missing-host-curve.npz")
    monkeypatch.setenv("LABS_IV_EXTRAPOLATION", "linear")

    manifest = PackManifest.load(ROOT)
    job_id = f"timing/{family}/hidden_1/hidden_64"
    job = next(job for job in manifest.jobs if job.job_id == job_id)
    config = manifest.resolved_job_config(ROOT, job)

    assert config.non_linearity == family
    assert config.document["execution"]["name"] == "reference_cpu"
    assert "execution_ref" not in config.document
    assert config.data["loader"] == {
        "batch_size": 1,
        "drop_last": False,
        "shuffle": False,
    }

    execution = config.document["execution"]
    device = apply_execution_profile(execution)
    seed_from_config(config.seed, execution)
    _, test_loader, data_fingerprint, split_fingerprint = _digits_loaders(
        config.data,
        execution,
        device,
    )
    inputs, labels, indices = next(iter(test_loader))
    assert inputs.shape == (1, 64)
    assert labels.shape == indices.shape == (1,)
    assert len(data_fingerprint) == len(split_fingerprint) == 64

    energy, network, free_layers, cost, _, _ = _build_energy_stack(
        cfg=config,
        weights_path=job.weights_path(ROOT),
        device=device,
    )
    minimizer = build_minimizer(
        fn=energy,
        free_layers=free_layers,
        simulation=config.simulation,
        equilibrium=config.equilibrium,
        repo_root=ROOT,
    )
    first_updater = minimizer._list_layers[0][0]
    assert type(first_updater).__name__ == expected_updater

    with torch.inference_mode():
        network.set_input(inputs, reset=True)
        states = minimizer.compute_equilibrium()
        cost.set_target(labels)
        errors = cost.error_fn()

    assert 1 <= minimizer.iterations_performed <= config.num_iterations
    assert errors.shape == (1,)
    assert all(torch.isfinite(state).all() for state in states.values())

    settings = minimizer._settings
    if family == "experimental":
        updater = config.simulation["updater"]
        assert settings.pwl_extrapolation == updater["extrapolation"] == "clamp"
        assert settings.experimental_newton_tol == updater["voltage_tolerance"]
    else:
        updater = config.simulation["updater"]
        assert settings.b_clamp == updater["linear_coefficient_clamp"] == 1e6
        assert settings.lambertw_backend == updater["backend"] == "torchlambertw"


@pytest.mark.parametrize("corruption", ("short", "shape", "dtype", "nonfinite"))
def test_replay_checkpoint_loading_is_atomic_and_strict(
    tmp_path: Path,
    corruption: str,
) -> None:
    manifest = PackManifest.load(ROOT)
    job = manifest.demo_job()
    config = manifest.resolved_job_config(ROOT, job)
    execution = config.document["execution"]
    device = apply_execution_profile(execution)
    states = torch.load(job.weights_path(ROOT), map_location=device, weights_only=True)

    if corruption == "short":
        states = states[:-1]
        message = "parameter count"
    elif corruption == "shape":
        states[0] = states[0][:-1]
        message = "shape"
    elif corruption == "dtype":
        states[0] = states[0].to(torch.float64)
        message = "dtype"
    else:
        states[0] = states[0].clone()
        states[0].reshape(-1)[0] = torch.nan
        message = "finite"

    checkpoint = tmp_path / f"{corruption}.pt"
    torch.save(states, checkpoint)
    with pytest.raises(ValueError, match=message):
        _build_energy_stack(
            cfg=config,
            weights_path=checkpoint,
            device=device,
        )
