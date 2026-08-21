from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.paper_figures import regenerate_mnist_assets, regenerate_paper_assets


ROOT = Path(__file__).resolve().parents[1]


def test_all_paper_assets_are_generated() -> None:
    outputs = regenerate_paper_assets(ROOT)
    assert len(outputs) == 26
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    manifest = json.loads((ROOT / "outputs" / "paper" / "asset_manifest.json").read_text())
    assert manifest["asset_count"] == 26
    output_names = {str(path.relative_to(ROOT / "outputs" / "paper")) for path in outputs}
    reference_names = {
        str(path.relative_to(ROOT / "paper" / "reference"))
        for path in (ROOT / "paper" / "reference").rglob("*")
        if path.is_file()
    }
    assert output_names == reference_names


def test_generated_tables_match_manuscript_tables() -> None:
    regenerate_paper_assets(ROOT)
    for reference in (ROOT / "paper" / "reference" / "tables").glob("*.tex"):
        generated = ROOT / "outputs" / "paper" / "tables" / reference.name
        assert generated.read_bytes() == reference.read_bytes()


def test_mnist_panels_and_protocol_are_directly_reproducible() -> None:
    outputs = regenerate_mnist_assets(ROOT)
    assert len(outputs) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)

    manifest = json.loads(
        (ROOT / "outputs" / "paper" / "mnist_asset_manifest.json").read_text()
    )
    protocol = json.loads(
        (ROOT / "data" / "paper" / "mnist" / "training_protocol.json").read_text()
    )
    training = protocol["accuracy_panel"]["double_shockley_training"]
    curves = json.loads(
        (
            ROOT
            / "data"
            / "paper"
            / "mnist"
            / "mean_test_accuracy_selected_runs_with_perfect_diode.json"
        ).read_text()
    )
    double_curve = curves["series"][0]
    assert manifest["asset_count"] == 2
    assert training["network"]["input_gain"] == 50.0
    assert training["training"]["effective_initial_learning_rates"] == [0.15, 0.08, 0.05]
    assert training["solver"]["adaptive_equilibrium"] is False
    assert double_curve["mean_accuracy_percent"][-1] == pytest.approx(
        training["reported_aggregate"]["final_mean_test_accuracy_percent"]
    )
    assert max(double_curve["mean_accuracy_percent"]) == pytest.approx(
        training["reported_aggregate"]["best_mean_test_accuracy_percent"]
    )
    assert protocol["pca_panel"]["network"]["input_gain"] == 20.0
