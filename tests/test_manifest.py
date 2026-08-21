from __future__ import annotations

import json
from pathlib import Path

from repro.manifest import PackManifest


ROOT = Path(__file__).resolve().parents[1]


def test_replay_manifest_shape() -> None:
    manifest = PackManifest.load(ROOT)
    assert len(manifest.jobs) == 195
    assert {job.group for job in manifest.jobs} == {"timing", "error_vs_iter", "vol_tol"}
    assert {job.family for job in manifest.jobs} == {
        "single_diode_exponential",
        "double_diode_exponential",
        "experimental",
    }


def test_versioned_artifact_checksums() -> None:
    PackManifest.load(ROOT).verify_checksums(ROOT)


def test_paper_manifest_has_exact_reference_coverage() -> None:
    payload = json.loads((ROOT / "paper" / "figure_manifest.json").read_text())
    declared = {entry["path"] for group in ("figures", "tables") for entry in payload[group]}
    reference = {
        str(path.relative_to(ROOT / "paper" / "reference"))
        for path in (ROOT / "paper" / "reference").rglob("*")
        if path.is_file()
    }
    assert len(declared) == 26
    assert declared == reference


def test_every_declared_figure_input_exists() -> None:
    payload = json.loads((ROOT / "paper" / "figure_manifest.json").read_text())
    for group in ("figures", "tables"):
        for entry in payload[group]:
            for relative in entry["inputs"]:
                assert (ROOT / relative).is_file(), (entry["path"], relative)


def test_scellier_software_and_paper_provenance_is_explicit() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Copyright (c) 2023 Benjamin Scellier and Maxence Ernoult" in license_text
    assert "A fast algorithm to simulate nonlinear resistive" in notice_text
    assert "https://proceedings.mlr.press/v235/scellier24a.html" in notice_text
    assert "Foundational coordinate-descent formulation" in citation_text
    assert "Acknowledgments and provenance" in readme_text
