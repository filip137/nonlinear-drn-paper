from __future__ import annotations

import json
from pathlib import Path

from repro.paper_figures import regenerate_paper_assets


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
