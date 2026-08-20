from __future__ import annotations

import json
from pathlib import Path

from repro.manifest import PackManifest, ReproJob


def aggregate_error_vs_iter(pack_root: Path) -> Path:
    manifest = PackManifest.load(pack_root)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for job in manifest.jobs_for_group("error_vs_iter"):
        summary = _read_comparison(pack_root, job)
        if summary is None:
            continue
        family = job.family
        hidden = f"hidden_{job.hidden_layers}"
        grouped.setdefault(family, {}).setdefault(hidden, []).append(
            {
                "iterations": job.num_iterations,
                **summary["node_weighted_rel_l1_percentiles"],
                "source": _pack_rel(job.comparison_dir(pack_root), pack_root),
            }
        )
    out_dir = pack_root / "outputs" / "summaries" / "error_vs_iter"
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, hidden in grouped.items():
        for rows in hidden.values():
            rows.sort(key=lambda item: item["iterations"])
        (out_dir / f"{family}.json").write_text(
            json.dumps({"non_linearity": family, "hidden": hidden}, indent=2),
            encoding="utf-8",
        )
    return out_dir


def aggregate_vol_tol(pack_root: Path) -> Path:
    manifest = PackManifest.load(pack_root)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for job in manifest.jobs_for_group("vol_tol"):
        summary = _read_comparison(pack_root, job)
        if summary is None:
            continue
        family = job.family
        hidden = f"hidden_{job.hidden_layers}"
        grouped.setdefault(family, {}).setdefault(hidden, []).append(
            {
                "rel_tol": job.rel_tol,
                "rel_tol_value": float(job.rel_tol),
                "p90": summary["node_weighted_rel_l1_percentiles"]["p90"],
                "source": _pack_rel(job.comparison_dir(pack_root), pack_root),
            }
        )
    out_dir = pack_root / "outputs" / "summaries" / "vol_tol"
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, hidden in grouped.items():
        for rows in hidden.values():
            rows.sort(key=lambda item: item["rel_tol_value"])
        (out_dir / f"{family}.json").write_text(
            json.dumps({"non_linearity": family, "hidden": hidden}, indent=2),
            encoding="utf-8",
        )
    return out_dir


def _read_comparison(pack_root: Path, job: ReproJob) -> dict | None:
    path = job.comparison_dir(pack_root) / "cross_layer_rel_l1_percentiles_node_weighted.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_rel(path: Path, pack_root: Path) -> str:
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)
