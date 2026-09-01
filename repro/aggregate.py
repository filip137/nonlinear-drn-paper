from __future__ import annotations

import json
from pathlib import Path

from repro.manifest import PackManifest, ReproJob
from repro.npz_compare import load_comparison_summary


def aggregate_error_vs_iter(pack_root: Path) -> Path:
    manifest = PackManifest.load(pack_root)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for job in manifest.jobs_for_group("error_vs_iter"):
        summary = _read_comparison(pack_root, manifest, job)
        cfg = manifest.resolved_job_config(pack_root, job)
        family = cfg.non_linearity
        hidden = f"hidden_{len(cfg.dims) - 2}"
        grouped.setdefault(family, {}).setdefault(hidden, []).append(
            {
                "iterations": cfg.num_iterations,
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
        summary = _read_comparison(pack_root, manifest, job)
        cfg = manifest.resolved_job_config(pack_root, job)
        family = cfg.non_linearity
        hidden = f"hidden_{len(cfg.dims) - 2}"
        rel_tol = cfg.equilibrium["relative_tolerance"]
        grouped.setdefault(family, {}).setdefault(hidden, []).append(
            {
                "rel_tol": f"{rel_tol:g}",
                "rel_tol_value": rel_tol,
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


def _read_comparison(
    pack_root: Path, manifest: PackManifest, job: ReproJob
) -> dict:
    return load_comparison_summary(pack_root, manifest, job)


def _pack_rel(path: Path, pack_root: Path) -> str:
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)
