from __future__ import annotations

import argparse
import os
from pathlib import Path

from repro.manifest import PackManifest


PACK_ROOT = Path(__file__).resolve().parents[1]


def configure_environment() -> None:
    # These variables avoid OpenMP shared-memory failures in restricted runners.
    defaults = {
        "KMP_DISABLE_SHM": "1",
        "KMP_SHM_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("DRN_B_CLAMP", "1e6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nonlinear DRNs and reproduce the paper's numerical assets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List manifest jobs and bundled artifact counts.")
    sub.add_parser("verify", help="Verify manifest checksums.")
    sub.add_parser(
        "figures",
        help="Regenerate every paper figure/table asset from bundled curated inputs.",
    )

    train = sub.add_parser("train", help="Train one DRN configuration with equilibrium propagation.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--output", type=Path, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--num-iterations", type=int, default=None)
    train.add_argument("--max-batches", type=int, default=None)
    train.add_argument("--max-eval-batches", type=int, default=None)
    train.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download MNIST into data/external/.",
    )

    smoke = sub.add_parser(
        "train-smoke",
        help="Train one batch with each of the three paper nonlinearities.",
    )
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    for name in ("validate", "compare", "all"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--group", choices=("timing", "error_vs_iter", "vol_tol", "all"), default="all")
        cmd.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        cmd.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
        cmd.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def selected_jobs(manifest: PackManifest, group: str, limit: int | None):
    jobs = manifest.jobs if group == "all" else manifest.jobs_for_group(group)
    return jobs[:limit] if limit is not None else jobs


def command_list(manifest: PackManifest) -> int:
    print(f"jobs: {len(manifest.jobs)}")
    for group in ("timing", "error_vs_iter", "vol_tol"):
        print(f"{group}: {len(manifest.jobs_for_group(group))}")
    missing_reference = [job.job_id for job in manifest.jobs if job.reference_npz is None]
    print(f"jobs_without_reference_npz: {len(missing_reference)}")
    return 0


def run_many(kind: str, manifest: PackManifest, args: argparse.Namespace) -> int:
    failures = []
    for job in selected_jobs(manifest, args.group, args.limit):
        try:
            if kind == "validate":
                from repro.digits_validate import run_validation

                result = run_validation(PACK_ROOT, job, device=args.device)
                print(f"[validate] {job.job_id} accuracy={result.accuracy:.4f} run_dir={result.run_dir}")
            elif kind == "compare":
                from repro.npz_compare import compare_job

                summary = compare_job(PACK_ROOT, job)
                if summary is None:
                    print(f"[compare] {job.job_id} skipped: no reference NPZ")
                else:
                    print(f"[compare] {job.job_id} p90={summary['node_weighted_rel_l1_percentiles']['p90']:.6g}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((job.job_id, str(exc)))
            print(f"[{kind}] {job.job_id} failed: {exc}")
    if failures:
        print(f"{kind}_failures: {len(failures)}")
        return 1
    return 0


def main() -> int:
    configure_environment()
    args = parse_args()
    manifest = PackManifest.load(PACK_ROOT)
    if args.command == "list":
        return command_list(manifest)
    if args.command == "verify":
        manifest.verify_checksums()
        print("checksums: ok")
        return 0
    if args.command == "figures":
        from repro.paper_figures import regenerate_paper_assets

        outputs = regenerate_paper_assets(PACK_ROOT)
        print(f"wrote {len(outputs)} paper assets under {PACK_ROOT / 'outputs' / 'paper'}")
        return 0
    if args.command == "train":
        from repro.train import run_training

        config_path = args.config
        if not config_path.is_absolute():
            config_path = PACK_ROOT / config_path
        result = run_training(
            PACK_ROOT,
            config_path,
            device=args.device,
            output_dir=args.output,
            epochs=args.epochs,
            num_iterations=args.num_iterations,
            max_batches=args.max_batches,
            max_eval_batches=args.max_eval_batches,
            download=args.download,
        )
        print(f"training output: {result.output_dir}")
        return 0
    if args.command == "train-smoke":
        from repro.train import run_smoke_suite

        results = run_smoke_suite(PACK_ROOT, device=args.device)
        print(f"completed {len(results)} nonlinear training smoke runs")
        return 0
    if args.command == "validate":
        return run_many("validate", manifest, args)
    if args.command == "compare":
        return run_many("compare", manifest, args)
    if args.command == "all":
        status = run_many("validate", manifest, args)
        if status:
            return status
        status = run_many("compare", manifest, args)
        if status:
            return status
        from repro.aggregate import aggregate_error_vs_iter, aggregate_vol_tol
        from repro.paper_figures import regenerate_paper_assets

        aggregate_error_vs_iter(PACK_ROOT)
        aggregate_vol_tol(PACK_ROOT)
        regenerate_paper_assets(PACK_ROOT)
        return 0
    raise AssertionError(args.command)
