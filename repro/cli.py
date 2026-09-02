from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from repro.manifest import PackManifest


PACK_ROOT = Path(__file__).resolve().parents[1]


def _positive_limit(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def configure_environment() -> None:
    # These variables avoid OpenMP shared-memory failures in restricted runners.
    defaults = {
        "KMP_DISABLE_SHM": "1",
        "KMP_SHM_DISABLE": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/matplotlib",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nonlinear DRNs and reproduce the paper's numerical assets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List manifest jobs and bundled artifact counts.")
    verify = sub.add_parser(
        "verify",
        help="Verify the installed scientific environment and manifest checksums.",
    )
    verify.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Also require an initialized CUDA device when set to cuda.",
    )
    sub.add_parser(
        "figures",
        help="Regenerate every paper figure/table asset from bundled curated inputs.",
    )
    sub.add_parser(
        "mnist-figures",
        help="Regenerate only the paper's MNIST accuracy and PCA panels.",
    )
    demo = sub.add_parser(
        "demo",
        help="Evaluate the bundled pretrained Digits model without writing artifacts.",
    )

    train = sub.add_parser("train", help="Train one DRN configuration with equilibrium propagation.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, default=None)
    train.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="JSON_POINTER=JSON_VALUE",
        help=(
            "Generate and save a complete resolved config with an explicit JSON "
            "Pointer replacement before training. May be repeated."
        ),
    )
    train.add_argument(
        "--iv-curve",
        type=Path,
        help="Use a repository-local measured I-V .npz file.",
    )
    train.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download MNIST into data/external/.",
    )

    smoke = sub.add_parser(
        "train-smoke",
        help="Train one batch with each of the three paper nonlinearities.",
    )

    for name in ("validate", "compare", "all"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--group", choices=("timing", "error_vs_iter", "vol_tol", "all"), default="all")
        cmd.add_argument(
            "--limit",
            type=_positive_limit,
            default=None,
            help="Optional positive smoke-test limit.",
        )
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

                result = run_validation(PACK_ROOT, manifest, job)
                print(f"[validate] {job.job_id} accuracy={result.accuracy:.4f} run_dir={result.run_dir}")
            elif kind == "compare":
                from repro.npz_compare import compare_job

                summary = compare_job(PACK_ROOT, manifest, job)
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
        from repro.environment import verify_environment

        try:
            requirements_file = (
                "requirements-cuda.txt" if args.device == "cuda" else "requirements.txt"
            )
            versions = verify_environment(requirements_file=requirements_file)
        except RuntimeError as exc:
            print(f"environment: failed: {exc}", file=sys.stderr)
            return 1
        print(
            "environment: ok "
            f"(Python {versions['python']}; NumPy {versions['numpy']}; "
            f"PyTorch {versions['torch']}; Torchvision {versions['torchvision']})"
        )
        if args.device == "cuda":
            from repro.device import cuda_summary

            try:
                cuda = cuda_summary()
            except RuntimeError as exc:
                print(f"cuda: failed: {exc}", file=sys.stderr)
                return 1
            print(
                "cuda: ok "
                f"(PyTorch {cuda['torch']}; CUDA {cuda['runtime']}; {cuda['device']})"
            )
        manifest.verify_checksums()
        manifest.verify_references()
        print("checksums: ok")
        return 0
    if args.command == "figures":
        from repro.paper_figures import regenerate_paper_assets

        outputs = regenerate_paper_assets(PACK_ROOT)
        print(f"wrote {len(outputs)} paper assets under {PACK_ROOT / 'outputs' / 'paper'}")
        return 0
    if args.command == "mnist-figures":
        from repro.paper_figures import regenerate_mnist_assets

        outputs = regenerate_mnist_assets(PACK_ROOT)
        for output in outputs:
            print(f"wrote {output.relative_to(PACK_ROOT)}")
        return 0
    if args.command == "demo":
        from repro.digits_validate import run_demo

        result = run_demo(PACK_ROOT, manifest)
        print(f"demo_job: {result.job_id}")
        print(f"config: {result.config}")
        print(f"weights: {result.weights}")
        print(f"architecture: {' -> '.join(str(width) for width in result.architecture)}")
        print(f"nonlinearity: {result.non_linearity}")
        print(f"iterations: {result.num_iterations}")
        print(f"inference_batch_size: {result.batch_size}")
        print(f"execution_profile: {result.execution_profile}")
        print(f"device: {result.device}")
        print(f"correct: {result.correct}/{result.total}")
        print(f"accuracy: {100.0 * result.accuracy:.2f}%")
        return 0
    if args.command == "train":
        from repro.runner import run_drn

        result = run_drn(
            args.config,
            repo_root=PACK_ROOT,
            output_dir=args.output,
            overrides=args.override,
            iv_curve=args.iv_curve,
            download=args.download,
        )
        print(f"training output: {result.output_dir}")
        return 0
    if args.command == "train-smoke":
        from repro.train import run_smoke_suite

        results = run_smoke_suite(PACK_ROOT)
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
