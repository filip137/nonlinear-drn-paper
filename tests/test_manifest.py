from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import repro.manifest as manifest_module
from repro.manifest import PackManifest, _load_release_inventory
from repro.strict_config import ConfigReference
import scripts.update_checksums as checksum_script
from scripts.update_checksums import tracked_release_paths


ROOT = Path(__file__).resolve().parents[1]


def _load_manifest_payload(
    payload: dict,
    *,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> PackManifest:
    monkeypatch.setattr(
        manifest_module,
        "load_validated_json",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(manifest_module, "_load_checksum_index", lambda path: {})
    return PackManifest.load(root)


def test_replay_manifest_shape() -> None:
    manifest = PackManifest.load(ROOT)
    assert len(manifest.jobs) == 195
    assert {job.group for job in manifest.jobs} == {"timing", "error_vs_iter", "vol_tol"}
    assert {
        manifest.resolved_job_config(ROOT, job).non_linearity
        for job in manifest.jobs
    } == {
        "single_diode_exponential",
        "double_diode_exponential",
        "experimental",
    }
    assert manifest.raw["schema_version"] == 2
    assert "checksums" not in manifest.raw
    assert manifest.execution_profile(ROOT)["name"] == "reference_cpu"


def test_versioned_artifact_checksums() -> None:
    PackManifest.load(ROOT).verify_checksums(ROOT)


def test_checksum_manifest_covers_every_versionable_file() -> None:
    manifest = PackManifest.load(ROOT)
    inventory = set(
        _load_release_inventory(ROOT / "data" / "release_inventory.txt")
    )
    tracked_raw = subprocess.run(
        ("git", "ls-files", "--cached", "-z"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tracked = {
        item.decode("utf-8")
        for item in tracked_raw.split(b"\0")
        if item and item.decode("utf-8") != "data/checksums.sha256"
    }

    assert inventory == tracked
    assert set(manifest.checksums) == inventory


def _checksum_manifest(checksums: dict[str, str]) -> PackManifest:
    return PackManifest(
        jobs=[],
        demo={},
        comparison={},
        execution_ref=ConfigReference(path="unused.json", sha256="0" * 64),
        checksums=checksums,
        raw={},
    )


def _write_release_fixture(root: Path) -> dict[str, str]:
    data = root / "data"
    data.mkdir()
    inventory = data / "release_inventory.txt"
    payload = root / "payload.txt"
    inventory.write_text(
        "data/release_inventory.txt\npayload.txt\n",
        encoding="utf-8",
    )
    payload.write_text("scientific payload\n", encoding="utf-8")
    return {
        "data/release_inventory.txt": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "payload.txt": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }


def test_checksum_verification_requires_exact_nonempty_inventory_coverage(
    tmp_path: Path,
) -> None:
    checksums = _write_release_fixture(tmp_path)
    _checksum_manifest(checksums).verify_checksums(tmp_path)

    with pytest.raises(ValueError, match="nonempty release index"):
        _checksum_manifest({}).verify_checksums(tmp_path)
    with pytest.raises(ValueError, match=r"missing=\['payload.txt'\]"):
        _checksum_manifest(
            {"data/release_inventory.txt": checksums["data/release_inventory.txt"]}
        ).verify_checksums(tmp_path)
    with pytest.raises(ValueError, match=r"unexpected=\['extra.txt'\]"):
        _checksum_manifest({**checksums, "extra.txt": "0" * 64}).verify_checksums(
            tmp_path
        )


def test_checksum_verification_fails_for_missing_inventory_file(tmp_path: Path) -> None:
    checksums = _write_release_fixture(tmp_path)
    (tmp_path / "payload.txt").unlink()

    with pytest.raises(FileNotFoundError, match="payload.txt"):
        _checksum_manifest(checksums).verify_checksums(tmp_path)


@pytest.mark.parametrize(
    "contents, message",
    [
        ("", "nonempty"),
        ("payload.txt\ndata/release_inventory.txt\n", "sorted"),
        (
            "data/release_inventory.txt\ndata/release_inventory.txt\n",
            "unique",
        ),
        ("data/release_inventory.txt\n../escape.txt\n", "normalized"),
        ("payload.txt\n", "include its own path"),
        (
            "data/checksums.sha256\ndata/release_inventory.txt\n",
            "cannot checksum itself",
        ),
    ],
)
def test_release_inventory_parser_is_strict(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "release_inventory.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_release_inventory(path)


def test_checksum_generation_uses_only_tracked_files_and_fails_on_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    data = tmp_path / "data"
    data.mkdir()
    (data / "release_inventory.txt").write_text(
        "data/release_inventory.txt\ntracked.txt\n",
        encoding="utf-8",
    )
    (data / "checksums.sha256").write_text("placeholder\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    subprocess.run(
        (
            "git",
            "add",
            "data/release_inventory.txt",
            "data/checksums.sha256",
            "tracked.txt",
        ),
        cwd=tmp_path,
        check=True,
    )

    assert tracked_release_paths(repo_root=tmp_path) == [
        "data/release_inventory.txt",
        "tracked.txt",
    ]
    monkeypatch.setattr(checksum_script, "ROOT", tmp_path)
    monkeypatch.setattr(
        checksum_script,
        "INVENTORY_PATH",
        data / "release_inventory.txt",
    )
    monkeypatch.setattr(
        checksum_script,
        "CHECKSUM_PATH",
        data / "checksums.sha256",
    )
    assert checksum_script.main() == 0
    generated = {
        relative: digest
        for digest, relative in (
            line.split("  ", maxsplit=1)
            for line in (data / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        )
    }
    assert set(generated) == {"data/release_inventory.txt", "tracked.txt"}
    _checksum_manifest(generated).verify_checksums(tmp_path)

    tracked.unlink()
    with pytest.raises(FileNotFoundError, match="tracked.txt"):
        tracked_release_paths(repo_root=tmp_path)


def test_pack_manifest_rejects_duplicate_job_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = copy.deepcopy(
        json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    )
    payload["jobs"][1]["job_id"] = payload["jobs"][0]["job_id"]

    with pytest.raises(ValueError, match="job_id to be unique"):
        _load_manifest_payload(payload, monkeypatch=monkeypatch, root=tmp_path)


@pytest.mark.parametrize(
    "unsafe_id",
    ("../escape", "/absolute", "timing//empty", "timing/unsafe value"),
)
def test_pack_manifest_rejects_unsafe_job_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_id: str,
) -> None:
    payload = copy.deepcopy(
        json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    )
    payload["jobs"][0]["job_id"] = unsafe_id

    with pytest.raises(ValueError, match="safe nonempty path segments"):
        _load_manifest_payload(payload, monkeypatch=monkeypatch, root=tmp_path)


@pytest.mark.parametrize("group", ("error_vs_iter", "vol_tol"))
def test_pack_manifest_requires_reference_states_for_comparison_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    group: str,
) -> None:
    payload = copy.deepcopy(
        json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    )
    job = next(item for item in payload["jobs"] if item["group"] == group)
    del job["assets"]["reference_states"]

    with pytest.raises(ValueError, match=rf"Expected {group} job .*reference_states"):
        _load_manifest_payload(payload, monkeypatch=monkeypatch, root=tmp_path)


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
