import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", ["single", "double"])
def test_comparison_command_and_failure_exit(tmp_path, monkeypatch, family):
    monkeypatch.setenv("DRN_KERNEL_CACHE", str(tmp_path / "kernels"))
    command = [sys.executable, str(ROOT / "scripts/compare_solvers.py")]
    config = json.loads((ROOT / "configs/small_network/example.json").read_text())
    if family == "single":
        profile = ROOT / "configs/simulator/default_single_shockley.json"
        config["simulation"] = json.loads(profile.read_text())["simulation"]
    config["network"]["input_gain"] = 1.5
    rows = config["network"]["input_voltages"]
    rows.append(rows[0])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command += ["--config", str(path)]
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["device"] == "cpu" and report["dtype"] == "float64"
    assert len(report["results"]) == len(rows) * 5
    assert all(r["converged"] for r in report["results"])
    for i in range(len(rows)):
        cd, torch_cd, dense, block, cached = report["results"][5 * i : 5 * i + 5]
        assert [r["solver"] for r in (cd, torch_cd, dense, block, cached)] == [
            "fast_cd", "torch_cd", "dense_lu", "block_cholesky", "newton_cached",
        ]
        for r in (torch_cd, cached):
            assert r["max_current_residual"] <= 1e-9
            np.testing.assert_allclose(r["voltages"], dense["voltages"], atol=1e-8, rtol=0)

        np.testing.assert_allclose(
            dense["voltages"], block["voltages"], atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            cd["voltages"], block["voltages"], atol=2e-5, rtol=1e-5
        )
    assert report["results"][1]["voltages"] == report["results"][-4]["voltages"]
    failed = subprocess.run(
        command + ["--max-sweeps", "0", "--max-iterations", "0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    assert not any(r["converged"] for r in json.loads(failed.stdout)["results"])
