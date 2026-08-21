from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import repro.device as device_module
from repro.device import resolve_device


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_device_returns_cpu_without_probing_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> bool:
        pytest.fail("CPU selection must not require CUDA initialization")

    monkeypatch.setattr(device_module.torch.cuda, "is_available", fail_if_called)

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_returns_requested_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)

    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(
        ValueError,
        match="Expected device to be 'cpu' or 'cuda'.*Provided value: 'mps'",
    ):
        resolve_device("mps")


def test_resolve_device_reports_actionable_cuda_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module.torch, "__version__", "2.12.1+cu130")
    monkeypatch.setattr(device_module.torch.version, "cuda", "13.0")

    with pytest.raises(RuntimeError) as error:
        resolve_device("cuda")

    message = str(error.value)
    assert message.startswith("Expected CUDA to be available for --device cuda.")
    assert sys.executable in message
    assert "torch='2.12.1+cu130'" in message
    assert "torch.version.cuda='13.0'" in message
    assert "torch.cuda.is_available()=False" in message
    assert "requirements-cuda.txt" in message
    assert "source .venv/bin/activate" in message
    assert "not moved to CPU" in message


def test_cuda_requirements_pin_supported_wheels_for_both_python_stacks() -> None:
    cpu = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    cuda = (ROOT / "requirements-cuda.txt").read_text(encoding="utf-8")

    assert "-r requirements-common.txt" in cpu
    assert "-r requirements-common.txt" in cuda
    assert "https://download.pytorch.org/whl/cu124" in cuda
    for requirement in (
        'torch==2.5.1+cu124; python_version == "3.12"',
        'torchvision==0.20.1+cu124; python_version == "3.12"',
        'torch==2.6.0+cu124; python_version == "3.13"',
        'torchvision==0.21.0+cu124; python_version == "3.13"',
    ):
        assert requirement in cuda
