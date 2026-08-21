#!/usr/bin/env python3
"""Compact command-line runner for Digits and MNIST nonlinear DRNs."""

from __future__ import annotations

import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "repro" / "vendor"))

from repro.cli import configure_environment
from repro.runner import main


if __name__ == "__main__":
    configure_environment()
    raise SystemExit(main(repo_root=PACK_ROOT))
