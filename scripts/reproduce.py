#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "repro" / "vendor"))

from repro.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
