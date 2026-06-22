from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "V2"
EQUILIB_DIR = ROOT / "equilib"

for path in (V2_DIR, EQUILIB_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tools.zone_cropper.ui_launcher import run_app


if __name__ == "__main__":
    run_app()
