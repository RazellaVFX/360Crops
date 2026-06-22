# 360 Zone Cropper — source package

Lightweight distribution: application code only. Python and libraries are **not** included.

## Contents

- `V2/` — application code (Qt UI + render engine)
- `equilib/` — vendored equilib projection library
- `requirements.txt` — dependencies to install with pip
- `run_zone_cropper.py` — entry point
- `run_zone_cropper.bat` — Windows launcher (uses `python` from PATH)

## Requirements

- Python 3.12+ (64-bit recommended)
- Windows 10/11
- NVIDIA GPU + driver (optional, for GPU rendering)

## Install

Open a terminal in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### CPU-only install

If you do not need CUDA:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy Pillow imagecodecs PySide6 PySide6-Fluent-Widgets
```

## Run

```powershell
python run_zone_cropper.py
```

Or double-click `run_zone_cropper.bat` (with the venv activated, or after installing into system Python).

## Notes

- `PySide6-Fluent-Widgets` pulls in `darkdetect` and `PySideSix-Frameless-Window` automatically.
- `imagecodecs` is optional at runtime but recommended for fast JPEG encoding.
- UI state is stored next to the app: `V2/tools/zone_cropper/ui_state_qt.json`.
