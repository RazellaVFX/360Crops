@echo off
setlocal
set "ROOT=%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found in PATH. Install Python 3.12+ and run: pip install -r requirements.txt
  pause
  exit /b 1
)
python "%ROOT%run_zone_cropper.py"
if errorlevel 1 pause
endlocal
