@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pytest -q
) else (
  py -3 -m pytest -q
)
pause
