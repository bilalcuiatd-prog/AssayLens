@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)
echo Starting ChEMBL enzyme dataset extraction for AssayLens AI...
"%PYTHON_EXE%" scripts\download_chembl_enzyme.py --target CHEMBL205 --endpoint Ki --min-assay-records 10 --seed 42
if errorlevel 1 (
    echo.
    echo Extraction failed. Read the error above and see docs\chembl_data_protocol.md
    pause
    exit /b 1
)
echo.
echo Ready: datasets\CHEMBL205_Ki\assaylens_import.csv
pause
