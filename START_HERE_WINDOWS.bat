@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo.
echo ========================================================
echo    AssayLens Unified Research Workbench v5.2
echo    DTI  -  DDI  -  PPI  -  Molecular ML
 echo ========================================================
echo.
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python launcher 'py' not found.
        echo Install Python 3.11 or 3.12 from python.org.
        pause
        exit /b 1
    )
    py -3.11 -m venv .venv >nul 2>nul
    if errorlevel 1 py -3.12 -m venv .venv >nul 2>nul
    if errorlevel 1 py -3 -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create a Python virtual environment.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -c "import fastapi,uvicorn,rdkit,sklearn,pandas,networkx,requests" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies into .venv ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed. Review the lines above.
        pause
        exit /b 1
    )
)
echo.
echo Web application: http://127.0.0.1:8000
echo API reference:    http://127.0.0.1:8000/api/docs
echo Keep this command window open while using the application.
echo.
".venv\Scripts\python.exe" -m uvicorn assaylens.studio_api:app --host 127.0.0.1 --port 8000
if errorlevel 1 echo Server stopped with an error. Check whether port 8000 is already in use.
pause
