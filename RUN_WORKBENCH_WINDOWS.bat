@echo off
setlocal
cd /d "%~dp0"
title AssayLens Workbench 1.1
echo.
echo AssayLens Research Workbench 1.1
echo Local web interface: http://127.0.0.1:8000
echo.
py -3 -c "import fastapi, pandas, rdkit, sklearn, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo Installing required Python packages...
    py -3 -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)
echo.
echo Starting application...
echo Close previous AssayLens command windows if port 8000 is already occupied.
py -3 -m uvicorn assaylens.studio_api:app --host 127.0.0.1 --port 8000
echo.
echo The web server stopped or could not start.
pause
