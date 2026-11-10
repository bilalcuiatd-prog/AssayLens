@echo off
cd /d "%~dp0"
docker build -f Dockerfile.custom -t assaylens-custom:2 .
if errorlevel 1 ( echo Docker sandbox build failed. & pause & exit /b 1 )
echo Isolated custom model runtime ready.
pause
