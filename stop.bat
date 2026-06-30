@echo off
REM Stop CVAT (Windows Batch)

echo.
echo Stopping CVAT...
docker compose down

echo.
echo CVAT stopped
echo.