@echo off
REM Start CVAT with SAM2 and auto-initialization (Windows Batch)

echo.
echo Starting CVAT with SAM2...

REM Copy Windows-specific .env file
copy /Y .env.windows .env >nul

docker compose up -d

echo.
echo ===================================
echo CVAT is starting up!
echo ===================================
echo.
echo Monitor progress:
echo   docker logs nuclio_init -f
echo.
echo Access CVAT at: http://localhost:8080
echo.
echo SAM2 deployment takes 2-5 minutes on first run
echo.