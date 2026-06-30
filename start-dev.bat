@echo off
REM Start CVAT in development mode and apply updates (Windows Batch)

echo.
echo ===================================================
echo   CVAT DEVELOPMENT UPDATE AND START SCRIPT
echo ===================================================
echo.

REM Copy Windows-specific .env file
copy /Y .env.windows .env >nul

echo [1/3] Updating and Rebuilding CVAT UI...
REM Rebuild cvat_ui to apply frontend changes
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.dev.yml -f docker-compose.dev-local.yml -f components/serverless/docker-compose.serverless.yml build --no-cache cvat_ui
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.dev.yml -f docker-compose.dev-local.yml -f components/serverless/docker-compose.serverless.yml up -d cvat_ui

echo [2/3] Ensuring all other services are running...
REM Start other services without forcing rebuild (unless missing)
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.dev.yml -f docker-compose.dev-local.yml -f components/serverless/docker-compose.serverless.yml up -d

echo.
echo Waiting for Nuclio to be ready...
timeout /t 5 /nobreak >nul

echo [3/3] Updating SAM2-OCT Function...

REM Check if the function container is already running
docker ps --filter "name=nuclio-nuclio-sam2-oct" --format "{{.ID}}" > nul 2>&1
if %errorlevel% equ 0 (
    echo Function container found. Performing HOT UPDATE...
    
    REM Copy updated code directly into the running container
    docker cp serverless/pytorch/sam2-OCT-interactor/. nuclio-nuclio-sam2-oct:/opt/nuclio/
    
    REM Restart the container to reload the code
    docker restart nuclio-nuclio-sam2-oct
    
    echo Hot update complete!
) else (
    echo Function container NOT found. Performing FULL DEPLOY...

    REM Check if nuctl is installed inside the container, if not install it
    echo Checking for nuctl...
    docker exec nuclio sh -c "if [ ! -f /usr/local/bin/nuctl ]; then echo 'Installing nuctl...'; wget -q -O /usr/local/bin/nuctl https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64 && chmod +x /usr/local/bin/nuctl; echo 'nuctl installed.'; else echo 'nuctl already installed.'; fi"

    REM Redeploy the function to apply backend code changes
    REM Using full path to nuctl as it might not be in PATH for non-interactive exec
    docker exec nuclio /usr/local/bin/nuctl deploy sam2-oct ^
      --namespace nuclio ^
      --project-name cvat ^
      --path /opt/nuclio/app/sam2-OCT-interactor ^
      --file /opt/nuclio/app/sam2-OCT-interactor/function.yaml ^
      --platform local
)

echo.
echo ===================================================
echo   UPDATE COMPLETE!
echo ===================================================
echo.
echo Access CVAT at: http://localhost:8080
echo.
pause
