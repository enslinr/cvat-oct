#!/bin/bash
# Start CVAT in development mode and apply updates (Linux/Mac/Git Bash)

echo ""
echo "==================================================="
echo "  CVAT DEVELOPMENT UPDATE AND START SCRIPT"
echo "==================================================="
echo ""

# Detect OS and copy appropriate .env file
# Git Bash on Windows sets OSTYPE to msys or MSYSTEM to MINGW*
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$MSYSTEM" ]]; then
    echo "Detected Windows environment (Git Bash)"
    cp .env.windows .env
else
    echo "Detected Linux/Mac environment"
    cp .env.linux .env
fi

echo "[1/3] Updating and Rebuilding CVAT UI..."
# Rebuild cvat_ui to apply frontend changes
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.dev.yml -f docker-compose.dev-local.yml -f components/serverless/docker-compose.serverless.yml build --no-cache cvat_ui
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.dev.yml -f docker-compose.dev-local.yml -f components/serverless/docker-compose.serverless.yml up -d cvat_ui

echo "[2/3] Ensuring all other services are running..."
# Start other services without forcing rebuild (unless missing)
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.dev.yml -f docker-compose.dev-local.yml -f components/serverless/docker-compose.serverless.yml up -d

echo ""
echo "Waiting for Nuclio to be ready..."
sleep 5

echo "[3/3] Updating SAM2-OCT Function..."

# Check if the function container is already running
if docker ps --filter "name=nuclio-nuclio-sam2-oct" --format "{{.ID}}" | grep -q .; then
    echo "Function container found. Performing HOT UPDATE..."
    
    # Copy updated code directly into the running container
    docker cp serverless/pytorch/sam2-OCT-interactor/. nuclio-nuclio-sam2-oct:/opt/nuclio/
    
    # Restart the container to reload the code
    docker restart nuclio-nuclio-sam2-oct
    
    echo "Hot update complete!"
else
    echo "Function container NOT found. Performing FULL DEPLOY..."

    # Check if nuctl is installed inside the container, if not install it
    echo "Checking for nuctl..."
    docker exec nuclio sh -c "if [ ! -f /usr/local/bin/nuctl ]; then echo 'Installing nuctl...'; wget -q -O /usr/local/bin/nuctl https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64 && chmod +x /usr/local/bin/nuctl; echo 'nuctl installed.'; else echo 'nuctl already installed.'; fi"

    # Redeploy the function to apply backend code changes
    # Using full path to nuctl as it might not be in PATH for non-interactive exec
    docker exec nuclio /usr/local/bin/nuctl deploy sam2-oct \
      --namespace nuclio \
      --project-name cvat \
      --path /opt/nuclio/app/sam2-OCT-interactor \
      --file /opt/nuclio/app/sam2-OCT-interactor/function.yaml \
      --platform local
fi

echo ""
echo "==================================================="
echo "  UPDATE COMPLETE!"
echo "==================================================="
echo ""
echo "Access CVAT at: http://localhost:8080"
echo ""
