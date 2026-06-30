#!/bin/sh
# Automatic Nuclio initialization script
# This script runs when the nuclio container starts and automatically:
# 1. Installs nuctl CLI tool
# 2. Creates the CVAT project
# 3. Deploys the SAM2 function

set -e

echo "==> Initializing Nuclio for CVAT..."

# Set nuctl to use the nuclio dashboard
export NUCTL_PLATFORM="local"
export NUCTL_NAMESPACE="nuclio"
export NUCTL_REGISTRY=""
export NUCTL_RUN_REGISTRY=""

# Check if nuctl is already installed
if ! command -v nuctl > /dev/null 2>&1; then
    echo "==> Installing nuctl..."
    cd /tmp
    wget -q https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
    chmod +x nuctl-1.13.0-linux-amd64
    mv nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
    echo "==> nuctl installed successfully"
else
    echo "==> nuctl already installed"
fi

# Wait for nuclio dashboard to be ready
echo "==> Waiting for Nuclio dashboard to be ready..."
sleep 5  # Give nuclio a bit more time to fully start up
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if wget --spider -q http://nuclio:8070 2>/dev/null; then
        echo "==> Nuclio dashboard is ready"
        break
    fi
    attempt=$((attempt + 1))
    echo "==> Waiting for dashboard... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "==> ERROR: Nuclio dashboard failed to become ready"
    exit 1
fi

# Check if CVAT project exists
echo "==> Checking for CVAT project..."
if nuctl --platform local get project cvat 2>/dev/null; then
    echo "==> CVAT project already exists"
else
    echo "==> Creating CVAT project..."
    nuctl --platform local create project cvat
fi

# Check if SAM2 function is already deployed
echo "==> Checking for SAM2 function..."
if nuctl --platform local get function sam2 --project-name cvat 2>/dev/null | grep -q "ready"; then
    echo "==> SAM2 function already deployed and ready"
else
    echo "==> Deploying SAM2 segmentation function..."
    echo "==> This may take 2-5 minutes on first run..."

    nuctl deploy \
        --project-name cvat \
        --path /opt/nuclio/app/sam2-interactor \
        --file /opt/nuclio/app/sam2-interactor/function.yaml \
        --platform local

    echo "==> SAM2 function deployed successfully"
fi

# Check if SAM2-OCT function is already deployed
echo "==> Checking for SAM2-OCT function..."
if nuctl --platform local get function sam2-oct --project-name cvat 2>/dev/null | grep -q "ready"; then
    echo "==> SAM2-OCT function already deployed and ready"
else
    echo "==> Deploying SAM2-OCT segmentation function..."
    echo "==> This may take 2-5 minutes on first run..."

    nuctl deploy \
        --project-name cvat \
        --path /opt/nuclio/app/sam2-OCT-interactor \
        --file /opt/nuclio/app/sam2-OCT-interactor/function.yaml \
        --platform local

    echo "==> SAM2-OCT function deployed successfully"
fi

echo "==> Nuclio initialization complete!"