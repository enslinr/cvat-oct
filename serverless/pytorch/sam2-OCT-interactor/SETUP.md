# SAM2 Semantic Interactor for CVAT - Setup Guide

This guide provides step-by-step instructions to deploy the SAM2 Semantic Interactor in CVAT locally.

## Prerequisites

- Docker Desktop installed and running
- Git
- At least 10GB free disk space
- Windows, macOS, or Linux

## Step 1: Clone the Repository

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
```

## Step 2: Download the SAM2 Model Checkpoint

1. Download the SAM2.1 Hiera Base Plus checkpoint:
   - URL: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
   - Size: ~309 MB

2. Place the checkpoint file in:
   ```
   cvat/serverless/pytorch/sam2-OCT-interactor/models/sam2.1_hiera_base_plus.pt
   ```

## Step 3: Start CVAT with Serverless Components

### On Windows (PowerShell):
```powershell
docker compose `
  -f docker-compose.yml `
  -f docker-compose.override.yml `
  -f components/serverless/docker-compose.serverless.yml `
  up -d
```

### On Linux/macOS (Bash):
```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

Wait for all containers to start (this may take several minutes on first run).

## Step 4: Install nuctl CLI in Nuclio Container

```bash
# Enter the Nuclio container
docker exec -it nuclio bash

# Download and install nuctl 1.13.0
wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
chmod +x nuctl-1.13.0-linux-amd64
mv nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl

# Verify installation
nuctl version

# Exit the container
exit
```

## Step 5: Create Nuclio Project

```bash
docker exec nuclio nuctl create project cvat --platform local
```

## Step 6: Deploy the SAM2 Function

### For CPU (recommended for most users):
```bash
docker exec nuclio nuctl deploy sam2-semantic \
  --namespace nuclio \
  --project-name cvat \
  --path /opt/nuclio/app/sam2-OCT-interactor \
  --file /opt/nuclio/app/sam2-OCT-interactor/function.yaml \
  --platform local
```

### For GPU (if you have NVIDIA GPU with CUDA support):
```bash
docker exec nuclio nuctl deploy sam2-semantic-gpu \
  --namespace nuclio \
  --project-name cvat \
  --path /opt/nuclio/app/sam2-OCT-interactor \
  --file /opt/nuclio/app/sam2-OCT-interactor/function-gpu.yaml \
  --platform local
```

**Note**: The deployment will take 10-15 minutes as it builds the Docker image and downloads dependencies.

## Step 7: Verify Deployment

```bash
# Check function status
docker exec nuclio nuctl get functions --namespace nuclio

# You should see:
# NAMESPACE | NAME          | PROJECT | STATE | REPLICAS | NODE PORT
# nuclio    | sam2-semantic | cvat    | ready | 1/1      | <port>

# Check function logs to verify model loaded
docker logs nuclio-nuclio-sam2-semantic 2>&1 | grep -i "model ready"

# You should see 4 workers report: "SAM2 model ready."
```

## Step 8: Access CVAT

1. Open your browser and navigate to: `http://localhost:8080`
2. Create an account or log in
3. Create a new task and upload an image
4. In the annotation view, the SAM2 Semantic Interactor should appear in the AI tools sidebar

## Using SAM2 in CVAT

1. Click on an object in the image (positive point)
2. Optionally add negative points to refine the segmentation
3. The SAM2 model will generate a segmentation mask
4. Adjust the threshold if needed to refine the results

## Troubleshooting

### Function Shows as "Unhealthy"
- Check logs: `docker logs nuclio-nuclio-sam2-semantic`
- Ensure the model checkpoint exists and is the correct size (~309 MB)
- Verify the function container is on the correct network: `docker inspect nuclio-nuclio-sam2-semantic --format '{{range $net, $config := .NetworkSettings.Networks}}{{$net}}{{"\n"}}{{end}}'`
  - Should show: `cvat_cvat`

### Function Stuck in "Building" State
This can happen if deployment was interrupted. Clean up and restart:

```bash
# Stop CVAT
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down

# Remove the Nuclio storage volume
docker volume rm nuclio-local-storage

# Restart CVAT (use the full command from Step 3)
```

### Out of Disk Space During Deployment
```bash
# Clean up Docker resources
docker system prune -a -f
docker builder prune -a -f
```

### 500 Error When Using SAM2
- Ensure the function is on the correct Docker network (see "Function Shows as Unhealthy")
- Check that `docker-compose.override.yml` contains the network configuration
- Restart the function if needed:
  ```bash
  docker exec nuclio nuctl delete function sam2-semantic --namespace nuclio
  # Then redeploy using Step 6
  ```

## System Requirements

### Minimum:
- CPU: 4 cores
- RAM: 8 GB
- Disk: 10 GB free space
- Network: Stable internet connection for initial setup

### Recommended:
- CPU: 8+ cores
- RAM: 16 GB
- Disk: 20 GB free space
- GPU: NVIDIA GPU with 8+ GB VRAM (for GPU deployment)

## Performance Notes

- **CPU mode**: ~2-5 seconds per inference (depends on image size)
- **GPU mode**: ~0.5-1 second per inference
- The model uses ~2 GB RAM per worker (4 workers = ~8 GB total)

## Stopping CVAT

```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down
```

## Restarting After PC Reboot

Simply run the start command from Step 3. The deployed function and configurations persist in Docker volumes.

## Additional Configuration

### Adjusting Number of Workers
Edit `function.yaml` or `function-gpu.yaml`:
```yaml
triggers:
  http:
    maxWorkers: 4  # Change this value (2-8 recommended)
```

### Using Different SAM2 Models
Available models:
- `sam2.1_hiera_tiny.pt` (smallest, fastest)
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt` (default, best balance)
- `sam2.1_hiera_large.pt` (largest, most accurate)

Download from: https://github.com/facebookresearch/segment-anything-2/blob/main/README.md

Update `function.yaml`:
```yaml
env:
  - name: SAM2_CHECKPOINT
    value: sam2.1_hiera_<model_size>.pt
```

## Getting Help

- CVAT Documentation: https://docs.cvat.ai/
- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
- Nuclio Documentation: https://nuclio.io/docs/latest/

## License

This integration follows the licenses of:
- CVAT: MIT License
- SAM2: Apache 2.0 License
- Nuclio: Apache 2.0 License