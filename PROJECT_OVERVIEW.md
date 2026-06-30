# CVAT-OCT — Project Overview

> **Purpose**: This document provides a comprehensive overview of the project structure, architecture, and development patterns. It complements the top-level [README](README.md) with deeper architectural detail.

## Project Identity

**Name**: CVAT-OCT (Computer Vision Annotation Tool, customized for OCT)  
**Base Project**: [CVAT by cvat.ai](https://github.com/cvat-ai/cvat)  
**Type**: Computer Vision Annotation Platform with Custom SAM2 Integration  
**License**: MIT

## What Makes This Different

This is **NOT** standard CVAT. Key customizations:

1. ✅ **Automated One-Command Setup** - No manual Nuclio configuration
2. ✅ **SAM2 Semantic Segmentation** - Auto-deploys on startup
3. ✅ **SAM2-OCT Custom Interactor** - Specialized for OCT (Optical Coherence Tomography) images
4. ✅ **Masks-to-Polygons Converter** - Convert mask annotations to editable polygons
5. ✅ **Cross-Platform Scripts** - PowerShell, Batch, and Bash scripts with OS detection
6. ✅ **Development Workflow Scripts** - Hot-reload for serverless functions

## Architecture Overview

### Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React, TypeScript, Redux, Canvas API |
| **Backend** | Django (Python), Django REST Framework |
| **Database** | PostgreSQL, Redis (in-memory & on-disk), ClickHouse (analytics) |
| **Serverless** | Nuclio (serverless functions platform) |
| **AI Models** | SAM2 (Segment Anything Model 2), PyTorch |
| **Container** | Docker, Docker Compose |
| **Reverse Proxy** | Traefik |
| **Monitoring** | Grafana, Vector |

### Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Traefik :8080                        │
│                    (Reverse Proxy & Load Balancer)           │
└────────────────┬──────────────────────┬─────────────────────┘
                 │                      │
        ┌────────▼──────────┐  ┌───────▼──────────┐
        │   cvat_ui :80     │  │ cvat_server :8080│
        │   (React/nginx)   │  │  (Django/uWSGI)  │
        └───────────────────┘  └────────┬─────────┘
                                        │
        ┌───────────────────────────────┼──────────────────┐
        │                               │                  │
   ┌────▼─────┐  ┌────────────┐  ┌────▼────┐  ┌─────────▼────────┐
   │ cvat_db  │  │cvat_redis_ │  │  cvat_  │  │  cvat_worker_*   │
   │(Postgres)│  │inmem/disk  │  │clickhouse│  │ (import/export/  │
   └──────────┘  └────────────┘  └─────────┘  │ annotation/etc)  │
                                               └──────────────────┘
        ┌──────────────────────────────────────────────┐
        │              Nuclio :8070                    │
        │         (Serverless Platform)                │
        │  ┌──────────────────────────────────────┐   │
        │  │  SAM2-OCT Function (PyTorch + SAM2)  │   │
        │  │  - Exposed via HTTP API              │   │
        │  │  - Auto-deployed by nuclio_init      │   │
        │  └──────────────────────────────────────┘   │
        └──────────────────────────────────────────────┘
                          ▲
                          │
                  ┌───────┴──────────┐
                  │   nuclio_init    │
                  │ (Init Container) │
                  │ - Runs once      │
                  │ - Deploys SAM2   │
                  └──────────────────┘
```

## Directory Structure

### Root Level
```
cvat-oct/
├── cvat/                         # Django backend application
│   ├── apps/                     # Django apps (engine, dataset_manager, etc)
│   ├── settings/                 # Django settings
│   └── requirements/             # Python dependencies
│
├── cvat-ui/                      # React frontend application
│   └── src/                      # TypeScript/React source
│
├── cvat-core/                    # Core JS library (shared logic)
│   └── src/
│       └── annotations-actions/  # ⚡ Custom: Masks-to-polygons converter
│
├── cvat-canvas/                  # Canvas rendering library
├── cvat-canvas3d/                # 3D annotation canvas
├── cvat-data/                    # Data loading/processing
├── cvat-sdk/                     # Python SDK
├── cvat-cli/                     # Command-line interface
│
├── serverless/                   # ⚡ Serverless functions
│   ├── init-nuclio.sh            # ⚡ Auto-initialization script
│   └── pytorch/
│       ├── sam2-interactor/      # Standard SAM2 implementation
│       └── sam2-OCT-interactor/  # ⚡ Custom OCT-specialized SAM2
│           ├── main.py           # Nuclio handler
│           ├── model_handler.py  # SAM2 model logic
│           ├── function.yaml     # Nuclio function config
│           └── models/           # Model checkpoints directory
│
├── components/
│   ├── serverless/
│   │   └── docker-compose.serverless.yml  # Nuclio service
│   └── analytics/                # Grafana dashboards
│
├── docker-compose.yml            # Base services
├── docker-compose.override.yml   # ⚡ SAM2 config + auto-init
├── docker-compose.dev.yml        # Development overrides
├── docker-compose.dev-local.yml  # Local dev overrides
│
├── .env.windows                  # ⚡ Windows-specific config (semicolons)
├── .env.linux                    # ⚡ Linux/Mac config (colons)
│
├── start.sh / start.bat / start.ps1      # ⚡ Startup scripts
├── start-dev.sh / start-dev.bat          # ⚡ Dev workflow scripts
├── stop.sh / stop.bat / stop.ps1         # ⚡ Shutdown scripts
│
├── README.md                     # Main entry point: features, setup, usage
├── MASKS_TO_POLYGONS_IMPLEMENTATION.md  # ⚡ Masks-to-polygons feature notes
└── PROJECT_OVERVIEW.md           # ⚡ This file (architecture reference)
```

**Legend**: ⚡ = Custom/modified from standard CVAT

## Configuration System

### Environment Files

The project uses OS-specific `.env` files to handle Docker Compose path separator differences:

- **`.env.windows`**: Uses semicolons (`;`) for `COMPOSE_FILE` paths
  - Required because Windows uses colons for drive letters (e.g., `C:`)
  - Used by: PowerShell, CMD, Git Bash on Windows
  
- **`.env.linux`**: Uses colons (`:`) for `COMPOSE_FILE` paths
  - Standard Unix/Linux path separator
  - Used by: Linux, macOS

**Example**:
```bash
# .env.linux
COMPOSE_FILE=docker-compose.yml:docker-compose.dev.yml:docker-compose.override.yml

# .env.windows
COMPOSE_FILE=docker-compose.yml;docker-compose.dev.yml;docker-compose.override.yml
```

### Startup Scripts with OS Detection

All startup scripts now include OS detection to automatically select the correct `.env` file:

```bash
# Detects Windows environment in Git Bash
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$MSYSTEM" ]]; then
    echo "Detected Windows environment (Git Bash)"
    cp .env.windows .env
else
    echo "Detected Linux/Mac environment"
    cp .env.linux .env
fi
```

**Detection Variables**:
- `$OSTYPE == "msys"`: Git Bash on Windows
- `$OSTYPE == "win32"`: Alternative Windows detection
- `$MSYSTEM`: Git Bash environment variable (MINGW32/MINGW64)

## Development Workflows

### Standard Startup

**Windows PowerShell/CMD**:
```powershell
.\start.bat        # or .\start.ps1
```

**Linux/Mac/Git Bash**:
```bash
./start.sh
```

**Manual**:
```bash
docker compose up -d
```

### Development Mode with Hot Reload

**Windows**:
```cmd
start-dev.bat
```

**Linux/Mac/Git Bash**:
```bash
./start-dev.sh
```

**What it does**:
1. Detects OS and copies appropriate `.env` file
2. Rebuilds `cvat_ui` with `--no-cache` (applies frontend changes)
3. Starts all services
4. Updates SAM2-OCT function:
   - **Hot Update**: If container exists, copies code and restarts
   - **Full Deploy**: If new, deploys function via `nuctl`

### Stopping Services

```bash
./stop.sh         # Linux/Mac/Git Bash
.\stop.bat        # Windows CMD
.\stop.ps1        # Windows PowerShell
```

## Custom Implementations

### 1. SAM2-OCT Interactor

**Location**: `serverless/pytorch/sam2-OCT-interactor/`

**Purpose**: Specialized SAM2 model for OCT (Optical Coherence Tomography) medical image segmentation

**Key Features**:
- Multi-class segmentation (configurable via `SAM2_NUM_CLASSES`)
- Target class selection (via `SAM2_TARGET_CLASS`)
- Point-based interaction (user clicks to guide segmentation)
- Polygon output (converts masks to polygons automatically)

**Configuration** (in `docker-compose.override.yml`):
```yaml
environment:
  - SAM2_CONFIG=sam2.1/sam2.1_hiera_b+.yaml
  - SAM2_CHECKPOINT=/opt/nuclio/models/sam2.1_hiera_base_plus.pt
  - SAM2_NUM_CLASSES=3        # Number of output channels
  - SAM2_TARGET_CLASS=1       # Which channel to use for CVAT
```

**API Contract**:
```python
# Input
{
    "image": "base64_encoded_image",
    "pos_points": [[x1, y1], [x2, y2]],  # Positive points (foreground)
    "neg_points": [[x3, y3]]              # Negative points (background)
}

# Output
{
    "polygons": [
        {
            "points": [x1, y1, x2, y2, ...],
            "type": "polygon"
        }
    ]
}
```

### 2. Masks-to-Polygons Converter

**Location**: `cvat-core/src/annotations-actions/masks-to-polygons.ts`

**Purpose**: Convert mask annotations to editable polygon annotations

**Usage**:
- Menu → Run actions → "Masks to polygons"
- Right-click mask → Run action → "Masks to polygons"

**Parameters**:
- `simplification`: Tolerance for polygon simplification (0.5-10.0)

### 3. Automated Nuclio Initialization

**Location**: `serverless/init-nuclio.sh`

**Purpose**: Automatically deploys SAM2 functions on startup (no manual setup required)

**Process**:
1. Wait for Nuclio dashboard to be healthy
2. Install `nuctl` CLI tool
3. Create `cvat` project
4. Deploy SAM2-OCT function
5. Verify deployment success

**Container**: `nuclio_init` (runs once, then sleeps)

## Common Development Patterns

### Pattern 1: Cross-Platform Script Fixes

**Example**: Recent fix for `start.sh` and `start-dev.sh`

**Problem**: Git Bash on Windows needs `.env.windows` (semicolons) but script was using `.env.linux` (colons)

**Solution**:
```bash
# Add OS detection before copying .env
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$MSYSTEM" ]]; then
    cp .env.windows .env
else
    cp .env.linux .env
fi
```

**Pattern**: Always consider OS differences when working with:
- Path separators
- Line endings (CRLF vs LF)
- Shell syntax
- Docker volume paths

### Pattern 2: Serverless Function Development

**Workflow**:
1. Edit code in `serverless/pytorch/sam2-OCT-interactor/`
2. Run `start-dev.sh` or `start-dev.bat`
3. Script automatically detects running container and does:
   - **Hot update**: Copy files → Restart container
   - **Full deploy**: Use `nuctl deploy`

**Key Files**:
- `main.py`: Nuclio handler (receives HTTP requests)
- `model_handler.py`: Model logic (SAM2 inference)
- `function.yaml`: Nuclio configuration

**Testing**:
```bash
# Check function status
docker exec nuclio nuctl get function sam2-oct --project-name cvat

# View function logs
docker logs nuclio-nuclio-sam2-oct -f
```

### Pattern 3: Frontend Development

**Workflow**:
1. Edit TypeScript/React in `cvat-ui/src/`
2. Rebuild UI:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache cvat_ui
   docker compose up -d --force-recreate cvat_ui
   ```
3. Hard refresh browser (Ctrl+Shift+R)

**Key Directories**:
- `cvat-ui/src/components/`: React components
- `cvat-ui/src/actions/`: Redux actions
- `cvat-ui/src/reducers/`: Redux reducers
- `cvat-core/src/`: Shared business logic

## Troubleshooting Knowledge Base

### Issue: `nuclio_init` container not found

**Cause**: Wrong `.env` file used (path separator mismatch)

**Solution**: Ensure correct `.env` file for your OS/shell
- Windows (Git Bash): Use `.env.windows`
- Linux/Mac: Use `.env.linux`

**Prevention**: Use startup scripts with OS detection

### Issue: SAM2 function not deploying

**Debugging Steps**:
```bash
# 1. Check nuclio_init logs
docker logs nuclio_init -f

# 2. Check Nuclio health
docker exec nuclio wget -q -O- http://localhost:8070/api/healthcheck

# 3. Manually trigger init
docker exec nuclio_init sh -c "/init-nuclio.sh"

# 4. Check function status
docker exec nuclio nuctl get function --project-name cvat
```

### Issue: Frontend changes not appearing

**Cause**: Browser caching or UI container not rebuilt

**Solution**:
1. Rebuild UI container with `--no-cache`
2. Force recreate container
3. Hard refresh browser (Ctrl+Shift+R)

## Monitoring and Logs

### Key Log Commands

```bash
# Initialization progress
docker logs nuclio_init -f

# Backend server
docker logs cvat_server -f

# Nuclio platform
docker logs nuclio -f

# SAM2-OCT function
docker logs nuclio-nuclio-sam2-oct -f

# All running containers
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Health Checks

```bash
# CVAT backend
curl http://localhost:8080/api/server/health

# Nuclio dashboard
curl http://localhost:8070/api/healthcheck

# Function status
docker exec nuclio nuctl get function sam2-oct --project-name cvat
```

## Important Notes for Future Tasks

1. **Always consider OS compatibility**: This project runs on Windows, Linux, and Mac
   - Path separators differ (`;` vs `:`)
   - Line endings differ (CRLF vs LF)
   - Shell syntax differs (PowerShell/Batch/Bash)

2. **Multiple Docker Compose files**: Always specify all required files
   ```bash
   docker compose -f docker-compose.yml \
                  -f docker-compose.override.yml \
                  -f docker-compose.dev.yml \
                  -f components/serverless/docker-compose.serverless.yml
   ```
   Or use the `.env` file's `COMPOSE_FILE` variable

3. **Development vs Production**: 
   - `start.sh` / `start.bat`: Production mode
   - `start-dev.sh` / `start-dev.bat`: Development mode (rebuilds, hot reload)

4. **Container dependencies**:
   - `nuclio_init` depends on `nuclio` being healthy
   - `cvat_server` depends on database and Redis
   - Always check `depends_on` in compose files

5. **Custom code locations**:
   - Serverless functions: `serverless/pytorch/sam2-OCT-interactor/`
   - Frontend actions: `cvat-core/src/annotations-actions/`
   - Backend: `cvat/apps/`

## Quick Reference

### Start Development Session
```bash
./start-dev.sh     # Linux/Mac/Git Bash
start-dev.bat      # Windows
```

### Check Everything is Running
```bash
docker ps
docker logs nuclio_init -f
```

### Access Points
- CVAT UI: http://localhost:8080
- Nuclio Dashboard: http://localhost:8070 (if enabled)
- Grafana: http://localhost:8080/analytics

### Emergency Reset
```bash
docker compose down -v  # ⚠️ DESTROYS ALL DATA
docker compose up -d
```

---

**Base CVAT version**: see [CHANGELOG.md](CHANGELOG.md)  
**Purpose**: Architecture reference and onboarding for development tasks
