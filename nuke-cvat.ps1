<#
.NUKES CVAT + DOCKER DESKTOP STATE

This script will:
  - Remove ALL Docker containers
  - Remove ALL Docker images
  - Remove ALL Docker volumes
  - Prune Docker system & builder caches
  - Delete Docker Desktop bind-mount folders
  - (Optionally, but enabled by default) unregister `docker-desktop` and `docker-desktop-data` WSL distros
  - Clean CVAT project caches (node_modules, dist, __pycache__, etc.)

EDIT THE $CvatRoot PATH BELOW TO MATCH YOUR CVAT PROJECT ROOT.
#>

param(
    # Defaults to the directory this script lives in (the repo root).
    [string]$CvatRoot = $PSScriptRoot
)

Write-Host "====================================================" -ForegroundColor Yellow
Write-Host "        CVAT / DOCKER DESKTOP NUCLEAR RESET" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will DELETE:" -ForegroundColor Red
Write-Host "  - ALL Docker containers (for ALL projects)"
Write-Host "  - ALL Docker images"
Write-Host "  - ALL Docker volumes"
Write-Host "  - Docker build cache, networks, etc."
Write-Host "  - CVAT data volumes (tasks, DB, logs, etc.)"
Write-Host "  - Docker Desktop bind-mount artifacts"
Write-Host "  - (Later) docker-desktop and docker-desktop-data WSL distros"
Write-Host ""
Write-Host "CVAT project root: $CvatRoot"
Write-Host ""

$confirm = Read-Host "Type 'NUKE' to proceed, or anything else to abort"
if ($confirm -ne "NUKE") {
    Write-Host "Aborting. Nothing was changed." -ForegroundColor Cyan
    exit 0
}

# -----------------------------
# PHASE 1: Docker cleanup
# -----------------------------
Write-Host ""
Write-Host "PHASE 1: Removing ALL containers..." -ForegroundColor Yellow
try {
    $containers = docker ps -aq 2>$null
    if ($containers) {
        $containers | ForEach-Object { docker rm -f $_ | Out-Null }
        Write-Host "  - Removed containers: $($containers.Count)"
    } else {
        Write-Host "  - No containers found."
    }
} catch {
    Write-Host "  ! Error while removing containers: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "PHASE 2: Removing ALL images..." -ForegroundColor Yellow
try {
    $images = docker images -aq 2>$null
    if ($images) {
        $images | ForEach-Object { docker rmi -f $_ | Out-Null }
        Write-Host "  - Removed images: $($images.Count)"
    } else {
        Write-Host "  - No images found."
    }
} catch {
    Write-Host "  ! Error while removing images: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "PHASE 3: Removing ALL Docker volumes..." -ForegroundColor Yellow
try {
    $vols = docker volume ls -q 2>$null
    if ($vols) {
        $vols | ForEach-Object { docker volume rm $_ | Out-Null }
        Write-Host "  - Removed volumes: $($vols.Count)"
    } else {
        Write-Host "  - No volumes found."
    }
} catch {
    Write-Host "  ! Error while removing volumes: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "PHASE 4: Pruning Docker system & builder cache..." -ForegroundColor Yellow
try {
    docker system prune -a --volumes -f | Out-Null
    docker builder prune -a -f | Out-Null
    Write-Host "  - Docker system & builder pruned."
} catch {
    Write-Host "  ! Error while pruning Docker: $_" -ForegroundColor Red
}

# -----------------------------
# PHASE 5: Clean CVAT project caches (Windows side)
# -----------------------------
Write-Host ""
Write-Host "PHASE 5: Cleaning CVAT project caches at $CvatRoot ..." -ForegroundColor Yellow

if (-not (Test-Path $CvatRoot)) {
    Write-Host "  - CVAT root not found, skipping project cleanup." -ForegroundColor DarkYellow
} else {
    try {
        # Common UI / Node artifacts
        $uiPaths = @(
            (Join-Path $CvatRoot "cvat-ui\node_modules"),
            (Join-Path $CvatRoot "cvat-ui\dist"),
            (Join-Path $CvatRoot "cvat-ui\build")
        )

        foreach ($p in $uiPaths) {
            if (Test-Path $p) {
                Write-Host "  - Removing $p ..."
                Remove-Item -Recurse -Force $p
            }
        }

        # Remove __pycache__ and .pyc files
        Write-Host "  - Removing __pycache__ and *.pyc under $CvatRoot ..."
        Get-ChildItem -Path $CvatRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | `
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        Get-ChildItem -Path $CvatRoot -Recurse -Include "*.pyc" -File -ErrorAction SilentlyContinue | `
            Remove-Item -Force -ErrorAction SilentlyContinue

        Write-Host "  - CVAT project cleanup done."
    } catch {
        Write-Host "  ! Error while cleaning CVAT project: $_" -ForegroundColor Red
    }
}

# -----------------------------
# PHASE 6: Stop Docker Desktop & WSL, clean bind mounts, unregister distros
# -----------------------------
Write-Host ""
Write-Host "PHASE 6: Nuclear WSL / Docker Desktop data reset" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will:" -ForegroundColor Red
Write-Host "  - Stop Docker Desktop app (if running)"
Write-Host "  - Run 'wsl --shutdown'"
Write-Host "  - Delete docker-desktop bind-mount folders"
Write-Host "  - Unregister 'docker-desktop' and 'docker-desktop-data' WSL distros"
Write-Host ""
$confirm2 = Read-Host "Type 'YES' to proceed with WSL unregister (THIS CANNOT BE UNDONE easily), anything else to skip this phase"
if ($confirm2 -ne "YES") {
    Write-Host "Skipping WSL / Docker Desktop distro unregister. Docker-side cleanup is complete." -ForegroundColor Cyan
    exit 0
}

Write-Host ""
Write-Host "Stopping Docker Desktop (if running)..." -ForegroundColor Yellow
try {
    $dockerProcs = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
    if ($dockerProcs) {
        $dockerProcs | Stop-Process -Force
        Write-Host "  - Docker Desktop stopped."
    } else {
        Write-Host "  - Docker Desktop was not running."
    }
} catch {
    Write-Host "  ! Error stopping Docker Desktop: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Running 'wsl --shutdown'..." -ForegroundColor Yellow
try {
    wsl --shutdown
    Write-Host "  - WSL shutdown requested."
} catch {
    Write-Host "  ! Error running 'wsl --shutdown': $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Deleting docker-desktop bind-mount folders..." -ForegroundColor Yellow
# This path is accessible only while docker-desktop is registered,
# but may not exist on all systems.
$bindMountRoot = "\\wsl.localhost\docker-desktop\mnt\host\wsl\docker-desktop-bind-mounts"

try {
    if (Test-Path $bindMountRoot) {
        Write-Host "  - Removing contents of $bindMountRoot ..."
        Get-ChildItem -Path $bindMountRoot -Force -ErrorAction SilentlyContinue | `
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  - Bind-mount artifacts removed."
    } else {
        Write-Host "  - Bind-mount root not found (may already be gone)."
    }
} catch {
    Write-Host "  ! Error deleting bind-mount folders: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Unregistering docker-desktop WSL distros..." -ForegroundColor Yellow
try {
    wsl --unregister docker-desktop 2>$null
    Write-Host "  - Unregistered 'docker-desktop' (or it was already absent)."
} catch {
    Write-Host "  ! Error unregistering 'docker-desktop': $_" -ForegroundColor Red
}

try {
    wsl --unregister docker-desktop-data 2>$null
    Write-Host "  - Unregistered 'docker-desktop-data' (or it was already absent)."
} catch {
    Write-Host "  ! Error unregistering 'docker-desktop-data': $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host " NUKE COMPLETE." -ForegroundColor Green
Write-Host " - Docker objects removed (containers, images, volumes, cache)"
Write-Host " - CVAT project caches cleaned (where path existed)"
Write-Host " - Docker Desktop WSL distros removed"
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start Docker Desktop again (it will recreate its WSL distros)."
Write-Host "  2. In a new terminal: cd $CvatRoot"
Write-Host "  3. Rebuild CVAT: docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml build --no-cache cvat_ui"
Write-Host "  4. Start CVAT:   docker compose -f docker-compose.yml -f docker-compose.override.yml -f components/serverless/docker-compose.serverless.yml up -d"
Write-Host "  5. Create superuser: docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'"
Write-Host ""
