# CVAT-OCT

**A customized fork of [CVAT](https://github.com/cvat-ai/cvat) for interactive, AI-assisted segmentation of retinal Optical Coherence Tomography (OCT) images.**

This repository was developed as part of an MSc dissertation (University of the Witwatersrand). It extends the open-source Computer Vision Annotation Tool (CVAT) with a custom [SAM2](https://github.com/facebookresearch/sam2)-based interactor specialized for OCT structures, a masks-to-polygons conversion action, and one-command, cross-platform setup.

> This is **not** standard CVAT. See [What's different](#whats-different) below and [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the full architecture.

---

## Table of contents

- [What's different](#whats-different)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Model checkpoints](#model-checkpoints)
- [Custom features](#custom-features)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Attribution & license](#attribution--license)
- [Citation](#citation)

## What's different

Compared with upstream CVAT, this fork adds:

1. **SAM2-OCT interactor** — a Nuclio serverless function (`serverless/pytorch/sam2-OCT-interactor/`) that runs a SAM2-based model adapted for multi-class OCT segmentation. Returns editable polygons directly.
2. **Standard SAM2 interactor** — `serverless/pytorch/sam2-interactor/`, auto-deployed on startup.
3. **Masks-to-polygons action** — convert raster mask annotations into editable polygons from the CVAT UI (`cvat-core/src/annotations-actions/masks-to-polygons.ts`).
4. **One-command, cross-platform setup** — `start` / `stop` scripts for PowerShell, CMD, and Bash, with automatic OS detection for the correct `.env` file.
5. **Automated Nuclio initialization** — serverless functions deploy automatically on first start (no manual Nuclio configuration).

## Architecture

CVAT-OCT runs as a set of Docker containers orchestrated by Docker Compose: a React/TypeScript frontend, a Django backend, PostgreSQL/Redis/ClickHouse for storage and analytics, and a Nuclio serverless platform that hosts the SAM2 inference functions. See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for a full diagram and component breakdown.

## Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose
- Git
- ~8 GB+ free RAM; an NVIDIA GPU is recommended for SAM2 inference (CPU works but is slow)
- SAM2 model checkpoint(s) — see [Model checkpoints](#model-checkpoints) (not included in the repo)

## Quick start

Clone the repository:

```bash
git clone https://github.com/enslinr/cvat-oct.git
cd cvat-oct
```

Place the required model checkpoints (see [below](#model-checkpoints)), then start everything with a single command.

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Windows (CMD):**
```cmd
start.bat
```

**Linux / macOS / Git Bash:**
```bash
./start.sh
```

The startup script copies the correct `.env` file for your platform (`.env.windows` or `.env.linux`), brings up the stack, and deploys the SAM2 serverless function automatically.

Once running, open **http://localhost:8080** and log in. To stop:

```bash
./stop.sh        # Linux/macOS/Git Bash
.\stop.ps1       # Windows PowerShell
stop.bat         # Windows CMD
```

For active development (frontend rebuilds and serverless hot-reload), use `start-dev.sh` / `start-dev.bat` instead.

## Model checkpoints

Model weights are **not** committed to this repository (`*.pt` / `*.pth` are git-ignored). You must provide them locally before starting:

| Checkpoint | Location | Source |
|------------|----------|--------|
| Base SAM2 (`sam2.1_hiera_base_plus.pt`) | `serverless/pytorch/sam2-interactor/models/` and `serverless/pytorch/sam2-OCT-interactor/models/` | [facebookresearch/sam2 releases](https://github.com/facebookresearch/sam2/releases) |
| Fine-tuned OCT weights (e.g. `MGU/`, `NR206/`) | `serverless/pytorch/sam2-OCT-interactor/models/` | Trained as part of this project; available on request |

The SAM2-OCT function reads its checkpoint path from the `SAM2_CHECKPOINT` environment variable (see `serverless/pytorch/sam2-OCT-interactor/function.yaml` and `docker-compose.override.yml`). OCT-specific behaviour is configured via `SAM2_NUM_CLASSES` and `SAM2_TARGET_CLASS` in your `.env` file.

## Custom features

### SAM2-OCT interactor
Interactive segmentation tuned for OCT imagery. Users place positive/negative click points; the model returns multi-class masks converted to polygons. See [serverless/pytorch/sam2-OCT-interactor/README.md](serverless/pytorch/sam2-OCT-interactor/README.md) and [SETUP.md](serverless/pytorch/sam2-OCT-interactor/SETUP.md).

### Masks-to-polygons
Converts mask annotations into editable polygon annotations from within the CVAT UI (**Menu → Run actions → "Masks to polygons"**). Implementation notes: [MASKS_TO_POLYGONS_IMPLEMENTATION.md](MASKS_TO_POLYGONS_IMPLEMENTATION.md).

## Repository layout

```
cvat-oct/
├── cvat/                  # Django backend
├── cvat-ui/               # React/TypeScript frontend
├── cvat-core/             # Shared JS logic (incl. masks-to-polygons action)
├── cvat-canvas/ ...       # Canvas, data, SDK, CLI libraries
├── serverless/
│   └── pytorch/
│       ├── sam2-interactor/       # Standard SAM2 function
│       └── sam2-OCT-interactor/   # Custom OCT SAM2 function
├── components/            # Serverless + analytics compose components
├── docker-compose*.yml    # Service definitions and overrides
├── start* / stop*         # Cross-platform launch/stop scripts
├── PROJECT_OVERVIEW.md    # Architecture & development reference
└── README.md              # This file
```

## Documentation

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — architecture, configuration, development workflows, troubleshooting
- [MASKS_TO_POLYGONS_IMPLEMENTATION.md](MASKS_TO_POLYGONS_IMPLEMENTATION.md) — masks-to-polygons feature notes
- [serverless/pytorch/sam2-OCT-interactor/](serverless/pytorch/sam2-OCT-interactor/) — OCT function README & setup
- [CHANGELOG.md](CHANGELOG.md) — upstream CVAT changelog (records the base version this fork is built on)
- [CVAT documentation](https://docs.cvat.ai/) — for general CVAT usage

## Attribution & license

This project is a fork of **[CVAT](https://github.com/cvat-ai/cvat)** by Intel Corporation and CVAT.ai Corporation, used under the MIT License. It also incorporates **[SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/sam2)** by Meta AI.

All original CVAT copyright notices are retained. This fork and its customizations are likewise released under the [MIT License](LICENSE).

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff), and also cite the upstream [CVAT](https://github.com/cvat-ai/cvat) and [SAM2](https://github.com/facebookresearch/sam2) projects.
