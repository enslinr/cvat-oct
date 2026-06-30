# SAM2 Semantic Interactor

This serverless package embeds the Semantic SAM2 codebase so CVAT can call it
as an interactor. Positive/negative point prompts are converted into semantic
masks – you select which class channel is returned to CVAT via environment
variables.

## Layout

```
sam2-interactor/
├── configs/…                # Hydra config copied from sam2 repo
├── models/…                 # Checkpoint (`sam2.1_hiera_base_plus.pt`)
├── sam2/…                   # Upstream SAM2 package
├── semantic_sam2/…         # Semantic decoders + utils
├── function.yaml           # CPU build
├── function-gpu.yaml       # CUDA build (optional)
├── main.py                 # Nuclio handler
├── model_handler.py        # SAM2 inference wrapper
└── requirements.txt        # Python dependencies
```

## Deploying with Nuclio

1. Mount this directory and `models/` into the Nuclio container (already done
   in `docker-compose.override.yml`). Ensure the checkpoint filename matches
   the `SAM2_CHECKPOINT` environment variable (`sam2.1_hiera_base_plus.pt`).

2. Install `nuctl` in the Nuclio container (once):

   ```sh
   wget -O /usr/local/bin/nuctl \
     https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
   chmod +x /usr/local/bin/nuctl
   ```

3. Deploy the function (CPU example):

   ```sh
   nuctl deploy sam2-semantic \
     --project-name cvat \
     --platform local \
     --path /opt/nuclio/app/sam2-interactor \
     --file /opt/nuclio/app/sam2-interactor/function.yaml
   ```

   Use `function-gpu.yaml` if you need CUDA.

4. Refresh the CVAT UI. The “SAM2 Semantic” interactor will appear under
   **AI Tools → Interactors**.

## Environment variables

| Variable              | Purpose                                            | Default |
|-----------------------|----------------------------------------------------|---------|
| `SAM2_CONFIG`         | Hydra config path (inside the container)           | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| `SAM2_CHECKPOINT`     | Checkpoint filename under `/opt/nuclio/models`     | `sam2_semantic.pt` |
| `SAM2_NUM_CLASSES`    | Total semantic channels in the checkpoint          | `2`     |
| `SAM2_TARGET_CLASS`   | Channel returned to CVAT                           | `num_classes - 1` |
| `SAM2_MASK_THRESHOLD` | Threshold applied before contour extraction        | `0.5`   |
| `SAM2_SAFE_LOAD`      | Use staged checkpoint loading (handles mismatches) | `1`     |
| `SAM2_DEVICE`         | Force `"cpu"` or `"cuda:0"`                        | auto    |

Adjust these via the Nuclio environment if your model differs.
