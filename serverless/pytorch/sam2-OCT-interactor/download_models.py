#!/usr/bin/env python3
"""Download the fine-tuned SAM2-OCT checkpoints from Hugging Face.

Usage:
    pip install huggingface_hub
    python download_models.py                 # all fine-tuned checkpoints
    python download_models.py MGU NR206       # only the named ones

Weights are placed under ./models/ (git-ignored). The base SAM2 checkpoint
(sam2.1_hiera_base_plus.pt) is NOT hosted on Hugging Face here — download it from
Meta's releases if you need the vanilla SAM2 interactor:
https://github.com/facebookresearch/sam2/releases

Model card and license (CC BY-NC 4.0): https://huggingface.co/enslinr/sam2-oct
"""
import os
import sys

REPO_ID = "enslinr/sam2-oct"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# key -> file path within the Hugging Face repo
CHECKPOINTS = {
    "MGU": "MGU/final_runs_Glaucoma_last.pt",
    "MGU_prompted": "MGU_prompted/MGU_prompt_training_last.pt",
    "NR206": "NR206/final_runs_NR206_last.pt",
}


def main(argv):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

    requested = argv or list(CHECKPOINTS)
    unknown = [k for k in requested if k not in CHECKPOINTS]
    if unknown:
        sys.exit(f"Unknown checkpoint(s): {unknown}. Choose from: {list(CHECKPOINTS)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    for key in requested:
        filename = CHECKPOINTS[key]
        print(f"Downloading {filename} ...")
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODELS_DIR)
        print(f"  -> {path}")

    print(
        "\nDone. Set SAM2_CHECKPOINT to the checkpoint you want to serve.\n"
        "Note: NR206 uses fewer classes than MGU — see the README "
        "('Selecting a checkpoint') before switching."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
