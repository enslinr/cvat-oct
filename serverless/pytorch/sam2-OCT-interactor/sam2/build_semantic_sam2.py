# build_semantic_sam2.py

"""
Semantic SAM2 builder that uses the standard SAM2 infrastructure
with an optional custom checkpoint loader for handling architecture mismatches.
"""

import logging
from typing import Optional, List

from sam2.build_sam import build_sam2
from sam2.semantic_sam2.semantic_sam2_components import DCESemanticMaskDecoder, SemanticMaskDecoder
from sam2.semantic_sam2.semantic_sam2_utils import load_checkpoint_staged_safe


def build_semantic_sam2(
    config_file: str,
    ckpt_path: Optional[str] = None,
    device: str = "cuda",
    mode: str = "eval",
    hydra_overrides_extra: Optional[List[str]] = None,
    apply_postprocessing: bool = True,
    use_load_checkpoint_staged_safe: bool = False,
    **kwargs,
):
    """
    Build a Semantic SAM2 model with optional safe checkpoint loading.
    
    This is a wrapper around sam2.build_sam.build_sam2 that adds the ability
    to use a custom checkpoint loader that handles architecture mismatches
    (e.g., different numbers of mask tokens/classes).
    
    Args:
        config_file: Path to config file (e.g., "sam2.1_hiera_l_semantic")
        ckpt_path: Path to checkpoint file (can be standard SAM2 checkpoint)
        device: Device to load model on ("cuda" or "cpu")
        mode: Model mode - "eval" or "train"
        hydra_overrides_extra: Optional list of Hydra config overrides
        apply_postprocessing: Whether to apply dynamic multimask postprocessing
        use_load_checkpoint_staged_safe: Whether to use safe checkpoint loading.
            Set to True when:
            - Loading a standard SAM2 checkpoint into a semantic model with
              different number of classes
            - The checkpoint has different architecture (e.g., 4 mask tokens
              but your model has 9 class tokens)
            Set to False when:
            - Loading a semantic checkpoint that matches your model architecture
            - Loading a previously saved checkpoint from training
        **kwargs: Additional arguments passed to build_sam2
    
    Returns:
        Loaded SAM2Semantic model
    
    Example:
        # Loading standard SAM2 checkpoint into semantic model (different # classes)
        model = build_semantic_sam2(
            config_file="sam2.1_hiera_l_semantic",
            ckpt_path="sam2.1_hiera_large.pt",
            use_load_checkpoint_staged_safe=True  # Use safe loader
        )
        
        # Loading a saved semantic checkpoint (matching architecture)
        model = build_semantic_sam2(
            config_file="sam2.1_hiera_l_semantic",
            ckpt_path="my_trained_semantic_model.pt",
            use_load_checkpoint_staged_safe=False  # Use standard loader
        )
    """
    # Handle mutable default argument
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []
    
    # Override the model target to be SAM2Semantic
    # This ensures that we instantiate the correct class which accepts our custom arguments
    hydra_overrides_extra.append("++model._target_=sam2.semantic_sam2.semantic_sam2_components.SAM2Semantic")
    
    # Determine checkpoint path for build_sam2
    # If using safe loader, we'll load checkpoint manually after model creation
    ckpt_path_for_build = None if use_load_checkpoint_staged_safe else ckpt_path

    # Build model using standard SAM2 infrastructure
    model = build_sam2(
        config_file=config_file,
        ckpt_path=ckpt_path_for_build,
        device=device,
        mode=mode,
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        **kwargs,
    )
    
    # If using safe loader, load checkpoint with architecture adaptation
    if use_load_checkpoint_staged_safe:
        if ckpt_path is None:
            logging.warning(
                "use_load_checkpoint_staged_safe=True but ckpt_path is None. "
                "Model will be initialized with random weights."
            )
        else:
            try:
                logging.info(f"Loading checkpoint with safe loader from: {ckpt_path}")
                load_checkpoint_staged_safe(model, ckpt_path)
                
                # Ensure model is on correct device after loading
                # (safe loader loads to CPU first, then we move to device)
                model = model.to(device)
                
                # Re-apply mode setting after checkpoint loading
                if mode == "eval":
                    model.eval()
                elif mode == "train":
                    model.train()
                
                logging.info("✓ Safe checkpoint loading completed successfully")
                
            except Exception as e:
                logging.error(f"Failed to load checkpoint with safe loader: {e}")
                logging.error(
                    "Model was created but checkpoint loading failed. "
                    "The model has randomly initialized weights."
                )
                raise RuntimeError(
                    f"Safe checkpoint loading failed: {e}\n"
                    "This might happen if:\n"
                    "  1. The checkpoint file is corrupted\n"
                    "  2. The checkpoint format is incompatible\n"
                    "  3. The file path is incorrect\n"
                    "Check the error message above for details."
                ) from e
    
    return model