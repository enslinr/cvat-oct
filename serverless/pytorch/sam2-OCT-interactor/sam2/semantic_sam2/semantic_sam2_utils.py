# semantic_sam2_utils.py

"""
Utilities for Semantic SAM2, including safe checkpoint loading that handles
different numbers of mask tokens/classes.
"""

import torch
import logging
from pathlib import Path
from typing import Optional
import numpy as np


def load_checkpoint_staged_safe(model, ckpt_path: Optional[str]):
    """
    Load checkpoint for Semantic SAM2, handling:
    1. Different number of mask tokens (classes)
    2. SAM2-specific key names
    3. Memory attention/encoder components
    4. PyTorch 2.6+ compatibility
    
    This is a drop-in replacement for the standard _load_checkpoint function
    in sam2.build_sam that handles semantic segmentation models.
    
    Args:
        model: SAM2Semantic model instance
        ckpt_path: Path to checkpoint file (can be None)
    
    Returns:
        None (modifies model in-place)
    """
    if ckpt_path is None:
        logging.info("No checkpoint path provided, skipping checkpoint loading")
        return
    
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint not found: {ckpt_path}")
        return
    
    logging.info(f"Loading checkpoint from: {ckpt_path}")
    
    # ========== Helper Functions ==========
    
    def _strip_module_prefix(sd):
        """Remove 'module.' prefix from DDP checkpoints."""
        if any(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd
    
    def _safe_torch_load(path):
        """Handle PyTorch 2.6+ default change to weights_only=True."""
        try:
            # Try secure mode first
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only
            return torch.load(path, map_location="cpu")
        except Exception:
            # Checkpoint contains non-tensor objects
            return torch.load(path, map_location="cpu", weights_only=False)
    
    def _get_tensor_from_dict(sd, possible_keys):
        """Try multiple possible key names to find a tensor."""
        for key in possible_keys:
            if key in sd:
                return sd[key]
        return None
    
    def _extract_state_dict(checkpoint):
        """Extract state_dict from various checkpoint formats."""
        if isinstance(checkpoint, dict):
            # Training checkpoint with multiple keys
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                return checkpoint["model"]
            elif "model_state" in checkpoint:
                return checkpoint["model_state"]
            else:
                # Assume the dict itself is the state dict
                return checkpoint
        else:
            # Direct state dict
            return checkpoint
    
    # ========== Semantic Prompt Encoder Adaptation ==========
    
    def _adapt_semantic_prompt_encoder(model, state_dict):
        """
        Adapt prompt encoder(s) when the number of classes differs between 
        checkpoint and model.
        
        Handles both single prompt encoder (sam_prompt_encoder) and multiple 
        prompt encoders (sam_prompt_encoders).
        
        Returns:
            bool: True if any adaptation was performed
        """
        # Try to find class embeddings in checkpoint to determine K_ckpt
        ckpt_class_embeddings = _get_tensor_from_dict(state_dict, [
            "sam_prompt_encoder.class_embeddings.weight",
            "prompt_encoder.class_embeddings.weight",
        ])
        
        K_ckpt = None
        if ckpt_class_embeddings is not None and ckpt_class_embeddings.ndim == 2:
            K_ckpt = int(ckpt_class_embeddings.shape[0])
        
        if K_ckpt is not None:
            logging.info(f"Checkpoint prompt encoder has {K_ckpt} class embeddings")
        else:
            logging.info("Checkpoint prompt encoder class count unknown (may be standard SAM2)")
        
        # Check if we have multiple prompt encoders
        if hasattr(model, "sam_prompt_encoders") and model.sam_prompt_encoders is not None:
            # Handle multiple prompt encoders
            adapted_any = False
            for encoder_name, prompt_encoder in model.sam_prompt_encoders.items():
                K_model = prompt_encoder.num_classes
                
                logging.info(f"Prompt encoder '{encoder_name}' has {K_model} classes")
                
                # Determine if we need to adapt this encoder
                need_adaptation = (K_ckpt is None) or (K_ckpt != K_model)
                
                if need_adaptation:
                    logging.info(f"→ Adapting prompt encoder '{encoder_name}': initializing for {K_model} classes")
                    _adapt_prompt_encoder_classes(prompt_encoder, state_dict, K_model, K_ckpt, encoder_name)
                    adapted_any = True
                else:
                    logging.info(f"ℹ  Prompt encoder '{encoder_name}' classes match checkpoint")
            
            return adapted_any
        else:
            # Single prompt encoder (original behavior)
            if not hasattr(model, "sam_prompt_encoder") or model.sam_prompt_encoder is None:
                return False
                
            prompt_encoder = model.sam_prompt_encoder
            K_model = prompt_encoder.num_classes
            
            logging.info(f"Model prompt encoder has {K_model} classes")
            
            # Determine if we need to adapt
            need_adaptation = (K_ckpt is None) or (K_ckpt != K_model)
            
            if need_adaptation:
                logging.info(f"→ Adapting prompt encoder: initializing for {K_model} classes")
                _adapt_prompt_encoder_classes(prompt_encoder, state_dict, K_model, K_ckpt)
                return True
            
            return False
    
    def _adapt_prompt_encoder_classes(prompt_encoder, state_dict, K_model, K_ckpt, encoder_name=None):
        """Adapt prompt encoder for different number of classes."""
        device = prompt_encoder.class_embeddings.weight.device
        
        # 1. Adapt class embeddings
        _adapt_class_embeddings(prompt_encoder, state_dict, K_model, K_ckpt, device, encoder_name)
        
        # 2. Adapt semantic_mask_downscaling (first conv layer takes num_classes channels)
        _adapt_semantic_mask_downscaling(prompt_encoder, state_dict, K_model, K_ckpt, device, encoder_name)
    
    def _adapt_class_embeddings(prompt_encoder, state_dict, K_model, K_ckpt, device, encoder_name=None):
        """Adapt class embeddings for different number of classes."""
        # Build list of possible checkpoint keys
        possible_keys = [
            "sam_prompt_encoder.class_embeddings.weight",
            "prompt_encoder.class_embeddings.weight",
        ]
        if encoder_name:
            possible_keys.insert(0, f"sam_prompt_encoders.{encoder_name}.class_embeddings.weight")
        
        ckpt_embeddings = _get_tensor_from_dict(state_dict, possible_keys)
        
        with torch.no_grad():
            if ckpt_embeddings is not None and ckpt_embeddings.ndim == 2:
                # Copy what we can from checkpoint
                embed_dim = prompt_encoder.class_embeddings.weight.shape[1]
                n_copy = min(K_ckpt or 0, K_model)
                
                if n_copy > 0 and ckpt_embeddings.shape[1] == embed_dim:
                    prompt_encoder.class_embeddings.weight.data[:n_copy].copy_(
                        ckpt_embeddings[:n_copy].to(device)
                    )
                    # Initialize remaining classes by repeating the last loaded embedding
                    if n_copy < K_model:
                        last_embed = ckpt_embeddings[n_copy - 1:n_copy].to(device)
                        for i in range(n_copy, K_model):
                            prompt_encoder.class_embeddings.weight.data[i].copy_(last_embed[0])
                else:
                    # Just initialize with small random values (default PyTorch init is fine)
                    pass
            # If no checkpoint embeddings, keep the randomly initialized ones
        
        name_str = f" ({encoder_name})" if encoder_name else ""
        logging.info(f"✓ Adapted class embeddings{name_str} to {K_model} classes")
    
    def _adapt_semantic_mask_downscaling(prompt_encoder, state_dict, K_model, K_ckpt, device, encoder_name=None):
        """Adapt semantic_mask_downscaling conv layer for different number of input classes."""
        # The first conv in semantic_mask_downscaling takes num_classes input channels
        # We need to adapt this if K_model != K_ckpt
        
        if not hasattr(prompt_encoder, "semantic_mask_downscaling"):
            return
        
        # Build list of possible checkpoint keys for the first conv weight
        possible_weight_keys = [
            "sam_prompt_encoder.semantic_mask_downscaling.0.weight",
            "prompt_encoder.semantic_mask_downscaling.0.weight",
        ]
        if encoder_name:
            possible_weight_keys.insert(0, f"sam_prompt_encoders.{encoder_name}.semantic_mask_downscaling.0.weight")
        
        ckpt_weight = _get_tensor_from_dict(state_dict, possible_weight_keys)
        
        first_conv = prompt_encoder.semantic_mask_downscaling[0]
        
        with torch.no_grad():
            if ckpt_weight is not None and ckpt_weight.ndim == 4:
                # ckpt_weight shape: [out_channels, in_channels(=K_ckpt), kH, kW]
                out_c, in_c_ckpt, kH, kW = ckpt_weight.shape
                _, in_c_model, _, _ = first_conv.weight.shape
                
                if in_c_model == in_c_ckpt:
                    # Same number of classes, direct copy
                    first_conv.weight.data.copy_(ckpt_weight.to(device))
                else:
                    # Different number of classes, need to adapt
                    n_copy = min(in_c_ckpt, in_c_model)
                    
                    # Copy weights for classes that exist in both
                    first_conv.weight.data[:, :n_copy].copy_(
                        ckpt_weight[:, :n_copy].to(device)
                    )
                    
                    # For additional classes, initialize by copying the last channel's weights
                    if n_copy < in_c_model:
                        last_channel = ckpt_weight[:, n_copy - 1:n_copy].to(device)
                        for i in range(n_copy, in_c_model):
                            first_conv.weight.data[:, i:i+1].copy_(last_channel)
            # If no checkpoint weights, keep the randomly initialized ones
        
        name_str = f" ({encoder_name})" if encoder_name else ""
        logging.info(f"✓ Adapted semantic_mask_downscaling{name_str} for {K_model} input classes")
    
    # ========== Semantic Decoder Adaptation ==========
    
    def _adapt_semantic_decoder(model, state_dict):
        """
        Adapt decoder(s) when the number of classes differs between 
        checkpoint and model. Clones head-0 to all class heads.
        
        Handles both single decoder (sam_mask_decoder) and multiple 
        decoders (sam_mask_decoders).
        
        Returns:
            bool: True if any adaptation was performed
        """
        # Try to find mask tokens in checkpoint to determine K_ckpt
        ckpt_mask_tokens = _get_tensor_from_dict(state_dict, [
            "sam_mask_decoder.mask_tokens.weight",
            "mask_decoder.mask_tokens.weight",
        ])
        
        K_ckpt = None
        if ckpt_mask_tokens is not None and ckpt_mask_tokens.ndim == 2:
            K_ckpt = int(ckpt_mask_tokens.shape[0])
        
        logging.info(f"Checkpoint has {K_ckpt} class tokens")
        
        # Check if we have multiple decoders
        if hasattr(model, "sam_mask_decoders") and model.sam_mask_decoders is not None:
            # Handle multiple decoders
            adapted_any = False
            for decoder_name, decoder in model.sam_mask_decoders.items():
                device = decoder.mask_tokens.weight.device
                K_model = decoder.num_class_tokens
                
                logging.info(f"Decoder '{decoder_name}' has {K_model} class tokens")
                
                # Determine if we need to adapt this decoder
                need_adaptation = (K_ckpt is None) or (K_ckpt != K_model)
                
                if need_adaptation:
                    logging.info(f"→ Adapting decoder '{decoder_name}': cloning head-0 to {K_model} classes")
                    _clone_head_to_all_classes(decoder, state_dict, K_model, device, decoder_name)
                    adapted_any = True
                else:
                    logging.info(f"ℹ  Decoder '{decoder_name}' classes match checkpoint")
            
            return adapted_any
        else:
            # Single decoder (original behavior)
            decoder = model.sam_mask_decoder
            device = decoder.mask_tokens.weight.device
            K_model = decoder.num_class_tokens
            
            logging.info(f"Model has {K_model} class tokens")
            
            # Determine if we need to adapt
            need_adaptation = (K_ckpt is None) or (K_ckpt != K_model)
            
            if need_adaptation:
                logging.info(f"→ Adapting decoder: cloning head-0 to {K_model} classes")
                _clone_head_to_all_classes(decoder, state_dict, K_model, device)
                return True
            
            return False
    
    def _clone_head_to_all_classes(decoder, state_dict, K_model, device, decoder_name=None):
        """Clone head-0 weights to all class heads."""
        
        # 1. Adapt Mask Tokens
        _adapt_mask_tokens(decoder, state_dict, K_model, device, decoder_name)
        
        # 2. Adapt Hypernetwork MLPs
        _adapt_hypernetwork_mlps(decoder, state_dict, K_model, decoder_name)
        
        # 3. Adapt IoU Prediction Head
        _adapt_iou_head(decoder, state_dict, K_model, device, decoder_name)
    
    def _adapt_mask_tokens(decoder, state_dict, K_model, device, decoder_name=None):
        """Adapt mask token embeddings."""
        # Build list of possible checkpoint keys
        possible_keys = [
            "sam_mask_decoder.mask_tokens.weight",
            "mask_decoder.mask_tokens.weight",
        ]
        # Also try decoder-specific keys if we have a decoder name
        if decoder_name:
            possible_keys.insert(0, f"sam_mask_decoders.{decoder_name}.mask_tokens.weight")
        
        ckpt_tokens = _get_tensor_from_dict(state_dict, possible_keys)
        
        with torch.no_grad():
            if ckpt_tokens is not None and ckpt_tokens.ndim == 2 and ckpt_tokens.shape[0] >= 1:
                # Use first token from checkpoint
                proto = ckpt_tokens[0].to(device)
            else:
                # Use current model's first token
                proto = decoder.mask_tokens.weight.data[0].detach().clone()
            
            # Expand to all classes
            decoder.mask_tokens.weight.data.copy_(
                proto.unsqueeze(0).expand(K_model, -1)
            )
        
        name_str = f" ({decoder_name})" if decoder_name else ""
        logging.info(f"✓ Adapted mask tokens{name_str} to {K_model} classes")
    
    def _adapt_hypernetwork_mlps(decoder, state_dict, K_model, decoder_name=None):
        """Adapt hypernetwork MLPs for all classes."""
        
        # Try to load head-0 from checkpoint
        base_prefixes = [
            "sam_mask_decoder.output_hypernetworks_mlps.0",
            "mask_decoder.output_hypernetworks_mlps.0",
        ]
        # Also try decoder-specific prefix if we have a decoder name
        if decoder_name:
            base_prefixes.insert(0, f"sam_mask_decoders.{decoder_name}.output_hypernetworks_mlps.0")
        
        head0_state = {}
        for prefix in base_prefixes:
            for key, param in state_dict.items():
                if key.startswith(prefix):
                    # Extract layer key: "...mlps.0.layers.0.weight" -> "layers.0.weight"
                    layer_key = key.split(f"{prefix}.")[-1]
                    if layer_key not in head0_state:  # Don't overwrite if already found
                        head0_state[layer_key] = param
        
        # Load head-0 if found
        if head0_state:
            try:
                decoder.output_hypernetworks_mlps[0].load_state_dict(
                    head0_state, strict=False
                )
                name_str = f" ({decoder_name})" if decoder_name else ""
                logging.info(f"✓ Loaded hypernetwork head-0{name_str} from checkpoint")
            except Exception as e:
                logging.warning(f"Could not load hypernetwork head-0: {e}")
        
        # Clone head-0 to all other heads
        with torch.no_grad():
            head0_state_dict = decoder.output_hypernetworks_mlps[0].state_dict()
            for i in range(1, K_model):
                decoder.output_hypernetworks_mlps[i].load_state_dict(head0_state_dict)
        
        name_str = f" ({decoder_name})" if decoder_name else ""
        logging.info(f"✓ Cloned hypernetwork head-0{name_str} to {K_model} heads")
    
    def _adapt_iou_head(decoder, state_dict, K_model, device, decoder_name=None):
        """Adapt IoU prediction head for all classes."""
        
        # Get the final linear layer
        last_layer = decoder.iou_prediction_head.layers[-1]
        
        # Build list of possible checkpoint keys for weight
        weight_keys = [
            "sam_mask_decoder.iou_prediction_head.layers.2.weight",
            "mask_decoder.iou_prediction_head.layers.2.weight",
            "sam_mask_decoder.iou_prediction_head.layers.3.weight",  # depth=4
            "mask_decoder.iou_prediction_head.layers.3.weight",
        ]
        # Also try decoder-specific keys if we have a decoder name
        if decoder_name:
            weight_keys.insert(0, f"sam_mask_decoders.{decoder_name}.iou_prediction_head.layers.2.weight")
            weight_keys.insert(1, f"sam_mask_decoders.{decoder_name}.iou_prediction_head.layers.3.weight")
        
        # Build list of possible checkpoint keys for bias
        bias_keys = [
            "sam_mask_decoder.iou_prediction_head.layers.2.bias",
            "mask_decoder.iou_prediction_head.layers.2.bias",
            "sam_mask_decoder.iou_prediction_head.layers.3.bias",
            "mask_decoder.iou_prediction_head.layers.3.bias",
        ]
        if decoder_name:
            bias_keys.insert(0, f"sam_mask_decoders.{decoder_name}.iou_prediction_head.layers.2.bias")
            bias_keys.insert(1, f"sam_mask_decoders.{decoder_name}.iou_prediction_head.layers.3.bias")
        
        ckpt_weight = _get_tensor_from_dict(state_dict, weight_keys)
        ckpt_bias = _get_tensor_from_dict(state_dict, bias_keys)
        
        with torch.no_grad():
            # Get row-0 from checkpoint or current model
            if (ckpt_weight is not None and ckpt_weight.ndim == 2 and
                ckpt_weight.shape[0] >= 1 and
                ckpt_weight.shape[1] == last_layer.weight.shape[1]):
                w0 = ckpt_weight[0:1].to(device)
                b0 = (ckpt_bias[0:1].to(device) if ckpt_bias is not None
                      else last_layer.bias.data[0:1].clone())
            else:
                w0 = last_layer.weight.data[0:1].detach().clone()
                b0 = last_layer.bias.data[0:1].detach().clone()

            # Replace last layer if its output dim doesn't match K_model
            if last_layer.weight.shape[0] != K_model:
                in_features = last_layer.weight.shape[1]
                has_bias = last_layer.bias is not None
                new_layer = torch.nn.Linear(in_features, K_model, bias=has_bias).to(device)
                layer_idx = len(decoder.iou_prediction_head.layers) - 1
                decoder.iou_prediction_head.layers[layer_idx] = new_layer
                last_layer = new_layer

            # Expand to all classes
            last_layer.weight.data.copy_(w0.expand(K_model, -1))
            if last_layer.bias is not None:
                last_layer.bias.data.copy_(b0.expand(K_model))
        
        name_str = f" ({decoder_name})" if decoder_name else ""
        logging.info(f"✓ Adapted IoU head{name_str} to {K_model} classes")
    
    # ========== Main Loading Logic ==========
    
    try:
        # 1. Load checkpoint file
        checkpoint = _safe_torch_load(str(checkpoint_path))
        logging.info(f"✓ Loaded checkpoint file")
    except Exception as e:
        logging.error(f"✗ Failed to load checkpoint: {e}")
        return
    
    # 2. Extract state dict
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    
    # 3. Filter to matching shapes
    model_sd = model.state_dict()
    filtered = {}
    mismatched = []
    broadcast_count = 0
    
    # Check if model has multiple decoders (sam_mask_decoders)
    # In this case, sam_mask_decoder is just an alias to the first decoder
    has_multi_decoders = hasattr(model, "sam_mask_decoders") and model.sam_mask_decoders is not None
    has_multi_prompt_encoders = hasattr(model, "sam_prompt_encoders") and model.sam_prompt_encoders is not None
    
    prompt_encoder_broadcast_count = 0
    
    for k, v in state_dict.items():
        # Handle sam_prompt_encoder.* keys specially when we have multi-prompt-encoders
        # Skip direct matching - only allow broadcast to sam_prompt_encoders.*
        if k.startswith("sam_prompt_encoder.") and not k.startswith("sam_prompt_encoders."):
            if has_multi_prompt_encoders:
                # Don't do direct match - fall through to broadcast logic below
                pass
            elif k in model_sd:
                # Single prompt encoder mode - do direct match
                if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                    filtered[k] = v
                else:
                    mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))
                continue  # Don't fall through to broadcast
            
            # Broadcast sam_prompt_encoder -> sam_prompt_encoders.* mapping
            suffix = k[len("sam_prompt_encoder."):]
            for model_key in model_sd:
                if model_key.startswith("sam_prompt_encoders."):
                    # Extract encoder name and check if suffix matches
                    # model_key format: sam_prompt_encoders.{encoder_name}.{suffix}
                    parts = model_key.split(".", 2)  # ['sam_prompt_encoders', 'encoder_name', 'rest']
                    if len(parts) >= 3 and parts[2] == suffix:
                        if hasattr(v, "shape") and v.shape == model_sd[model_key].shape:
                            filtered[model_key] = v
                            prompt_encoder_broadcast_count += 1
                        else:
                            mismatched.append((f"{k} -> {model_key}", 
                                             tuple(getattr(v, "shape", ())), 
                                             tuple(model_sd[model_key].shape)))
        
        # Handle sam_mask_decoder.* keys specially when we have multi-decoders
        # Skip direct matching - only allow broadcast to sam_mask_decoders.*
        elif k.startswith("sam_mask_decoder.") and not k.startswith("sam_mask_decoders."):
            if has_multi_decoders:
                # Don't do direct match - fall through to broadcast logic below
                pass
            elif k in model_sd:
                # Single decoder mode - do direct match
                if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                    filtered[k] = v
                else:
                    mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))
                continue  # Don't fall through to broadcast
            
            # Broadcast sam_mask_decoder -> sam_mask_decoders.* mapping
            suffix = k[len("sam_mask_decoder."):]
            for model_key in model_sd:
                if model_key.startswith("sam_mask_decoders."):
                    # Extract decoder name and check if suffix matches
                    # model_key format: sam_mask_decoders.{decoder_name}.{suffix}
                    parts = model_key.split(".", 2)  # ['sam_mask_decoders', 'decoder_name', 'rest']
                    if len(parts) >= 3 and parts[2] == suffix:
                        if hasattr(v, "shape") and v.shape == model_sd[model_key].shape:
                            filtered[model_key] = v
                            broadcast_count += 1
                        else:
                            mismatched.append((f"{k} -> {model_key}", 
                                             tuple(getattr(v, "shape", ())), 
                                             tuple(model_sd[model_key].shape)))
        elif k in model_sd:
            # Direct match for non-decoder keys
            if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))
        
        # Handle sam_prompt_encoders.* keys from checkpoint (if any) - direct match only
        elif k.startswith("sam_prompt_encoders.") and k in model_sd:
            if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))
        
        # Handle sam_mask_decoders.* keys from checkpoint (if any) - direct match only
        elif k.startswith("sam_mask_decoders.") and k in model_sd:
            if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))
    
    if prompt_encoder_broadcast_count > 0:
        logging.info(f"ℹ  Broadcast {prompt_encoder_broadcast_count} sam_prompt_encoder weights to sam_prompt_encoders")
    
    if broadcast_count > 0:
        logging.info(f"ℹ  Broadcast {broadcast_count} sam_mask_decoder weights to sam_mask_decoders")
    
    if has_multi_prompt_encoders:
        logging.info(f"ℹ  Multi-prompt-encoder mode: skipping direct sam_prompt_encoder.* loads (using broadcast only)")
    
    if has_multi_decoders:
        logging.info(f"ℹ  Multi-decoder mode: skipping direct sam_mask_decoder.* loads (using broadcast only)")
    
    if mismatched:
        logging.info(f"ℹ  Skipping {len(mismatched)} keys with shape mismatch (showing up to 5):")
        for k, s_ckpt, s_model in mismatched[:5]:
            logging.info(f"   - {k}: ckpt{s_ckpt} vs model{s_model}")
    
    # 4. Load with non-strict mode
    try:
        result = model.load_state_dict(filtered, strict=False)
        
        # Filter out batch norm tracking keys from report
        missing = [k for k in result.missing_keys if "num_batches_tracked" not in k]
        unexpected = [k for k in result.unexpected_keys if "num_batches_tracked" not in k]
        
        if missing or unexpected:
            logging.info(f"ℹ  load_state_dict(strict=False):")
            if missing:
                logging.info(f"   missing keys: {len(missing)}")
                if len(missing) <= 10:
                    for k in missing:
                        logging.info(f"      - {k}")
            if unexpected:
                logging.info(f"   unexpected keys: {len(unexpected)}")
                if len(unexpected) <= 10:
                    for k in unexpected:
                        logging.info(f"      - {k}")
        else:
            logging.info("✓ Loaded checkpoint with strict=False (all keys matched)")
    
    except Exception as e:
        logging.warning(f"⚠  Full model load failed: {e}")
        logging.info("→ Attempting component-wise loading...")
        
        # Try loading components separately
        components = [
            ("image_encoder", model.image_encoder),
            ("sam_prompt_encoder", model.sam_prompt_encoder),
            ("sam_mask_decoder", model.sam_mask_decoder),
            ("memory_attention", model.memory_attention),
            ("memory_encoder", model.memory_encoder),
        ]
        
        for name, module in components:
            try:
                comp_state = {
                    k.replace(f"{name}.", "", 1): v
                    for k, v in filtered.items()
                    if k.startswith(f"{name}.")
                }
                if comp_state:
                    module.load_state_dict(comp_state, strict=False)
                    logging.info(f"✓ Loaded {name}")
                else:
                    logging.info(f"⚠  No {name} weights in checkpoint")
            except Exception as ce:
                logging.warning(f"⚠  Failed to load {name}: {ce}")
    
    # 5. Adapt prompt encoder for semantic segmentation
    try:
        prompt_encoder_adapted = _adapt_semantic_prompt_encoder(model, state_dict)
        if prompt_encoder_adapted:
            logging.info("✓ Prompt encoder adapted for semantic segmentation")
        else:
            logging.info("ℹ  No prompt encoder adaptation needed (classes match)")
    except Exception as e:
        logging.warning(f"⚠  Prompt encoder adaptation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Adapt decoder for semantic segmentation
    try:
        was_adapted = _adapt_semantic_decoder(model, state_dict)
        if was_adapted:
            logging.info("✓ Decoder adapted for semantic segmentation")
        else:
            logging.info("ℹ  No decoder adaptation needed (classes match)")
    except Exception as e:
        logging.warning(f"⚠  Decoder adaptation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("✓ Checkpoint loading complete")


def get_semantic_sam2_configs():
    """
    Return a dictionary of available semantic SAM2 configurations.
    Useful for documentation and validation.
    """
    return {
        "hiera_tiny": {
            "config": "sam2.1_hiera_t_semantic",
            "base_config": "sam2.1_hiera_t",
            "embed_dim": 96,
            "num_heads": 1,
        },
        "hiera_small": {
            "config": "sam2.1_hiera_s_semantic",
            "base_config": "sam2.1_hiera_s",
            "embed_dim": 96,
            "num_heads": 1,
        },
        "hiera_base_plus": {
            "config": "sam2.1_hiera_b+_semantic",
            "base_config": "sam2.1_hiera_b+",
            "embed_dim": 112,
            "num_heads": 2,
        },
        "hiera_large": {
            "config": "sam2.1_hiera_l_semantic",
            "base_config": "sam2.1_hiera_l",
            "embed_dim": 144,
            "num_heads": 2,
        },
    }


def print_model_info(model):
    """Print useful information about a semantic SAM2 model."""
    print("=" * 60)
    print("Semantic SAM2 Model Information")
    print("=" * 60)
    
    # Basic info
    print(f"Model type: {type(model).__name__}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Image size: {model.image_size}")
    print(f"Device: {model.device}")
    
    # Decoder info
    decoder = model.sam_mask_decoder
    print(f"\nMask Decoder:")
    print(f"  Num class tokens: {decoder.num_class_tokens}")
    print(f"  Num mask tokens: {decoder.num_mask_tokens}")
    print(f"  Transformer dim: {decoder.transformer_dim}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    print("=" * 60)


def verify_semantic_output(masks, num_classes):
    """
    Verify that the output from semantic SAM2 has the expected shape.
    
    Args:
        masks: Output masks from predictor
        num_classes: Expected number of classes
    
    Returns:
        bool: True if valid, False otherwise
    """
    if masks.shape[0] != num_classes:
        logging.error(
            f"Expected {num_classes} classes, got {masks.shape[0]} masks"
        )
        return False
    
    logging.info(f"✓ Output shape verified: {masks.shape}")
    return True
