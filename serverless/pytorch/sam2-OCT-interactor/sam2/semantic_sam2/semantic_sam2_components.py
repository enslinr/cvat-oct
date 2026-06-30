# semantic_sam2_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from importlib import import_module

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class SemanticMaskDecoder(MaskDecoder):
    """
    Extended mask decoder for semantic segmentation with multiple class outputs.
    Similar to your SAM1 MaskDecoderSemantic but adapted for SAM2.
    """
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_class_tokens: int = 11,  # Number of semantic classes
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        hypernet_output_dim: Optional[int] = None
    ) -> None:
        """
        Initialize semantic mask decoder.
        
        Args:
            num_class_tokens: Number of semantic classes (K)
            Other args are same as base MaskDecoder
        """
        # Initialize parent with num_multimask_outputs = num_class_tokens - 1
        # because parent expects num_multimask_outputs, but we'll override the tokens
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=3,  # Will be overridden
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            dynamic_multimask_via_stability=dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=dynamic_multimask_stability_thresh,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
        )
        
        self.num_class_tokens = num_class_tokens
        self.num_mask_tokens = num_class_tokens
        
        # Replace mask tokens with semantic class tokens
        self.mask_tokens = nn.Embedding(num_class_tokens, transformer_dim)


# Increasing dims
        self.hypernet_output_dim = (
            transformer_dim // 8 if hypernet_output_dim is None else hypernet_output_dim
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4,
                self.hypernet_output_dim,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )
            # project widened decoder features down to 32-ch before adding feat_s0,
            # then lift them back up for the hyper-networks
            self.upscale_residual_proj = nn.Conv2d(
                self.hypernet_output_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.upscale_post_add_proj = nn.Conv2d(
                transformer_dim // 8, self.hypernet_output_dim, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, self.hypernet_output_dim, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        
        # Update IoU head to predict IoU for each class (override parent's 4-output head)
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            num_class_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        sam_tokens_out = mask_tokens_out

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

            # dc1, ln1, act1, dc2, act2 = self.output_upscaling
            # feat_s0, feat_s1 = high_res_features
            # upscaled_embedding = dc1(src)
            # upscaled_embedding = ln1(upscaled_embedding + feat_s1)
            # upscaled_embedding = act1(upscaled_embedding)
            # upscaled_embedding = dc2(upscaled_embedding)
            # fusion = self.upscale_residual_proj(upscaled_embedding) + feat_s0
            # fusion = act2(fusion)
            # upscaled_embedding = self.upscale_post_add_proj(fusion)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits






class ClassTaggedPromptEncoder(PromptEncoder):
    """
    Extended prompt encoder that can handle class-specific prompts.
    Similar to your SAM1 version.
    """
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        num_classes: int = 11,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            mask_in_chans=mask_in_chans,
            activation=activation,
        )
        self.num_classes = num_classes
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        self._dense_prompt_eps = 1e-3
        self.semantic_mask_downscaling = nn.Sequential(
            nn.Conv2d(num_classes, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def labels_to_dense_logits(
        self,
        label_masks: torch.Tensor,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convert integer semantic labels into per-class logit maps sized for the dense prompt pipeline.

        Args:
            label_masks: Tensor of shape (B,H,W), (B,1,H,W), or (B,C,H,W) containing class indices.
            ignore_index: Optional index to treat as void; corresponding logits default to zero.

        Returns:
            Tensor of shape (B,num_classes,H',W') with logit values for each semantic class.
        """
        if label_masks is None:
            raise ValueError("label_masks must be provided for semantic dense prompting.")

        if label_masks.dim() == 4 and label_masks.size(1) == 1:
            label_masks = label_masks[:, 0]
        elif label_masks.dim() == 4 and label_masks.size(1) == self.num_classes:
            logits = label_masks.to(self.class_embeddings.weight.device, dtype=torch.float32)
            if logits.shape[-2:] != self.mask_input_size:
                logits = F.interpolate(
                    logits,
                    size=self.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            return logits

        if label_masks.dim() != 3:
            raise ValueError(
                f"Expected label masks with shape (B,H,W) or (B,1,H,W); got {list(label_masks.shape)}."
            )

        device = self.class_embeddings.weight.device
        labels = label_masks.to(device=device, dtype=torch.long)

        invalid = labels < 0
        if ignore_index is not None:
            invalid = invalid | (labels == ignore_index)
        invalid = invalid | (labels >= self.num_classes)

        clamped = labels.clamp(min=0, max=self.num_classes - 1)
        one_hot = F.one_hot(clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        if invalid.any():
            one_hot = one_hot.masked_fill(invalid.unsqueeze(1), 0.0)

        eps = self._dense_prompt_eps
        probs = one_hot * (1.0 - 2.0 * eps) + eps
        logits = torch.log(probs / (1.0 - probs))
        logits.masked_fill_(invalid.unsqueeze(1), 0.0)

        if logits.shape[-2:] != self.mask_input_size:
            logits = F.interpolate(
                logits,
                size=self.mask_input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return logits

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        if masks is None:
            return super()._embed_masks(masks)

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        if masks.dtype not in (torch.float16, torch.float32, torch.float64):
            masks = self.labels_to_dense_logits(masks, ignore_index=None)
        elif masks.dim() == 4 and masks.size(1) == 1 and self.num_classes > 1:
            # If a single-channel dense prompt is provided, replicate conversion to semantic logits.
            masks = self.labels_to_dense_logits(masks[:, 0], ignore_index=None)
        elif masks.dim() == 4 and masks.size(1) != self.num_classes and masks.size(1) != 1:
            raise ValueError(
                f"Unsupported dense prompt channel count {masks.size(1)}; expected 1 or {self.num_classes}."
            )

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        if masks.size(1) == self.num_classes:
            logits = masks.to(dtype=torch.float32)
            if logits.shape[-2:] != self.mask_input_size:
                logits = F.interpolate(
                    logits,
                    size=self.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            mask_embedding = self.semantic_mask_downscaling(logits)
            class_presence = torch.sigmoid(logits).mean(dim=(2, 3), keepdim=True)
            class_bias = torch.matmul(
                class_presence.squeeze(-1).squeeze(-1),
                self.class_embeddings.weight,
            )
            mask_embedding = mask_embedding + class_bias.unsqueeze(-1).unsqueeze(-1)
            return mask_embedding

        # Fallback to default single-channel handling.
        if masks.size(1) != 1:
            raise ValueError(
                f"Unexpected dense prompt shape {masks.shape}; unable to encode."
            )
        return super()._embed_masks(masks)


class SAM2Semantic(SAM2Base):
    """
    Semantic SAM2 that outputs per-class masks instead of generic object masks.
    Extends SAM2Base to use semantic decoder.
    """
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_classes: int = 11,
        mask_decoder_cls: Union[str, Type[MaskDecoder]] = SemanticMaskDecoder,
        mask_decoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_names: Optional[List[str]] = None,
        decoder_class_counts: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        # Store num_classes before calling super
        self.num_classes = num_classes
        self.decoder_names = decoder_names
        self.decoder_class_counts = decoder_class_counts
        self.mask_decoder_cls = self._resolve_mask_decoder_cls(mask_decoder_cls)
        self.mask_decoder_kwargs = dict(mask_decoder_kwargs or {})  

        # Call parent init (this will call _build_sam_heads)
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            **kwargs
        )

    @staticmethod
    def _resolve_mask_decoder_cls(mask_decoder_cls):
        if isinstance(mask_decoder_cls, str):
            module_path, _, attr = mask_decoder_cls.rpartition(".")
            if not module_path:
                raise ValueError("mask_decoder_cls must include a module path")
            return getattr(import_module(module_path), attr)
        return mask_decoder_cls

    def _build_sam_heads(self):
        """Override to build semantic decoder instead of standard decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        
        # Build semantic mask decoder
        from sam2.modeling.sam.transformer import TwoWayTransformer
        

        decoder_kwargs: Dict[str, Any] = {
            "num_class_tokens": self.num_classes,
            "transformer": TwoWayTransformer(
                 depth=2,
                 embedding_dim=self.sam_prompt_embed_dim,
                 mlp_dim=2048,
                 num_heads=8,
            ),
            "transformer_dim": self.sam_prompt_embed_dim,
            "iou_head_depth": 3,
            "iou_head_hidden_dim": 256,
            "use_high_res_features": self.use_high_res_features_in_sam,
            "iou_prediction_use_sigmoid": self.iou_prediction_use_sigmoid,
            "pred_obj_scores": self.pred_obj_scores,
            "pred_obj_scores_mlp": self.pred_obj_scores_mlp,
            "use_multimask_token_for_obj_ptr": self.use_multimask_token_for_obj_ptr,
        }
        decoder_kwargs.update(self.mask_decoder_kwargs)
        if self.sam_mask_decoder_extra_args:
            decoder_kwargs.update(self.sam_mask_decoder_extra_args)

        decoder_cls = self.mask_decoder_cls
        
        if self.decoder_names:
            # Build separate prompt encoders and mask decoders for each dataset
            self.sam_prompt_encoders = nn.ModuleDict()
            self.sam_mask_decoders = nn.ModuleDict()
            
            for name in self.decoder_names:
                # Get class count for this decoder
                num_classes_for_decoder = self.num_classes
                if self.decoder_class_counts and name in self.decoder_class_counts:
                    num_classes_for_decoder = self.decoder_class_counts[name]
                
                # Build prompt encoder for this decoder with its specific class count
                self.sam_prompt_encoders[name] = ClassTaggedPromptEncoder(
                    embed_dim=self.sam_prompt_embed_dim,
                    image_embedding_size=(
                        self.sam_image_embedding_size,
                        self.sam_image_embedding_size,
                    ),
                    input_image_size=(self.image_size, self.image_size),
                    mask_in_chans=16,
                    num_classes=num_classes_for_decoder,
                )
                
                # Build mask decoder for this decoder
                current_decoder_kwargs = decoder_kwargs.copy()
                # Need to create a new transformer for each decoder
                current_decoder_kwargs["transformer"] = TwoWayTransformer(
                    depth=2,
                    embedding_dim=self.sam_prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                )
                current_decoder_kwargs["num_class_tokens"] = num_classes_for_decoder
                
                self.sam_mask_decoders[name] = decoder_cls(**current_decoder_kwargs)
            
            # Set the default prompt encoder and decoder to the first one for backward compatibility
            self.sam_prompt_encoder = self.sam_prompt_encoders[self.decoder_names[0]]
            self.sam_mask_decoder = self.sam_mask_decoders[self.decoder_names[0]]
        else:
            # Single decoder case - build single prompt encoder
            self.sam_prompt_encoder = ClassTaggedPromptEncoder(
                embed_dim=self.sam_prompt_embed_dim,
                image_embedding_size=(
                    self.sam_image_embedding_size,
                    self.sam_image_embedding_size,
                ),
                input_image_size=(self.image_size, self.image_size),
                mask_in_chans=16,
                num_classes=self.num_classes,
            )
            self.sam_mask_decoder = decoder_cls(**decoder_kwargs)




# Previous mask decoder creation
        # self.sam_mask_decoder = SemanticMaskDecoder(
        #     num_class_tokens=self.num_classes,
        #     transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=self.sam_prompt_embed_dim,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     transformer_dim=self.sam_prompt_embed_dim,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        #     use_high_res_features=self.use_high_res_features_in_sam,
        #     iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
        #     pred_obj_scores=self.pred_obj_scores,
        #     pred_obj_scores_mlp=self.pred_obj_scores_mlp,
        #     use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
        #     **(self.sam_mask_decoder_extra_args or {}),
        # )
# Previous mask decoder creation end
        
        # Object pointer projection
        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
            
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def build_semantic_dense_prompt(
        self,
        label_masks: torch.Tensor,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Public helper to convert label masks into semantic dense prompt logits.
        Ensures the output is on the model device and matches the prompt encoder expectations.
        """
        logits = self.sam_prompt_encoder.labels_to_dense_logits(
            label_masks,
            ignore_index=ignore_index,
        )
        return logits.to(self.device)

    def _prepare_semantic_mask_input(
        self,
        mask_inputs: torch.Tensor,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        if mask_inputs.dim() == 3:
            mask_inputs = mask_inputs.unsqueeze(1)

        if mask_inputs.dtype not in (torch.float16, torch.float32, torch.float64):
            mask_inputs = self.sam_prompt_encoder.labels_to_dense_logits(
                mask_inputs,
                ignore_index=ignore_index,
            )

        if mask_inputs.dim() == 3:
            mask_inputs = mask_inputs.unsqueeze(1)

        if mask_inputs.dim() != 4:
            raise ValueError(f"Semantic mask prompt must be 4D; got shape {list(mask_inputs.shape)}.")

        if mask_inputs.size(0) == 0:
            return mask_inputs

        if mask_inputs.size(1) not in (1, self.num_classes):
            raise ValueError(
                f"Semantic dense prompt must have 1 or {self.num_classes} channels; got {mask_inputs.size(1)}."
            )

        if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
            mask_inputs = F.interpolate(
                mask_inputs.float(),
                size=self.sam_prompt_encoder.mask_input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        return mask_inputs.float()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        ignore_index: Optional[int] = None,
        decoder_ids: Optional[List[str]] = None,
    ):
        """
        Override to support multi-channel semantic dense prompts.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if decoder_ids is not None and hasattr(self, "sam_mask_decoders") and hasattr(self, "sam_prompt_encoders"):
            # Handle multiple decoders with their own prompt encoders
            batch_size = backbone_features.size(0)
            assert len(decoder_ids) == batch_size, "decoder_ids must match batch size"
            
            # Group indices by decoder name
            from collections import defaultdict
            indices_by_decoder = defaultdict(list)
            for idx, name in enumerate(decoder_ids):
                indices_by_decoder[name].append(idx)
            
            # Determine max classes for padding
            max_classes_in_batch = self.num_classes
            if self.decoder_class_counts:
                 max_classes_in_batch = max(self.decoder_class_counts.get(name, self.num_classes) for name in decoder_ids)

            # These will be populated
            all_low_res_multimasks = [None] * batch_size
            all_ious = [None] * batch_size
            all_sam_output_tokens = [None] * batch_size
            all_object_score_logits = [None] * batch_size
            
            for name, indices in indices_by_decoder.items():
                if name not in self.sam_mask_decoders:
                    raise ValueError(f"Unknown decoder name: {name}")
                if name not in self.sam_prompt_encoders:
                    raise ValueError(f"Unknown prompt encoder name: {name}")
                
                decoder = self.sam_mask_decoders[name]
                prompt_encoder = self.sam_prompt_encoders[name]
                indices_tensor = torch.tensor(indices, device=device)
                
                # Slice inputs for this decoder group
                sub_backbone_features = backbone_features[indices_tensor]
                sub_point_coords = sam_point_coords[indices_tensor]
                sub_point_labels = sam_point_labels[indices_tensor]
                sub_high_res_features = None
                if high_res_features is not None:
                    sub_high_res_features = [f[indices_tensor] for f in high_res_features]
                
                # Prepare mask inputs for this decoder's prompt encoder if provided
                sub_semantic_mask_prompt = None
                if mask_inputs is not None:
                    sub_mask_inputs = mask_inputs[indices_tensor]
                    if sub_mask_inputs.dim() == 3:
                        sub_mask_inputs = sub_mask_inputs.unsqueeze(1)
                    if sub_mask_inputs.dtype not in (torch.float16, torch.float32, torch.float64):
                        sub_semantic_mask_prompt = prompt_encoder.labels_to_dense_logits(
                            sub_mask_inputs,
                            ignore_index=ignore_index,
                        )
                    else:
                        sub_semantic_mask_prompt = sub_mask_inputs
                    
                    # Resize if needed
                    if sub_semantic_mask_prompt.shape[-2:] != prompt_encoder.mask_input_size:
                        sub_semantic_mask_prompt = F.interpolate(
                            sub_semantic_mask_prompt.float(),
                            size=prompt_encoder.mask_input_size,
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                    sub_semantic_mask_prompt = sub_semantic_mask_prompt.float()
                
                # Use this decoder's prompt encoder
                sub_sparse_embeddings, sub_dense_embeddings = prompt_encoder(
                    points=(sub_point_coords, sub_point_labels),
                    boxes=None,
                    masks=sub_semantic_mask_prompt,
                )
                sub_sparse_embeddings = sub_sparse_embeddings.clone()
                sub_dense_embeddings = sub_dense_embeddings.clone()
                sub_image_pe = prompt_encoder.get_dense_pe().clone()
                
                # Forward pass for this group
                (
                    sub_low_res_multimasks,
                    sub_ious,
                    sub_sam_output_tokens,
                    sub_object_score_logits,
                ) = decoder(
                    image_embeddings=sub_backbone_features,
                    image_pe=sub_image_pe,
                    sparse_prompt_embeddings=sub_sparse_embeddings,
                    dense_prompt_embeddings=sub_dense_embeddings,
                    multimask_output=multimask_output,
                    repeat_image=False,
                    high_res_features=sub_high_res_features,
                )
                
                # Scatter results back
                for i, original_idx in enumerate(indices):
                    # Pad mask if needed
                    mask = sub_low_res_multimasks[i]
                    if mask.shape[0] < max_classes_in_batch:
                        # Pad with very small value (logits) so they are ignored/background
                        # or just zeros? Logits -> sigmoid. 
                        # If we pad with -inf, sigmoid -> 0.
                        padding = torch.full((max_classes_in_batch - mask.shape[0], mask.shape[1], mask.shape[2]), -100.0, device=device)
                        mask = torch.cat([mask, padding], dim=0)
                    
                    all_low_res_multimasks[original_idx] = mask
                    
                    # Pad IoU if needed
                    iou = sub_ious[i]
                    if iou.shape[0] < max_classes_in_batch:
                        padding_iou = torch.zeros((max_classes_in_batch - iou.shape[0],), device=device)
                        iou = torch.cat([iou, padding_iou], dim=0)
                    all_ious[original_idx] = iou
                    
                    # Pad sam_output_tokens if needed
                    tokens = sub_sam_output_tokens[i]
                    if tokens.shape[0] < max_classes_in_batch:
                         # Pad tokens. Tokens are [num_classes, embed_dim]
                         padding_tokens = torch.zeros((max_classes_in_batch - tokens.shape[0], tokens.shape[1]), device=device)
                         tokens = torch.cat([tokens, padding_tokens], dim=0)
                    all_sam_output_tokens[original_idx] = tokens
                    
                    all_object_score_logits[original_idx] = sub_object_score_logits[i]
            
            # Stack results
            low_res_multimasks = torch.stack(all_low_res_multimasks)
            ious = torch.stack(all_ious)
            sam_output_tokens = torch.stack(all_sam_output_tokens)
            object_score_logits = torch.stack(all_object_score_logits)
            
        else:
            # Single decoder path - use the default prompt encoder
            if mask_inputs is not None:
                semantic_mask_prompt = self._prepare_semantic_mask_input(
                    mask_inputs,
                    ignore_index=ignore_index,
                )
            else:
                semantic_mask_prompt = None

            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels),
                boxes=None,
                masks=semantic_mask_prompt,
            )
            sparse_embeddings = sparse_embeddings.clone()
            dense_embeddings = dense_embeddings.clone()
            image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
            
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,
                high_res_features=high_res_features,
            )
        low_res_multimasks = low_res_multimasks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

###################################################################################################################################################################

# Other Implementations

###################################################################################################################################################################

# SAM-DCE (SAM-DCE: ADDRESSING TOKEN UNIFORMITY AND SEMANTIC OVER-SMOOTHING IN MEDICAL SEGMENTATION)

class DCESemanticMaskDecoder(MaskDecoder):
    """
    SAM2 semantic decoder with ML-DCE (MCC + ICC) added.
    - num_class_tokens includes background (C_tot = C_fg + 1).
    - Class queries exist only for foreground classes (C_fg).
    - Foreground slice of decoder mask tokens is enhanced; background untouched.
    """
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_class_tokens: int = 11,  # includes background
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        hypernet_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=3,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            dynamic_multimask_via_stability=dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=dynamic_multimask_stability_thresh,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
        )

        # ---- semantic tokens
        self.num_class_tokens = num_class_tokens              # K = C_tot (incl. bg)
        self.num_mask_tokens = num_class_tokens
        self.mask_tokens = nn.Embedding(num_class_tokens, transformer_dim)

        # ---- upscaling head (as you had it)
        self.hypernet_output_dim = (
            transformer_dim // 8 if hypernet_output_dim is None else hypernet_output_dim
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, self.hypernet_output_dim, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)
            self.upscale_residual_proj = nn.Conv2d(self.hypernet_output_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.upscale_post_add_proj = nn.Conv2d(transformer_dim // 8, self.hypernet_output_dim, kernel_size=1, stride=1)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, self.hypernet_output_dim, 3)
             for _ in range(self.num_mask_tokens)]
        )

        # =========================
        # ML-DCE (MCC + ICC)
        # =========================
        D = transformer_dim
        H = 8  # heads; you can expose as an arg
        self.num_fg_classes = max(0, num_class_tokens - 1)  # exclude background

        # Foreground class queries Q0 ∈ [C_fg, D]
        self.class_queries = nn.Embedding(self.num_fg_classes, D)

        # ---- MCC projections (Q ↔ decoder mask tokens) + residual norms
        self.mcc_q = nn.Linear(D, D, bias=False)
        self.mcc_k = nn.Linear(D, D, bias=False)
        self.mcc_v = nn.Linear(D, D, bias=False)
        self.mcc_attn = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)
        self.mcc_out = nn.Linear(D, D, bias=True)     # "Linear" after (concat)attn
        self.mcc_ln1 = nn.LayerNorm(D)
        self.mcc_mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.mcc_ln2 = nn.LayerNorm(D)

        # ---- ICC projections (Q ↔ encoder tokens) + residual norms
        self.icc_q = nn.Linear(D, D, bias=False)
        self.icc_k = nn.Linear(D, D, bias=False)
        self.icc_v = nn.Linear(D, D, bias=False)
        self.icc_attn = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)
        self.icc_out = nn.Linear(D, D, bias=True)
        self.icc_ln1 = nn.LayerNorm(D)
        self.icc_mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.icc_ln2 = nn.LayerNorm(D)

        # 1x1 to bring high-res encoder feats to D (handles 32ch when D=256)
        in_ch = (transformer_dim // 8) if use_high_res_features else transformer_dim
        self.icc_enc_proj = nn.Conv2d(in_ch, D, kernel_size=1)

        # Fuse MCC/ICC into a class-guidance delta and add residually to fg tokens
        self.alpha = nn.Parameter(torch.tensor(1.0))          # scale MCC
        self.beta  = nn.Parameter(torch.tensor(1.0))          # scale ICC
        self.cls_fuse = nn.Linear(2 * D, D)                   # fuse([α*MCC, β*ICC])
        
        # Update IoU head to predict IoU for each class (override parent's 4-output head)
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            num_class_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

    # @staticmethod
    # def _attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    #     # Q:[B,C,D], K:[B,N,D], V:[B,N,D] -> out:[B,C,D]
    #     d = Q.shape[-1]
    #     attn = torch.softmax(Q @ K.transpose(-1, -2) / (d ** 0.5), dim=-1)
    #     return attn @ V

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )
        sam_tokens_out = mask_tokens_out
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks + iou + returns decoder mask tokens (after ML-DCE gating)."""
        # ----- build tokens (IoU [+ optional obj] + K class mask tokens + sparse)
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)

        B = sparse_prompt_embeddings.size(0)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # ----- image stream
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # ----- transformer
        hs, src_tokens = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # [B, K(=C_tot), D]

        # =========================
        # ML-DCE: MCC + ICC + residual fusion (foreground only)
        # =========================
        if self.num_fg_classes > 0:
            # split bg / fg tokens correctly (keep batch dim!)
            t_bg = mask_tokens_out[:, :1, :]                                  # [B,1,D]
            fg_slice = mask_tokens_out[:, 1 : 1 + self.num_fg_classes, :]     # [B,C_fg,D]

            # class queries Q ∈ [B, C_fg, D]
            Q = self.class_queries.weight.unsqueeze(0).expand(b, -1, -1)

            # ---------- MCC (Q ↔ foreground decoder tokens) ----------
            Qm = self.mcc_q(Q)            # [B,C_fg,D]
            Km = self.mcc_k(fg_slice)     # [B,C_fg,D]    <-- foreground memory only
            Vm = self.mcc_v(fg_slice)     # [B,C_fg,D]

            # Multi-head cross-attention
            T_agg, _ = self.mcc_attn(Qm, Km, Vm)   # [B,C_fg,D]
            T_proj   = self.mcc_out(T_agg)
            T_res1   = self.mcc_ln1(Q + T_proj)    # Add & Norm
            T_ffn    = self.mcc_mlp(T_res1)
            T_MCC    = self.mcc_ln2(T_res1 + T_ffn)

            # Build encoder tokens for ICC
            if high_res_features is not None and len(high_res_features) >= 1:
                # enc_map = self.icc_enc_proj(high_res_features[0])   # feat_s0 -> [B,D,Hs,Ws]
                enc_map = image_embeddings
            else:
                enc_map = image_embeddings                           # [B,D,H',W']
            enc_tok = enc_map.flatten(2).transpose(1, 2)            # [B,N,D]

            Qi = self.icc_q(Q)               # [B,C_fg,D]
            Ki = self.icc_k(enc_tok)         # [B,N,D]
            Vi = self.icc_v(enc_tok)         # [B,N,D]

            T_agg_i, _ = self.icc_attn(Qi, Ki, Vi)   # [B,C_fg,D]
            T_proj_i   = self.icc_out(T_agg_i)
            T_res1_i   = self.icc_ln1(Q + T_proj_i)  # Add & Norm
            T_ffn_i    = self.icc_mlp(T_res1_i)
            T_ICC      = self.icc_ln2(T_res1_i + T_ffn_i)

            # ---- fuse and inject into foreground slice
            fused = self.cls_fuse(torch.cat([self.alpha * T_MCC, self.beta * T_ICC], dim=-1))  # [B,C_fg,D]
            fg_new = fg_slice + fused
            mask_tokens_out = torch.cat([t_bg, fg_new], dim=1)   # background untouched

        # ----- upscaling head (unchanged)
        src_tokens = src_tokens.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src_tokens)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src_tokens) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # ----- hypernets
        hyper_in = torch.stack(
            [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
             for i in range(self.num_mask_tokens)],
            dim=1,
        )  # [B,K,C']
        b2, c2, h2, w2 = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b2, c2, h2 * w2)).view(b2, -1, h2, w2)

        # ----- IoU / object score heads
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits
