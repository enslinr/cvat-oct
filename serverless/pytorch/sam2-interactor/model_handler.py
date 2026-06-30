import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _log(msg: str):
    """Write log message to both file and stderr"""
    log_file = Path("/tmp/sam2_init.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_file, "a") as f:
            f.write(f"{msg}\n")
            f.flush()
    except Exception:
        pass
    print(msg, file=sys.stderr, flush=True)


class ModelHandler:
    """Serverless wrapper around the SAM2 model."""

    def __init__(self) -> None:
        _log("[SAM2] Starting ModelHandler initialization...")

        device_env = os.environ.get("SAM2_DEVICE")
        if device_env:
            device = torch.device(device_env)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        _log(f"[SAM2] Using device: {self.device}")

        self.config_name = os.environ.get(
            "SAM2_CONFIG",
            "sam2.1/sam2.1_hiera_b+.yaml",
        )
        _log(f"[SAM2] Config: {self.config_name}")

        self.checkpoint_path = Path(os.environ.get("SAM2_CHECKPOINT", "/opt/nuclio/sam2/models/sam2.1_hiera_base_plus.pt"))
        _log(f"[SAM2] Checkpoint path: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            _log(f"[SAM2] ERROR: Checkpoint not found at {self.checkpoint_path}")
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Set SAM2_CHECKPOINT or mount the file into the container."
            )

        _log(f"[SAM2] Checkpoint found, size: {self.checkpoint_path.stat().st_size / (1024**3):.2f} GB")

        self.default_threshold = float(os.environ.get("SAM2_MASK_THRESHOLD", "0.0"))
        _log(f"[SAM2] Mask threshold: {self.default_threshold}")

        # Build standard SAM2 model
        _log("[SAM2] Loading model...")
        _log(f"[SAM2_mine] self.config_name: {str(self.config_name)}")
        _log(f"[SAM2_mine] self.checkpoint_path: {str(self.checkpoint_path)}")
        try:
            self.model = build_sam2(
                config_file=self.config_name,
                ckpt_path=str(self.checkpoint_path),
                device=str(self.device),
            )
            _log("[SAM2] Model loaded successfully!")
        except Exception as e:
            _log(f"[SAM2] ERROR loading model: {type(e).__name__}: {e}")
            raise

        self.model.eval()
        _log("[SAM2] ModelHandler initialization complete!")

    def handle(
        self,
        image: Image.Image,
        pos_points: Sequence[Sequence[float]],
        neg_points: Sequence[Sequence[float]],
        threshold: float,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        predictor = SAM2ImagePredictor(self.model, mask_threshold=threshold)
        np_image = np.asarray(image.convert("RGB"))
        predictor.set_image(np_image)

        point_coords, point_labels = self._build_points(pos_points, neg_points)

        if point_coords is None:
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=False,
                return_logits=False,
            )
        else:
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
                return_logits=False,
            )

        mask_array = self._select_mask(masks)
        polygon = self._mask_to_polygon(mask_array)
        return mask_array, polygon

    def _build_points(
        self,
        pos_points: Sequence[Sequence[float]],
        neg_points: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        coords: List[Tuple[float, float]] = []
        labels: List[int] = []

        for pt in pos_points:
            x, y = self._extract_xy(pt)
            if x is not None:
                coords.append((x, y))
                labels.append(1)

        for pt in neg_points:
            x, y = self._extract_xy(pt)
            if x is not None:
                coords.append((x, y))
                labels.append(0)

        if not coords:
            return None, None

        coord_array = np.asarray(coords, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        return coord_array, label_array

    @staticmethod
    def _extract_xy(point: Sequence[float] | Dict[str, float]) -> Tuple[float | None, float | None]:
        if isinstance(point, dict):
            if "x" in point and "y" in point:
                return float(point["x"]), float(point["y"])
            return None, None
        if isinstance(point, Sequence) and len(point) >= 2:
            return float(point[0]), float(point[1])
        return None, None

    def _select_mask(self, masks: np.ndarray) -> np.ndarray:
        # Standard SAM2 returns binary masks with shape (num_masks, H, W)
        # We take the first mask
        if masks.ndim == 3:
            mask = masks[0]
        elif masks.ndim == 2:
            mask = masks
        else:
            raise ValueError(f"Unexpected mask shape: {tuple(masks.shape)}")

        # Convert boolean/float mask to uint8 (0-255)
        mask_np = (mask > 0).astype(np.uint8) * 255
        return mask_np

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
        if mask.max() == 0:
            return []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        if largest.ndim == 3:
            largest = largest[:, 0, :]

        if largest.shape[0] < 3:
            return []

        return [[int(x), int(y)] for x, y in largest.tolist()]



if __name__ == "__main__":
    model_handler = ModelHandler()

