import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import cv2
from skimage import measure

import numpy as np
import torch
from PIL import Image, ImageOps

from sam2.build_semantic_sam2 import build_semantic_sam2
from sam2.sam2_image_predictor_semantic import SAM2ImagePredictor


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

        self.num_objects = int(os.environ.get("SAM2_NUM_OBJECTS", "9"))
        _log(f"[SAM2] Number of expected objects: {self.num_objects}")

        # Number of output classes the checkpoint was trained with, INCLUDING background.
        # MGU (peripapillary): 10 layers + background = 11. NR206 (macular): 8 layers + background = 9.
        # Must match the loaded SAM2_CHECKPOINT, or weight loading fails with a shape mismatch.
        self.num_classes = int(os.environ.get("SAM2_NUM_CLASSES", "11"))
        _log(f"[SAM2] Number of classes (incl. background): {self.num_classes}")

        # Build standard SAM2 model
        _log("[SAM2] Loading model...")
        _log(f"[SAM2_mine] self.config_name: {str(self.config_name)}")
        _log(f"[SAM2_mine] self.checkpoint_path: {str(self.checkpoint_path)}")
        try:

            hydra_overrides = [
                "++model._target_=sam2.semantic_sam2.semantic_sam2_components.SAM2Semantic",
                f"++model.num_classes={self.num_classes}",  # from SAM2_NUM_CLASSES (incl. background)
                "++model.num_maskmem=0",
                "++model.use_mask_input_as_output_without_sam=false",
                "++model.multimask_output_in_sam=false",
                "++model.pred_obj_scores=false",
                "++model.pred_obj_scores_mlp=false",
                "++model.fixed_no_obj_ptr=false",
            ]
            self.model = build_semantic_sam2(
                config_file=self.config_name,
                ckpt_path=str(self.checkpoint_path),
                device=str(self.device),
                hydra_overrides_extra=hydra_overrides,
                apply_postprocessing=False,
                use_load_checkpoint_staged_safe=True
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
    ) -> Tuple[np.ndarray, List[Dict[str, any]]]:
        """
        Handle OCT image segmentation, returning multiple object masks.

        Returns:
            Tuple of (label_mask, polygons) where:
            - label_mask: 2D array with pixel values 0-N (0=background, 1-N=objects)
            - polygons: List of dicts with 'points' and 'label' for each object
        """
        predictor = SAM2ImagePredictor(self.model, mask_threshold=threshold)
        image = image.convert("RGB")
        image = ImageOps.equalize(image)        
        np_image = np.asarray(image)

        predictor.set_image(np_image)

        point_coords, point_labels = self._build_points(pos_points, neg_points)

        if point_coords is None:
            masks, iou_predictions, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=False,
                return_logits=True,
            )
        else:
            masks, iou_predictions, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
                return_logits=True,
            )

        _log(f"[SAM2] Received masks from predictor")
        _log(f"[SAM2-mine] Mask shape: {masks.shape}")

        # masks shape: [batch_size, num_classes, H, W] - logits
        # Create argmax mask where each pixel is assigned to the most confident class
        masks_np = masks
        if masks_np.ndim == 4:
            # Remove batch dimension if present
            masks_np = masks_np[0]  # Now shape: [num_classes, H, W]

        # Get the class with max confidence for each pixel
        argmax_mask = np.argmax(masks_np, axis=0)  # Shape: [H, W]
        _log(f"[SAM2] Argmax mask shape: {argmax_mask.shape}, unique labels: {np.unique(argmax_mask)}")

        # Create label mask from argmax (each pixel already has its class label)
        label_mask = argmax_mask.astype(np.uint8)

        # Un-pad the mask to match original image dimensions
        # The predictor pads the image to 1024x1024, so the mask is also 1024x1024
        # We need to crop it back to the original image size to align coordinates
        orig_w, orig_h = image.size
        target_size = 1024
        
        if orig_h < target_size or orig_w < target_size:
            pad_top = (target_size - orig_h) // 2
            pad_left = (target_size - orig_w) // 2
            
            # Crop the mask
            label_mask = label_mask[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]
            _log(f"[SAM2] Un-padded mask from {argmax_mask.shape} to {label_mask.shape} (top={pad_top}, left={pad_left})")

        # Extract polygons for each object
        polygons = self._extract_all_polygons(label_mask)

        _log(f"[SAM2] Created label mask with {len(polygons)} objects")

        return label_mask, polygons

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

    # def _extract_all_polygons(self, label_mask: np.ndarray) -> List[Dict[str, any]]:
    #     """
    #     Extract polygons for each labeled object in the mask.
    #     """
    #     unique_labels = np.unique(label_mask)
    #     object_labels = unique_labels[unique_labels > 0]

    #     _log(f"[SAM2] Extracting polygons for {len(object_labels)} labeled objects")

    #     # Call once for ALL labels - it handles everything internally
    #     shapes = self.extract_polygons_with_boundary_snapping(
    #         label_mask,
    #         tolerance=2.0,
    #         snap_distance=1.5,
    #         min_area=100
    #     )

    #     _log(f"[SAM2] Total: {len(object_labels)} objects → {len(shapes)} shapes")

    #     return shapes

    def _extract_all_polygons(self, label_mask: np.ndarray) -> List[Dict[str, any]]:
        """
        Edge-graph extractor entry point. Each boundary point is shared verbatim
        by the two regions it separates — no per-region drift possible.
        See `_extract_aligned_polygons_via_edge_graph` for the algorithm.
        """
        return self._extract_aligned_polygons_via_edge_graph(
            label_mask, min_area=100.0, simplify_tol=0.5
        )

    def _extract_aligned_polygons_via_edge_graph(
        self,
        label_mask: np.ndarray,
        min_area: float = 100.0,
        simplify_tol: float = 0.5,
    ) -> List[Dict[str, any]]:
        """
        Extract one polygon per connected region such that adjacent regions share
        IDENTICAL boundary point sequences by construction.

        Algorithm:
            1. Pad the label mask with an OUTSIDE sentinel (-1) so the image edge
               is itself a real boundary.
            2. Build a planar boundary graph: every pixel-pixel adjacency where
               the two pixels carry different labels contributes one undirected
               unit boundary edge between two grid corners.
            3. Junctions = corners with degree != 2 (≥3 labels meeting, image-edge
               turns at multi-label corners, or saddles).
            4. Trace polylines along boundary edges through degree-2 corners,
               stopping at junctions or returning to the start (closed loops with
               no junctions). Each polyline carries a fixed (label_left, label_right)
               pair along its entire length.
            5. Optionally simplify each polyline once (Douglas-Peucker), preserving
               endpoints. Both regions sharing the polyline therefore get the
               same simplified geometry.
            6. For each label L, stitch its boundary cycles by walking polyline
               instances oriented so L is on the LEFT, hopping junction-to-junction
               using the standard "CCW-next from reverse-incoming" rule.
            7. Filter cycles by area and emit shapes.
        """
        import math
        from collections import defaultdict

        H, W = label_mask.shape
        OUTSIDE = -1

        padded = np.full((H + 2, W + 2), OUTSIDE, dtype=np.int64)
        padded[1:H + 1, 1:W + 1] = label_mask
        PH, PW = padded.shape

        def to_xy(corner: Tuple[int, int]) -> Tuple[float, float]:
            cc, cr = corner
            return (float(cc - 1.5), float(cr - 1.5))

        # adj[v] is a list of (neighbor_v, label_left, label_right). Walking
        # v -> neighbor_v: label_left is on walker's left, label_right on right.
        adj: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int, int]]] = defaultdict(list)

        # Horizontal edges: corners (c, r) <-> (c+1, r), separating padded
        # pixel (r-1, c) (above) from (r, c) (below). Walking east keeps the
        # above-pixel on the left.
        for r in range(PH + 1):
            ar = r - 1
            br = r
            ar_valid = 0 <= ar < PH
            br_valid = 0 <= br < PH
            for c in range(PW):
                la = int(padded[ar, c]) if ar_valid else OUTSIDE
                lb = int(padded[br, c]) if br_valid else OUTSIDE
                if la != lb:
                    adj[(c, r)].append(((c + 1, r), la, lb))
                    adj[(c + 1, r)].append(((c, r), lb, la))

        # Vertical edges: corners (c, r) <-> (c, r+1), separating padded pixel
        # (r, c-1) (west) from (r, c) (east). Walking south (y+) keeps the
        # east-pixel on the walker's left.
        for c in range(PW + 1):
            wc = c - 1
            ec = c
            wc_valid = 0 <= wc < PW
            ec_valid = 0 <= ec < PW
            for r in range(PH):
                lw = int(padded[r, wc]) if wc_valid else OUTSIDE
                le = int(padded[r, ec]) if ec_valid else OUTSIDE
                if lw != le:
                    adj[(c, r)].append(((c, r + 1), le, lw))
                    adj[(c, r + 1)].append(((c, r), lw, le))

        if not adj:
            return []

        junctions = {v for v in adj if len(adj[v]) != 2}

        def angle_of(corner_from, corner_to):
            x1, y1 = to_xy(corner_from)
            x2, y2 = to_xy(corner_to)
            a = math.atan2(y2 - y1, x2 - x1)
            return a if a >= 0 else a + 2 * math.pi

        # Cyclic-by-angle order of outgoing edges at each junction.
        junction_sorted: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int, int]]] = {}
        for J in junctions:
            junction_sorted[J] = sorted(adj[J], key=lambda e: angle_of(J, e[0]))

        visited_edges: set = set()
        polylines: List[Dict] = []

        def trace(start, first_entry):
            nei, ll, lr = first_entry
            ek = frozenset({start, nei})
            if ek in visited_edges:
                return None
            visited_edges.add(ek)
            path = [start, nei]
            prev, curr = start, nei
            while curr != start and curr not in junctions:
                next_v = None
                for nv, nll, nlr in adj[curr]:
                    if nv == prev:
                        continue
                    # Degree-2 non-junction corner: only one other edge exists,
                    # and topologically it must carry the same (ll, lr) pair.
                    if nll == ll and nlr == lr:
                        next_v = nv
                        break
                if next_v is None:
                    break
                ek2 = frozenset({curr, next_v})
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                path.append(next_v)
                prev, curr = curr, next_v
            return {"path": path, "ll": ll, "lr": lr, "closed": curr == start}

        for J in list(junctions):
            for entry in adj[J]:
                ek = frozenset({J, entry[0]})
                if ek in visited_edges:
                    continue
                pl = trace(J, entry)
                if pl is not None:
                    polylines.append(pl)

        # Closed loops with no junctions on them.
        for v in list(adj.keys()):
            if v in junctions:
                continue
            for entry in adj[v]:
                ek = frozenset({v, entry[0]})
                if ek in visited_edges:
                    continue
                pl = trace(v, entry)
                if pl is not None:
                    polylines.append(pl)

        # Per-polyline simplification. Apply ONCE per polyline; both regions
        # that share it then receive the same simplified geometry.
        if simplify_tol and simplify_tol > 0:
            for pl in polylines:
                path = pl["path"]
                if len(path) < 4:
                    continue
                pts_xy = np.array([to_xy(c) for c in path], dtype=np.float32)
                if pl["closed"]:
                    # Drop trailing duplicate of start before approxPolyDP.
                    if np.allclose(pts_xy[0], pts_xy[-1]):
                        pts_xy = pts_xy[:-1]
                    if len(pts_xy) < 4:
                        continue
                    contour = pts_xy.reshape(-1, 1, 2)
                    simplified = cv2.approxPolyDP(contour, simplify_tol, True).reshape(-1, 2)
                    if len(simplified) >= 3:
                        # Re-close by appending start at end so downstream code
                        # that expects path[0]==path[-1] keeps working.
                        pl["simplified_xy"] = np.vstack([simplified, simplified[:1]])
                else:
                    contour = pts_xy.reshape(-1, 1, 2)
                    simplified = cv2.approxPolyDP(contour, simplify_tol, False).reshape(-1, 2)
                    # approxPolyDP can drop endpoints in rare cases; force-include them.
                    if len(simplified) < 2 or not np.allclose(simplified[0], pts_xy[0]):
                        simplified = np.vstack([pts_xy[:1], simplified])
                    if not np.allclose(simplified[-1], pts_xy[-1]):
                        simplified = np.vstack([simplified, pts_xy[-1:]])
                    pl["simplified_xy"] = simplified

        def polyline_xy(pi: int) -> np.ndarray:
            pl = polylines[pi]
            if "simplified_xy" in pl:
                return pl["simplified_xy"]
            return np.array([to_xy(c) for c in pl["path"]], dtype=np.float32)

        # Directed-key index of polylines for stitching: from corner C, going
        # to corner N via polyline P with direction D.
        directed_index: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, str]] = {}
        for pi, pl in enumerate(polylines):
            path = pl["path"]
            if len(path) < 2:
                continue
            directed_index[(path[0], path[1])] = (pi, "fwd")
            directed_index[(path[-1], path[-2])] = (pi, "rev")

        def label_on_left(pi: int, direction: str) -> int:
            pl = polylines[pi]
            return pl["ll"] if direction == "fwd" else pl["lr"]

        cycles_per_label: Dict[int, List[List[Tuple[float, float]]]] = defaultdict(list)
        visited_directed: set = set()

        unique = np.unique(label_mask)
        object_labels = [int(L) for L in unique if int(L) > 0]

        for L in object_labels:
            candidates: List[Tuple[int, str]] = []
            for pi, pl in enumerate(polylines):
                if pl["ll"] == L:
                    candidates.append((pi, "fwd"))
                if pl["lr"] == L:
                    candidates.append((pi, "rev"))

            for start_pi, start_dir in candidates:
                if (start_pi, start_dir) in visited_directed:
                    continue

                cycle_xy: List[Tuple[float, float]] = []
                curr_pi, curr_dir = start_pi, start_dir
                safety_bound = len(candidates) + 2

                for _ in range(safety_bound):
                    if (curr_pi, curr_dir) in visited_directed:
                        break
                    visited_directed.add((curr_pi, curr_dir))

                    pts = polyline_xy(curr_pi)
                    if curr_dir == "rev":
                        pts = pts[::-1]
                    if not cycle_xy:
                        cycle_xy.extend([(float(p[0]), float(p[1])) for p in pts])
                    else:
                        # First point of this segment equals last point of previous; skip it.
                        cycle_xy.extend([(float(p[0]), float(p[1])) for p in pts[1:]])

                    if polylines[curr_pi]["closed"]:
                        break

                    path = polylines[curr_pi]["path"]
                    end_corner = path[-1] if curr_dir == "fwd" else path[0]
                    incoming_from = path[-2] if curr_dir == "fwd" else path[1]

                    if end_corner not in junctions:
                        break

                    sorted_edges = junction_sorted[end_corner]
                    rev_idx = None
                    for i, (nei, _, _) in enumerate(sorted_edges):
                        if nei == incoming_from:
                            rev_idx = i
                            break
                    if rev_idx is None:
                        break
                    next_idx = (rev_idx + 1) % len(sorted_edges)
                    next_nei = sorted_edges[next_idx][0]

                    key = (end_corner, next_nei)
                    if key not in directed_index:
                        break
                    next_pi, next_dir = directed_index[key]

                    if label_on_left(next_pi, next_dir) != L:
                        # Topology surprise; abort to avoid an infinite loop.
                        break

                    curr_pi, curr_dir = next_pi, next_dir

                # Drop trailing duplicate of cycle start, if any.
                if len(cycle_xy) >= 2 and cycle_xy[0] == cycle_xy[-1]:
                    cycle_xy = cycle_xy[:-1]
                if len(cycle_xy) >= 3:
                    cycles_per_label[L].append(cycle_xy)

        shapes: List[Dict[str, any]] = []
        for L in sorted(cycles_per_label.keys()):
            for cycle in cycles_per_label[L]:
                n = len(cycle)
                area_sum = 0.0
                for i in range(n):
                    x1, y1 = cycle[i]
                    x2, y2 = cycle[(i + 1) % n]
                    area_sum += x1 * y2 - x2 * y1
                area = abs(area_sum) * 0.5
                if area < min_area:
                    continue
                shapes.append({
                    "points": [[p[0], p[1]] for p in cycle],
                    "label": f"OCT_Object_{int(L)}",
                })

        _log(
            f"[SAM2] edge-graph extract: corners={len(adj)} junctions={len(junctions)} "
            f"polylines={len(polylines)} shapes={len(shapes)} (after area>={min_area})"
        )
        return shapes

    @staticmethod
    def _mask_to_polygon_aligned(label_mask: np.ndarray, target_label: int, min_area: int = 100) -> List[List[List[float]]]:
        """
        Convert a label mask to polygons using marching squares for perfect boundary alignment.
        
        The marching squares algorithm (via skimage.measure.find_contours) traces contours
        at the specified level, returning sub-pixel coordinates. Using level=0.5 ensures
        the contour is traced exactly at the boundary between foreground (1) and background (0)
        pixels, so adjacent masks will share identical edge coordinates.

        Args:
            label_mask: Multi-label mask with values 0-N (0=background, 1-N=objects)
            target_label: The label ID to extract polygons for
            min_area: Minimum polygon area in pixels to include (filters noise)

        Returns:
            List of polygons, where each polygon is a list of [x, y] coordinates
            Coordinates are at pixel boundaries (e.g., 5.5 between pixels 5 and 6)
        """
        # Create binary mask for the target label
        binary_mask = (label_mask == target_label).astype(np.float64)
        
        if binary_mask.max() == 0:
            return []

        # find_contours uses marching squares algorithm
        # level=0.5 traces the contour at the actual boundary between pixels
        # Returns coordinates in (row, col) format, i.e., (y, x)
        contours = measure.find_contours(binary_mask, level=0.5)
        
        if not contours:
            return []

        polygons = []
        
        for contour in contours:
            # Need at least 3 points for a valid polygon
            if len(contour) < 3:
                continue
            
            # Calculate polygon area using shoelace formula
            # skimage returns (row, col) = (y, x)
            x = contour[:, 1]  # columns = x
            y = contour[:, 0]  # rows = y
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            # Filter small noise contours
            if area < min_area:
                continue

            # Convert from (row, col) to [x, y] format
            # Remove duplicate consecutive points
            polygon = []
            prev_point = None
            
            for row, col in contour:
                x_coord = float(col)  # Already at boundary (e.g., 5.5)
                y_coord = float(row)
                
                current_point = (x_coord, y_coord)
                
                # Skip duplicate consecutive points (within floating point tolerance)
                if prev_point is None or (abs(x_coord - prev_point[0]) > 1e-9 or 
                                        abs(y_coord - prev_point[1]) > 1e-9):
                    polygon.append([x_coord, y_coord])
                    prev_point = current_point
            
            # Ensure we have at least 3 unique points
            if len(polygon) >= 3:
                polygons.append(polygon)

        return polygons


    # ----- OPTIONAL: If you need to keep integer coordinates -----
    # Some annotation tools require integer pixel coordinates.
    # In that case, you can round to the nearest pixel, but adjacent
    # masks will still share the same rounded boundary.

    @staticmethod
    def _mask_to_polygon_aligned_integer(label_mask: np.ndarray, target_label: int, min_area: int = 100) -> List[List[List[int]]]:
        """
        Same as above but rounds to integer coordinates.
        Adjacent masks will share the same rounded coordinates.
        """
        binary_mask = (label_mask == target_label).astype(np.float64)
        
        if binary_mask.max() == 0:
            return []

        contours = measure.find_contours(binary_mask, level=0.5)
        
        if not contours:
            return []

        polygons = []
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            x = contour[:, 1]
            y = contour[:, 0]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            if area < min_area:
                continue

            # Round to integers and remove duplicates
            polygon = []
            prev_point = None
            
            for row, col in contour:
                x_int = int(round(col))
                y_int = int(round(row))
                
                current_point = (x_int, y_int)
                if current_point != prev_point:
                    polygon.append([x_int, y_int])
                    prev_point = current_point
            
            if len(polygon) >= 3:
                polygons.append(polygon)

        return polygons
    





#--------------------------------------------------------------------
# new simplified shapes

    def extract_polygons_with_boundary_snapping(
        self,
        label_mask: np.ndarray,
        tolerance: float = 2.0,
        snap_distance: float = 1.0,
        min_area: int = 100
    ) -> List[Dict]:
        """
        Extract simplified polygons with post-process boundary snapping.
        
        This simpler approach:
        1. Simplifies each polygon independently
        2. Collects all boundary points
        3. Snaps nearby points to the same coordinates
        
        Args:
            label_mask: 2D array with integer labels
            tolerance: Douglas-Peucker tolerance
            snap_distance: Maximum distance to snap points together
            min_area: Minimum polygon area
            
        Returns:
            List of polygon dicts
        """
        unique = np.unique(label_mask)
        object_labels = [int(l) for l in unique if l > 0]
        
        # First pass: extract all simplified polygons
        all_polygons_raw = {}
        
        for label_id in object_labels:
            binary_mask = (label_mask == label_id).astype(float)
            
            if binary_mask.max() == 0:
                continue
            
            contours = measure.find_contours(binary_mask, level=0.5)
            
            label_polygons = []
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                x, y = contour[:, 1], contour[:, 0]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                
                if area < min_area:
                    continue
                
                xy = contour[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
                simplified = cv2.approxPolyDP(xy, tolerance, closed=True).reshape(-1, 2)
                
                if len(simplified) >= 3:
                    label_polygons.append(simplified)
            
            if label_polygons:
                all_polygons_raw[label_id] = label_polygons
        
        # Collect all unique points
        all_points = []
        for label_id, polygons in all_polygons_raw.items():
            for poly in polygons:
                for pt in poly:
                    all_points.append(tuple(pt))
        
        # Build point clusters (snap nearby points)
        point_map = self._build_snap_map(all_points, snap_distance)
        
        # Second pass: snap points and build final polygons
        result = []
        
        for label_id, polygons in all_polygons_raw.items():
            for poly in polygons:
                snapped = []
                for pt in poly:
                    key = tuple(pt)
                    snapped_pt = point_map.get(key, key)
                    snapped.append([float(snapped_pt[0]), float(snapped_pt[1])])
                
                # Remove consecutive duplicates
                unique_pts = [snapped[0]]
                for pt in snapped[1:]:
                    if pt != unique_pts[-1]:
                        unique_pts.append(pt)
                
                if len(unique_pts) >= 3:
                    result.append({
                        'points': unique_pts,
                        'label': f"OCT_Object_{label_id}"
                    })
        
        return result


    def _build_snap_map(
        self,
        points: List[Tuple[float, float]], 
        snap_distance: float
    ) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """Build a mapping from original points to snapped points."""
        if not points:
            return {}
        
        points_array = np.array(list(set(points)))
        
        if len(points_array) == 0:
            return {}
        
        # Find clusters of nearby points
        clusters = []
        used = set()
        
        for i, pt in enumerate(points_array):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j, other_pt in enumerate(points_array):
                if j in used:
                    continue
                
                dist = np.sqrt(np.sum((pt - other_pt) ** 2))
                if dist <= snap_distance:
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        # Build snap map: all points in a cluster map to the cluster centroid
        snap_map = {}
        
        for cluster in clusters:
            cluster_points = points_array[cluster]
            centroid = cluster_points.mean(axis=0)
            # Round centroid to 1 decimal place
            centroid = np.round(centroid, 1)
            
            for idx in cluster:
                original = tuple(points_array[idx])
                snap_map[original] = tuple(centroid)
        
        return snap_map











if __name__ == "__main__":
    # Local smoke-test harness (runs the handler directly, bypassing Nuclio).
    # Configure paths via environment variables, e.g.:
    #   SAM2_CHECKPOINT=./models/sam2.1_hiera_base_plus.pt \
    #   SAM2_TEST_IMAGE=/path/to/oct_image.png \
    #   python model_handler.py
    os.environ.setdefault("SAM2_CONFIG", "sam2.1/sam2.1_hiera_b+.yaml")
    os.environ.setdefault(
        "SAM2_CHECKPOINT",
        os.path.join(os.path.dirname(__file__), "models", "sam2.1_hiera_base_plus.pt"),
    )
    os.environ.setdefault("SAM2_DEVICE", "cuda")  # or "cpu" if no GPU
    os.environ.setdefault("SAM2_MASK_THRESHOLD", "0.0")
    os.environ.setdefault("SAM2_NUM_OBJECTS", "10")  # 10 OCT structures (no background)
    os.environ.setdefault("SAM2_NUM_CLASSES", "11")  # MGU: 10 layers + background

    print("Initializing model with local paths...")
    model_handler = ModelHandler()
    print("Model loaded successfully!")

    # Optional: point SAM2_TEST_IMAGE at an OCT image to run a sample segmentation.
    test_image_path = os.environ.get("SAM2_TEST_IMAGE", "")
    if test_image_path and os.path.exists(test_image_path):
        from PIL import Image
        test_image = Image.open(test_image_path)
    
        # Test with no points (automatic segmentation)
        label_mask, polygons = model_handler.handle(
            image=test_image,
            pos_points=[],
            neg_points=[],
            threshold=0.0
        )
    
        print(f"Segmentation complete!")
        print(f"  - Label mask shape: {label_mask.shape}")
        print(f"  - Number of objects found: {len(polygons)}")

        # Visualize the label mask with polygon overlays for debugging
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Show the label mask as background
        im = ax.imshow(label_mask, cmap='tab20', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Object ID')
        
        # Draw pixel boundaries as thin black lines
        h, w = label_mask.shape
        # Vertical lines (at each pixel boundary)
        for x in range(w + 1):
            ax.axvline(x - 0.5, color='black', linewidth=0.2, alpha=0.5)
        # Horizontal lines (at each pixel boundary)
        for y in range(h + 1):
            ax.axhline(y - 0.5, color='black', linewidth=0.2, alpha=0.5)
        
        # Define colors for polygon overlays (cycle through distinct colors)
        color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        
        # Draw each polygon as an overlay
        for i, poly in enumerate(polygons):
            points = poly['points']
            label = poly['label']
            
            if len(points) >= 3:
                # Convert points to numpy array for matplotlib
                poly_points = np.array(points)
                
                # Get color for this polygon (cycle through colors)
                color = color_list[i % len(color_list)]
                
                # Draw thin joining lines for polygon points
                ax.plot(
                    np.append(poly_points[:, 0], poly_points[0, 0]),  # Close the polygon
                    np.append(poly_points[:, 1], poly_points[0, 1]),
                    color=color,
                    linewidth=0.5,
                    linestyle='-',
                    alpha=0.7,
                    zorder=4
                )
                
                # Draw polygon vertices as dots (larger than lines)
                ax.scatter(poly_points[:, 0], poly_points[:, 1], color=color, s=3, zorder=5)
        
        ax.set_title(f'Label Mask with Polygon Overlays - {len(polygons)} objects detected')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add legend (limit to first 15 items to avoid overcrowding)
        if len(polygons) <= 15:
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right', fontsize=6, ncol=2)
        
        plt.tight_layout()
        plt.show()

        # Print polygon details
        print("\nPolygon details:")
        for i, poly in enumerate(polygons):
            print(f"  - {poly['label']}: {len(poly['points'])} points")
