import sys
from pathlib import Path

def _log(msg: str):
    """Write log message to both file and stderr"""
    log_file = Path("/tmp/sam2_init.log")
    try:
        with open(log_file, "a") as f:
            f.write(f"{msg}\n")
            f.flush()
    except Exception:
        pass
    print(msg, file=sys.stderr, flush=True)

_log("[MAIN] Starting imports...")

try:
    import base64
    _log("[MAIN] ✓ base64 imported")
except Exception as e:
    _log(f"[MAIN] ✗ Failed to import base64: {e}")
    raise

try:
    import io
    _log("[MAIN] ✓ io imported")
except Exception as e:
    _log(f"[MAIN] ✗ Failed to import io: {e}")
    raise

try:
    import json
    _log("[MAIN] ✓ json imported")
except Exception as e:
    _log(f"[MAIN] ✗ Failed to import json: {e}")
    raise

try:
    from typing import Any, Dict
    _log("[MAIN] ✓ typing imported")
except Exception as e:
    _log(f"[MAIN] ✗ Failed to import typing: {e}")
    raise

try:
    from PIL import Image
    _log("[MAIN] ✓ PIL.Image imported")
except Exception as e:
    _log(f"[MAIN] ✗ Failed to import PIL.Image: {e}")
    raise

try:
    from model_handler import ModelHandler
    _log("[MAIN] ✓ ModelHandler imported")
except Exception as e:
    _log(f"[MAIN] ✗ Failed to import ModelHandler: {e}")
    raise

_log("[MAIN] All imports successful!")


def _ensure_dict(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, (bytes, bytearray)):
        return json.loads(data.decode("utf-8"))
    if isinstance(data, str):
        return json.loads(data)
    raise ValueError("Unsupported payload type received by SAM2 interactor")


def init_context(context):
    _log("[MAIN] init_context called")
    context.logger.info("Initializing SAM2 interactor...")
    _log("[MAIN] Creating ModelHandler...")
    try:
        context.user_data.model = ModelHandler()
        _log("[MAIN] ModelHandler created successfully")
        context.logger.info("SAM2 model ready.")
    except Exception as e:
        _log(f"[MAIN] ERROR in init_context: {type(e).__name__}: {e}")
        context.logger.error(f"Failed to initialize: {e}")
        raise


def handler(context, event):
    try:
        context.logger.info("Handler called - parsing request")
        data = _ensure_dict(event.body)
        context.logger.info(f"Request data keys: {list(data.keys())}")

        threshold = float(data.get("threshold", context.user_data.model.default_threshold))
        pos_points = data.get("pos_points", [])
        neg_points = data.get("neg_points", [])
        context.logger.info(f"Params - threshold: {threshold}, pos: {len(pos_points)}, neg: {len(neg_points)}")

        image_bytes = base64.b64decode(data["image"])
        context.logger.info(f"Decoded image: {len(image_bytes)} bytes")
        image = Image.open(io.BytesIO(image_bytes))
        context.logger.info(f"Image loaded: {image.size} {image.mode}")

        label_mask, polygons = context.user_data.model.handle(
            image=image,
            pos_points=pos_points,
            neg_points=neg_points,
            threshold=threshold,
        )
        context.logger.info(f"Inference complete - label mask shape: {label_mask.shape}, objects: {len(polygons)}")

        # Return multiple polygon shapes for CVAT
        # Each polygon represents a different OCT object
        response = {
            "mask": label_mask.tolist(),  # Label mask for reference
            "shapes": [
                {
                    "type": "polygon",
                    "points": poly["points"],
                    "label": poly["label"]
                }
                for poly in polygons
            ]
        }

        return context.Response(
            body=json.dumps(response),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except Exception as e:
        context.logger.error(f"Handler error: {type(e).__name__}: {e}")
        import traceback
        context.logger.error(f"Traceback: {traceback.format_exc()}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )

