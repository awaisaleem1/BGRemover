import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Local model paths (text detection & inpainting)
CRAFT_MODEL_PATH = MODEL_DIR / "craft" / "craft_mlt_25k.pth"
LAMA_MODEL_PATH = MODEL_DIR / "lama" / "big-lama.safetensors"
SAM_MODEL_PATH = MODEL_DIR / "sam" / "sam_vit_h.pth"

# Background removal — BiRefNet via rembg (auto-downloaded to ~/.u2net/)
BG_REMOVAL_MODEL = "birefnet-general-lite"

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Device detection
def get_device(device_type="auto"):
    if device_type == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    return device_type

TEXT_DETECTION_DEVICE = get_device("auto")
INPAINTING_DEVICE = get_device("auto")
SAM_DEVICE = get_device("auto")
SAM_MODEL_TYPE = "vit_h"

# Output settings
OUTPUT_FORMAT = "png"
OUTPUT_BACKGROUND = "transparent"

# FastAPI settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}