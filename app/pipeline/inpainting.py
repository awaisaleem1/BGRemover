import torch
import numpy as np
import cv2
from pathlib import Path
import sys

try:
    import lama_cleaner
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config, HDStrategy
except ImportError:
    print("LaMa not available. Please install: pip install lama-cleaner")

from app.config import LAMA_MODEL_PATH, INPAINTING_DEVICE
from app.utils.image_io import resize_image
from app.utils.resize import resize_with_padding, remove_padding


class Inpainter:
    def __init__(self, model_path=LAMA_MODEL_PATH, device=INPAINTING_DEVICE):
        self.device = device
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load LaMa model"""
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config

            model_manager = ModelManager(name="lama", device=self.device)
            self.model = model_manager.model

            if self.model is None:
                raise RuntimeError("Failed to load LaMa model")

        except Exception as e:
            print(f"Error loading LaMa model: {e}")
            print("Using OpenCV inpainting as fallback")
            self.model = None

    def inpaint(self, image, mask):
        """
        Inpaint masked regions of the image.

        IMPORTANT: Preserves the alpha channel if present.
        """
        # ── Save alpha channel if it exists ───────────────────────────
        alpha_channel = None
        if len(image.shape) == 3 and image.shape[2] == 4:
            alpha_channel = image[:, :, 3].copy()
            # Convert RGBA → BGR for processing
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = image.copy()
        else:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8) * 255

        # ── Only inpaint within opaque regions ────────────────────────
        if alpha_channel is not None:
            opaque = (alpha_channel > 127).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, opaque)

        # Use LaMa or fallback
        if self.model is None:
            inpainted_bgr = self._inpaint_opencv_bgr(image_bgr, mask)
        else:
            inpainted_bgr = self._inpaint_lama(image_bgr, mask)

        # ── Re-attach alpha channel ───────────────────────────────────
        if alpha_channel is not None:
            b, g, r = cv2.split(inpainted_bgr)
            result = cv2.merge([b, g, r, alpha_channel])
            # Convert BGRA → RGBA
            result = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
            return result
        else:
            # Convert BGR → RGB
            return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    def _inpaint_lama(self, image_bgr, mask):
        """Run LaMa inpainting on BGR image"""
        h, w = image_bgr.shape[:2]
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32

        proc_img = image_bgr
        proc_mask = mask

        if new_h != h or new_w != w:
            proc_img = cv2.resize(image_bgr, (new_w, new_h))
            proc_mask = cv2.resize(mask, (new_w, new_h))

        try:
            from lama_cleaner.schema import Config

            config = Config(
                ldm_steps=20,
                ldm_sampler="plms",
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=128,
                hd_strategy_crop_trigger_size=1280,
                hd_strategy_resize_limit=2048
            )

            result = self.model(proc_img, proc_mask, config)

            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h))

            return result

        except Exception as e:
            print(f"LaMa inpainting failed: {e}, falling back to OpenCV")
            return self._inpaint_opencv_bgr(image_bgr, mask)

    def _inpaint_opencv_bgr(self, image_bgr, mask):
        """Fallback: OpenCV inpainting on BGR image"""
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

        inpainted = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
        return inpainted

    def remove_text(self, image, text_mask):
        """Remove text from image, preserving transparency."""
        return self.inpaint(image, text_mask)