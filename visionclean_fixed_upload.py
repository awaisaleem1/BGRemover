"""
VisionClean v3.1 — Smart Text Removal (watermarks only, not logos)
Models:
  - BiRefNet (via rembg) → background removal
  - CRAFT → text detection
  - OpenCV → text inpainting
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import os
from pathlib import Path
import uvicorn
import cv2
import numpy as np
import time
import io
from PIL import Image

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS — Background Removal (BiRefNet via rembg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    print("⚠ rembg not installed: pip install rembg")
    REMBG_AVAILABLE = False
    def remove(image_bytes, **kwargs):
        return image_bytes
    def new_session(model_name):
        raise ImportError("rembg not installed")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS — Text Detection (CRAFT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from craft_text_detector import Craft
    CRAFT_AVAILABLE = True
except ImportError:
    print("⚠ CRAFT not installed: pip install craft-text-detector --no-deps")
    CRAFT_AVAILABLE = False


app = FastAPI(
    title="VisionClean API",
    version="3.1.0",
    description="Background removal + smart watermark removal (protects logo text)"
)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("🎯 VisionClean API v3.1 — Smart Watermark Detection")
print("=" * 60)


# ====================================================================
# SMART TEXT DETECTOR — distinguishes watermarks from design text
# ====================================================================

class SmartTextDetector:
    """
    Detect WATERMARK text only — NOT logo/design text.
    
    Watermarks vs Design Text:
    ┌───────────────────────┬────────────────────────────────┐
    │ WATERMARK (remove)    │ DESIGN TEXT (keep)             │
    ├───────────────────────┼────────────────────────────────┤
    │ Semi-transparent      │ Fully opaque                   │
    │ At edges/corners      │ Centered / part of layout      │
    │ Repeated/tiled        │ Single instance                │
    │ Low contrast overlay  │ High contrast, clear borders   │
    │ Covers large area     │ Integrated with artwork        │
    │ Gray/white on photos  │ Styled, colored, designed      │
    │ Small thin text       │ Large decorative text          │
    └───────────────────────┴────────────────────────────────┘
    """

    def __init__(self):
        self.craft = None
        self._load()

    def _load(self):
        if not CRAFT_AVAILABLE:
            print("  ✗ CRAFT not available — using fallback")
            return

        craft_model_path = MODEL_DIR / "craft" / "craft_mlt_25k.pth"

        try:
            self.craft = Craft(
                output_dir=None,
                crop_type="poly",
                model_path=str(craft_model_path) if craft_model_path.exists() else None,
                cuda=False,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                canvas_size=1280,
                mag_ratio=1.5,
            )
            print("  ✓ CRAFT text detector loaded")
        except Exception as e:
            print(f"  ✗ CRAFT failed: {e}")
            self.craft = None

    def detect(self, image_rgb, sensitivity="low"):
        """
        Detect text with watermark filtering.
        
        Args:
            image_rgb: RGB numpy array
            sensitivity: "low" = only obvious watermarks (default, safe)
                        "medium" = moderate filtering
                        "high" = remove all detected text (old behavior)
        Returns:
            mask, num_regions
        """
        h, w = image_rgb.shape[:2]

        # Resize for speed
        max_size = 1500
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            image_rgb = cv2.resize(image_rgb,
                                   (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

        # Get raw text detections
        if self.craft is not None:
            raw_mask, raw_boxes = self._detect_craft_raw(image_rgb)
        else:
            raw_mask, raw_boxes = self._detect_fallback_raw(image_rgb)

        # Filter based on sensitivity
        if sensitivity == "high":
            # Old behavior — remove everything detected
            filtered_mask = raw_mask
        elif sensitivity == "medium":
            filtered_mask = self._filter_watermarks(image_rgb, raw_mask, raw_boxes,
                                                     strict=False)
        else:
            # "low" — only obvious watermarks (safest, default)
            filtered_mask = self._filter_watermarks(image_rgb, raw_mask, raw_boxes,
                                                     strict=True)

        # Resize mask back
        if scale != 1.0:
            filtered_mask = cv2.resize(filtered_mask, (w, h),
                                       interpolation=cv2.INTER_NEAREST)

        num = int(np.sum(filtered_mask > 0) / max(np.sum(filtered_mask > 0).clip(1), 1))
        # Count connected components as regions
        if filtered_mask.sum() > 0:
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(
                filtered_mask, connectivity=8
            )
            num = num_labels - 1
        else:
            num = 0

        return filtered_mask, num

    def _detect_craft_raw(self, image_rgb):
        """Raw CRAFT detection — returns mask + bounding boxes."""
        try:
            result = self.craft.detect_text(image_rgb)
            h, w = image_rgb.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            boxes = []

            polys = getattr(result, 'polys', None) or []
            if not polys:
                polys = result if isinstance(result, list) else []

            for poly in polys:
                if poly is None:
                    continue
                poly = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255)
                x, y, bw, bh = cv2.boundingRect(poly)
                boxes.append((x, y, bw, bh))

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            return mask, boxes

        except Exception as e:
            print(f"  CRAFT error: {e}")
            return self._detect_fallback_raw(image_rgb)

    def _detect_fallback_raw(self, image_rgb):
        """Fallback detection — returns mask + bounding boxes."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((h, w), dtype=np.uint8)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)

            if 50 < area < 8000 and 0.1 < aspect < 15:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                boxes.append((x, y, cw, ch))

        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask, boxes

    def _filter_watermarks(self, image_rgb, raw_mask, raw_boxes, strict=True):
        """
        THE KEY FIX: Filter out design text, keep only watermarks.
        
        Checks each text region against watermark criteria:
        1. Position — watermarks are usually at edges/corners
        2. Size — watermarks are usually small relative to image
        3. Opacity — watermarks are usually semi-transparent
        4. Contrast — watermarks have low contrast with background
        5. Coverage — if text covers too much area, it's design text
        6. Center proximity — text near center is likely the main design
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        filtered_mask = np.zeros_like(raw_mask)

        if not raw_boxes:
            return filtered_mask

        img_area = h * w
        img_center_x = w / 2
        img_center_y = h / 2

        # Calculate total text area
        total_text_area = np.sum(raw_mask > 0)
        text_coverage = total_text_area / img_area

        # If text covers more than 15% of image, it's likely a logo — skip ALL
        if text_coverage > 0.15:
            print(f"  📝 Text covers {text_coverage:.0%} of image — likely a logo, skipping")
            return filtered_mask

        for (x, y, bw, bh) in raw_boxes:
            box_area = bw * bh
            box_center_x = x + bw / 2
            box_center_y = y + bh / 2

            # ── Check 1: Is it in the center? ─────────────────────────
            # Text in the center 60% of the image is likely design text
            center_zone_x = (0.2 * w < box_center_x < 0.8 * w)
            center_zone_y = (0.2 * h < box_center_y < 0.8 * h)
            is_centered = center_zone_x and center_zone_y

            # ── Check 2: How big is it? ───────────────────────────────
            # Watermarks are usually small (< 5% of image area)
            size_ratio = box_area / img_area
            is_large = size_ratio > 0.05

            # ── Check 3: Is it semi-transparent? ──────────────────────
            # Extract the region and check contrast
            roi = gray[y:y+bh, x:x+bw]
            if roi.size == 0:
                continue

            roi_std = np.std(roi)
            roi_mean = np.mean(roi)

            # High contrast = design text (intentional)
            # Low contrast = watermark (subtle overlay)
            is_high_contrast = roi_std > 50

            # ── Check 4: Is it at an edge/corner? ─────────────────────
            edge_margin = 0.15  # 15% from edges
            at_top = y < h * edge_margin
            at_bottom = (y + bh) > h * (1 - edge_margin)
            at_left = x < w * edge_margin
            at_right = (x + bw) > w * (1 - edge_margin)
            is_at_edge = at_top or at_bottom or at_left or at_right

            # ── Check 5: Color — watermarks are usually gray/white ────
            roi_color = image_rgb[y:y+bh, x:x+bw]
            if roi_color.size > 0:
                color_std = np.std(roi_color, axis=(0, 1))
                is_grayscale = np.all(color_std < 30)  # Low color variation = gray
            else:
                is_grayscale = False

            # ── DECISION ──────────────────────────────────────────────
            is_watermark = False

            if strict:
                # STRICT (low sensitivity) — must match multiple watermark criteria
                watermark_score = 0
                if is_at_edge:
                    watermark_score += 2
                if not is_centered:
                    watermark_score += 1
                if not is_large:
                    watermark_score += 1
                if not is_high_contrast:
                    watermark_score += 2
                if is_grayscale:
                    watermark_score += 1

                # Need score >= 4 to be considered a watermark
                is_watermark = watermark_score >= 4

            else:
                # MEDIUM sensitivity — less strict
                watermark_score = 0
                if is_at_edge:
                    watermark_score += 2
                if not is_centered:
                    watermark_score += 1
                if not is_large:
                    watermark_score += 1
                if not is_high_contrast:
                    watermark_score += 1

                is_watermark = watermark_score >= 3

            if is_watermark:
                # Copy this region from raw_mask to filtered_mask
                region = raw_mask[y:y+bh, x:x+bw]
                filtered_mask[y:y+bh, x:x+bw] = region

        num_kept = 0
        if filtered_mask.sum() > 0:
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)
            num_kept = num_labels - 1

        total_detected = len(raw_boxes)
        print(f"  📝 Detected {total_detected} text regions → kept {num_kept} watermarks (filtered {total_detected - num_kept} design text)")

        return filtered_mask


# ====================================================================
# TEXT INPAINTER — OpenCV
# ====================================================================

class TextInpainter:
    """Remove text by inpainting masked regions using OpenCV."""

    def __init__(self):
        print("  ✓ OpenCV inpainter ready")

    def inpaint(self, image_bgr, mask):
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8) * 255
        else:
            mask = mask.astype(np.uint8) * 255

        if mask.sum() == 0:
            return image_bgr

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

        # Use larger radius for cleaner fills
        inpainted = cv2.inpaint(image_bgr, mask, 7, cv2.INPAINT_TELEA)
        return inpainted


# ====================================================================
# MAIN PIPELINE
# ====================================================================

class VisionCleanPipeline:

    def __init__(self):
        self.session = None
        self.model_name = None
        self._init_bg_model()

        self.text_detector = SmartTextDetector()
        self.text_inpainter = TextInpainter()

    def _init_bg_model(self):
        if not REMBG_AVAILABLE:
            print("  ✗ rembg not available")
            return

        models = [
            ("birefnet-general-lite", "Best quality/speed for CPU"),
            ("birefnet-general",      "Highest quality, slower"),
            ("isnet-general-use",     "Fallback"),
        ]

        print("  Loading BG model...")
        for name, desc in models:
            try:
                print(f"  ⏳ {name}...", end=" ", flush=True)
                self.session = new_session(name)
                self.model_name = name
                print(f"✓ ({desc})")
                return
            except Exception as e:
                print(f"✗ ({e})")

        print("  ⚠ Using default rembg")

    # ================================================================
    # PREPROCESSING
    # ================================================================

    def _preprocess_for_small_subject(self, pil_image):
        img = np.array(pil_image)
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()

        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pil_image, False

        significant = [c for c in contours if cv2.contourArea(c) > (h * w * 0.001)]
        if not significant:
            return pil_image, False

        combined = np.vstack(significant)
        x, y, bw, bh = cv2.boundingRect(combined)
        coverage = (bw * bh) / (w * h)

        if coverage > 0.6:
            return pil_image, False

        pad = int(max(bw, bh) * 0.25)
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(w, x + bw + pad)
        bottom = min(h, y + bh + pad)

        for _ in range(2):
            if (right - left) < 512:
                exp = (512 - (right - left)) // 2
                left = max(0, left - exp)
                right = min(w, right + exp)
            if (bottom - top) < 512:
                exp = (512 - (bottom - top)) // 2
                top = max(0, top - exp)
                bottom = min(h, bottom + exp)

        cropped = pil_image.crop((left, top, right, bottom))
        print(f"  📐 Subject: {coverage:.0%} → cropped to {right-left}x{bottom-top}")
        return cropped, True

    # ================================================================
    # BACKGROUND REMOVAL
    # ================================================================

    def _remove_background(self, image_bytes):
        try:
            t = time.time()
            print(f"  🤖 {self.model_name or 'default'}...", end=" ", flush=True)

            if self.session is not None:
                result = remove(image_bytes, session=self.session)
            else:
                result = remove(image_bytes)

            print(f"✓ ({time.time() - t:.1f}s)")
            return result
        except Exception as e:
            print(f"  ✗ {e}")
            return remove(image_bytes)

    # ================================================================
    # POST-PROCESSING — wheel-safe
    # ================================================================

    def _detect_wheel_regions(self, rgba_array):
        alpha = rgba_array[:, :, 3]
        rgb = rgba_array[:, :, :3]
        h, w = alpha.shape

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if len(rgb.shape) == 3 else rgb.copy()

        opaque = (alpha > 100).astype(np.uint8) * 255
        gray_masked = cv2.bitwise_and(gray, opaque)

        circles = cv2.HoughCircles(
            gray_masked, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w // 4,
            param1=100, param2=40,
            minRadius=int(min(h, w) * 0.05),
            maxRadius=int(min(h, w) * 0.25),
        )

        wheel_mask = np.zeros((h, w), dtype=np.uint8)

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (cx, cy, r) in circles:
                cv2.circle(wheel_mask, (cx, cy), int(r * 1.3), 255, -1)
            print(f"  🛞 {len(circles)} wheel(s) detected")
        else:
            binary = (alpha > 127).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                wheel_mask[y + int(bh * 0.7):y + bh, x:x + bw] = 255
            print("  🛞 Protecting bottom 30%")

        return wheel_mask

    def _remove_halo(self, rgba_array, wheel_mask):
        alpha = rgba_array[:, :, 3].copy()
        rgb = rgba_array[:, :, :3].copy()

        binary = (alpha > 200).astype(np.uint8) * 255

        erode_kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, erode_kernel, iterations=2)

        wheel_pixels = wheel_mask > 0
        eroded[wheel_pixels] = binary[wheel_pixels]

        dilate_kernel = np.ones((3, 3), np.uint8)
        edge_band = cv2.dilate(eroded, dilate_kernel, iterations=1) - eroded
        edge_band[wheel_pixels] = 0

        smooth_alpha = cv2.GaussianBlur(eroded.astype(np.float32), (5, 5), 1.0)

        final_alpha = eroded.astype(np.float32)
        final_alpha[edge_band > 0] = smooth_alpha[edge_band > 0]
        final_alpha = np.clip(final_alpha, 0, 255).astype(np.uint8)

        non_wheel_edge = (edge_band > 0) & (~wheel_pixels)
        if np.sum(non_wheel_edge) > 0:
            inpaint_mask = non_wheel_edge.astype(np.uint8)
            for c in range(3):
                filled = cv2.inpaint(rgb[:, :, c], inpaint_mask,
                                     inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                rgb[non_wheel_edge, c] = filled[non_wheel_edge]

        rgba_array[:, :, :3] = rgb
        rgba_array[:, :, 3] = final_alpha
        return rgba_array

    def _clean_mask(self, rgba_array, wheel_mask):
        alpha = rgba_array[:, :, 3]

        binary = (alpha > 127).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        wheel_pixels = wheel_mask > 0
        original_alpha = rgba_array[:, :, 3].copy()
        binary[wheel_pixels] = np.maximum(
            binary[wheel_pixels],
            (original_alpha[wheel_pixels] > 50).astype(np.uint8) * 255
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 2:
            sizes = stats[1:, cv2.CC_STAT_AREA]
            if len(sizes) > 0:
                max_size = np.max(sizes)
                clean = np.zeros_like(binary)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= max(max_size * 0.01, 100):
                        clean[labels == i] = 255
                binary = clean

        filled = binary.copy()
        border = cv2.copyMakeBorder(filled, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(border, None, (0, 0), 255)
        border = border[1:-1, 1:-1]
        holes_filled = binary | cv2.bitwise_not(border)

        if np.sum(holes_filled > 0) / max(np.sum(binary > 0), 1) < 1.5:
            binary = holes_filled

        rgba_array[:, :, 3] = np.where(binary > 0, rgba_array[:, :, 3], 0)
        return rgba_array

    def _remove_ground_shadow(self, rgba_array, wheel_mask):
        alpha = rgba_array[:, :, 3]
        h, w = alpha.shape

        binary = (alpha > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return rgba_array

        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)

        below_start = y + bh
        below_end = min(h, below_start + 40)

        for row in range(below_start, below_end):
            col_slice = slice(max(0, x - 20), min(w, x + bw + 20))
            row_alpha = alpha[row, col_slice]
            row_wheel = wheel_mask[row, col_slice]

            if np.mean(row_wheel > 0) > 0.3:
                continue

            if row_alpha.size > 0 and np.mean(row_alpha > 50) < 0.3:
                non_wheel = row_wheel == 0
                alpha_copy = alpha[row, col_slice].copy()
                alpha_copy[non_wheel & (alpha_copy < 200)] = 0
                alpha[row, col_slice] = alpha_copy

        rgba_array[:, :, 3] = alpha
        transparent = alpha < 10
        rgba_array[transparent, 0] = 0
        rgba_array[transparent, 1] = 0
        rgba_array[transparent, 2] = 0
        return rgba_array

    def _remove_sky_remnants(self, rgba_array):
        alpha = rgba_array[:, :, 3]
        rgb = rgba_array[:, :, :3]
        h, w = alpha.shape

        binary = (alpha > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return rgba_array

        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)

        top_end = y + int(bh * 0.10)
        scan_left = max(0, x - 10)
        scan_right = min(w, x + bw + 10)

        for row in range(max(0, y - 20), top_end):
            row_alpha = alpha[row, scan_left:scan_right]
            row_rgb = rgb[row, scan_left:scan_right]

            if row_alpha.size == 0 or row_rgb.shape[0] == 0:
                continue

            if row_rgb.shape[-1] >= 3:
                brightness = np.mean(row_rgb, axis=-1)
                is_sky = (brightness > 180) & (row_alpha > 20) & (row_alpha < 220)
                alpha_copy = alpha[row, scan_left:scan_right].copy()
                alpha_copy[is_sky] = 0
                alpha[row, scan_left:scan_right] = alpha_copy

        above_start = max(0, y - 30)
        above_end = max(0, y)
        if above_end > above_start:
            above_slice = alpha[above_start:above_end, scan_left:scan_right]
            above_slice[above_slice < 200] = 0

        rgba_array[:, :, 3] = alpha
        transparent = alpha < 10
        rgba_array[transparent, 0] = 0
        rgba_array[transparent, 1] = 0
        rgba_array[transparent, 2] = 0
        return rgba_array

    def _auto_crop(self, rgba_array, padding_pct=3):
        alpha = rgba_array[:, :, 3]
        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)

        if not np.any(rows) or not np.any(cols):
            return rgba_array

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = alpha.shape
        pad_h = int((rmax - rmin) * padding_pct / 100)
        pad_w = int((cmax - cmin) * padding_pct / 100)

        rmin = max(0, rmin - pad_h)
        rmax = min(h, rmax + pad_h + 1)
        cmin = max(0, cmin - pad_w)
        cmax = min(w, cmax + pad_w + 1)

        cropped = rgba_array[rmin:rmax, cmin:cmax]
        print(f"  ✂ {w}x{h} → {cmax-cmin}x{rmax-rmin}")
        return cropped

    # ================================================================
    # MAIN PIPELINE
    # ================================================================

    def process(self, image_bytes, remove_bg=True, remove_text=True,
                auto_crop=True, text_sensitivity="low"):
        """
        Pipeline v3.1:
          1. Smart text removal (watermarks only, single pass)
          2. Smart crop
          3. BiRefNet bg removal
          4. Wheel-safe post-processing
          5. Auto-crop
          
        text_sensitivity:
          "low"    = only obvious watermarks (safe for logos)
          "medium" = moderate (may catch some design text)
          "high"   = remove all text (old aggressive behavior)
        """
        start_time = time.time()

        pil_input = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = pil_input.size
        print(f"  📏 Input: {orig_w}x{orig_h}")

        # Downscale
        max_dim = max(orig_w, orig_h)
        if max_dim > 1500:
            scale = 1500 / max_dim
            new_size = (int(orig_w * scale), int(orig_h * scale))
            pil_input = pil_input.resize(new_size, Image.LANCZOS)
            print(f"  📏 Resized: {new_size[0]}x{new_size[1]}")

        # ── Step 1: Smart text removal (SINGLE pass only) ─────────────
        text_removed_count = 0
        if remove_text:
            print(f"  ── Text Removal (sensitivity={text_sensitivity}) ──")
            img_array = np.array(pil_input)

            text_mask, num_regions = self.text_detector.detect(
                img_array, sensitivity=text_sensitivity
            )
            text_removed_count = num_regions

            if num_regions > 0 and text_mask.sum() > 0:
                print(f"  🧹 Inpainting {num_regions} watermark regions...", end=" ", flush=True)
                t = time.time()
                bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                inpainted_bgr = self.text_inpainter.inpaint(bgr, text_mask)
                img_clean = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
                pil_input = Image.fromarray(img_clean)
                print(f"✓ ({time.time() - t:.1f}s)")
            else:
                print("  📝 No watermarks found — design text preserved ✓")

        # ── Step 2: Smart crop ────────────────────────────────────────
        if remove_bg:
            cropped, was_cropped = self._preprocess_for_small_subject(pil_input)
        else:
            cropped = pil_input

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")

        # ── Step 3: Background removal ────────────────────────────────
        if remove_bg:
            print("  ── Background Removal ──")
            output_bytes = self._remove_background(buf.getvalue())

            pil_result = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            rgba = np.array(pil_result)

            # Wheel detection & post-processing
            print("  🛞 Detecting wheels...")
            wheel_mask = self._detect_wheel_regions(rgba)

            print("  🧹 Cleaning edges...")
            rgba = self._remove_halo(rgba, wheel_mask)
            rgba = self._clean_mask(rgba, wheel_mask)
            rgba = self._remove_ground_shadow(rgba, wheel_mask)
            rgba = self._remove_sky_remnants(rgba)
        else:
            rgba = np.array(cropped.convert("RGBA"))

        # ── NO SECOND PASS — removed to prevent logo damage ──────────

        # ── Step 4: Auto-crop ─────────────────────────────────────────
        if auto_crop and remove_bg:
            rgba = self._auto_crop(rgba, padding_pct=3)

        # ── Output ────────────────────────────────────────────────────
        pil_out = Image.fromarray(rgba, "RGBA")
        out_buf = io.BytesIO()
        pil_out.save(out_buf, format="PNG")
        result_bytes = out_buf.getvalue()

        total = time.time() - start_time
        print(f"  ⏱ Total: {total:.1f}s")

        return result_bytes, total, text_removed_count


pipeline = VisionCleanPipeline()


# ====================================================================
# API ENDPOINTS
# ====================================================================

@app.get("/")
async def root():
    return {
        "service": "VisionClean v3.1",
        "model": pipeline.model_name or "default",
        "features": {
            "background_removal": REMBG_AVAILABLE,
            "text_detection_craft": CRAFT_AVAILABLE,
            "text_detection_fallback": True,
            "text_inpainting": "OpenCV",
            "smart_watermark_filter": True,
        }
    }


@app.get("/status")
async def status():
    return {
        "status": "online",
        "model": pipeline.model_name,
        "craft": CRAFT_AVAILABLE,
        "mode": "CPU"
    }


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    remove_bg: bool = True,
    remove_text: bool = True,
    auto_crop: bool = True,
    text_sensitivity: str = Query(
        default="low",
        description="Text removal sensitivity: low (watermarks only), medium, high (all text)"
    )
):
    """
    Process an image:
    - **remove_bg**: Remove background (BiRefNet)
    - **remove_text**: Remove watermarks (CRAFT + OpenCV)
    - **auto_crop**: Crop tightly to subject
    - **text_sensitivity**: low | medium | high
      - **low** (default): Only removes obvious watermarks. Safe for logos.
      - **medium**: More aggressive. May catch some design text.
      - **high**: Removes ALL detected text (old behavior).
    """
    try:
        # Validate sensitivity
        if text_sensitivity not in ("low", "medium", "high"):
            text_sensitivity = "low"

        print(f"\n{'=' * 60}")
        print(f"📤 {file.filename}")
        print(f"   BG: {remove_bg} | Text: {remove_text} ({text_sensitivity}) | Crop: {auto_crop}")

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(400, "Max 10MB")

        out_bytes, proc_time, text_count = pipeline.process(
            contents, remove_bg=remove_bg, remove_text=remove_text,
            auto_crop=auto_crop, text_sensitivity=text_sensitivity
        )

        ts = int(time.time())
        fname = f"visionclean_{Path(file.filename).stem}_{ts}.png"
        fpath = OUTPUT_DIR / fname

        with open(fpath, "wb") as f:
            f.write(out_bytes)

        print(f"✅ {fname} ({proc_time:.1f}s)\n{'=' * 60}")

        return JSONResponse({
            "success": True,
            "filename": fname,
            "download_url": f"http://127.0.0.1:8000/output/{fname}",
            "time": f"{proc_time:.1f}s",
            "model": pipeline.model_name,
            "text_regions_removed": text_count,
            "text_sensitivity": text_sensitivity,
            "format": "PNG (RGBA transparent)"
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/output/{filename}")
async def download(filename: str):
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path=p, filename=filename, media_type="image/png")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return 


if __name__ == "__main__":
    print(f"\n🌐 API:    http://127.0.0.1:8000")
    print(f"📚 Docs:   http://127.0.0.1:8000/docs")
    print(f"🖼 Upload: http://127.0.0.1:8000/upload")
    print(f"🤖 BG:     {pipeline.model_name or 'default'}")
    print(f"📝 Text:   CRAFT={'✓' if CRAFT_AVAILABLE else '✗(fallback)'} | Smart Filter=✓")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")