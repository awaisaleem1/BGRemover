import cv2
import numpy as np
from pathlib import Path
import time
from typing import Optional, Tuple

from app.pipeline.text_detection import TextDetector
from app.pipeline.inpainting import Inpainter
from app.pipeline.bg_removal import BackgroundRemover
from app.utils.image_io import load_image, save_image, resize_image
from app.config import OUTPUT_DIR, OUTPUT_FORMAT, OUTPUT_BACKGROUND


class Pipeline:
    def __init__(self,
                 text_detector: Optional[TextDetector] = None,
                 inpainter: Optional[Inpainter] = None,
                 bg_remover: Optional[BackgroundRemover] = None):
        self.text_detector = text_detector or TextDetector()
        self.inpainter = inpainter or Inpainter()
        self.bg_remover = bg_remover or BackgroundRemover()

    def process(self,
                input_path: str,
                output_path: Optional[str] = None,
                save_intermediate: bool = False,
                remove_text: bool = True,
                remove_bg: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Process image through the complete pipeline.
        """
        print(f"Loading image: {input_path}")
        image = load_image(input_path)
        original_shape = image.shape
        print(f"Original image shape: {original_shape}")

        image = resize_image(image, max_size=1024)
        processing_info = {
            'original_shape': original_shape,
            'processing_shape': image.shape,
            'steps': [],
            'timings': {}
        }

        # Step 1: Text Detection and Removal
        if remove_text:
            print("Step 1: Detecting text...")
            start_time = time.time()

            try:
                text_mask, polygons = self.text_detector.detect_text(image)
                processing_info['text_regions'] = len(polygons)
                processing_info['timings']['text_detection'] = time.time() - start_time
                processing_info['steps'].append('text_detection')

                if save_intermediate:
                    vis_path = OUTPUT_DIR / "intermediate_text_detection.png"
                    self.text_detector.visualize_detection(image, text_mask, str(vis_path))

                print(f"Step 2: Removing text ({processing_info['text_regions']} regions)...")
                start_time = time.time()

                if text_mask.sum() > 0:
                    image = self.inpainter.remove_text(image, text_mask)

                processing_info['timings']['text_removal'] = time.time() - start_time
                processing_info['steps'].append('text_removal')

                if save_intermediate:
                    text_removed_path = OUTPUT_DIR / "intermediate_text_removed.png"
                    save_image(image, str(text_removed_path))

            except Exception as e:
                print(f"Text processing failed: {e}")
                processing_info['text_error'] = str(e)

        # Step 2: Background Removal
        if remove_bg:
            print("Step 3: Removing background...")
            start_time = time.time()

            try:
                image_with_alpha, bg_mask = self.bg_remover.remove_background_auto(image)
                processing_info['timings']['bg_removal'] = time.time() - start_time
                processing_info['steps'].append('bg_removal')

                if save_intermediate:
                    bg_mask_path = OUTPUT_DIR / "intermediate_bg_mask.png"
                    save_image(bg_mask, str(bg_mask_path))

                image = image_with_alpha

            except Exception as e:
                print(f"Background removal failed: {e}")
                processing_info['bg_error'] = str(e)

        # Save result
        if output_path:
            print(f"Saving result to: {output_path}")
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CRITICAL: If image has alpha, force PNG format
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            fmt = OUTPUT_FORMAT
            if len(image.shape) == 3 and image.shape[2] == 4:
                fmt = "png"  # Force PNG for transparency

            save_image(image, output_path, format=fmt, background=OUTPUT_BACKGROUND)

        processing_info['total_time'] = sum(processing_info['timings'].values())
        processing_info['final_shape'] = image.shape

        print(f"Pipeline completed in {processing_info['total_time']:.2f} seconds")

        return image, processing_info

    def process_batch(self,
                      input_dir: str,
                      output_dir: str,
                      extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed = 0
        failed = 0

        for img_file in input_path.iterdir():
            if img_file.suffix.lower() in extensions:
                try:
                    # Always output as PNG for transparency
                    output_file = output_path / f"{img_file.stem}_cleaned.png"
                    self.process(str(img_file), str(output_file))
                    processed += 1
                    print(f"Processed: {img_file.name}")
                except Exception as e:
                    print(f"Failed to process {img_file.name}: {e}")
                    failed += 1

        print(f"Batch complete: {processed} successful, {failed} failed")
        return processed, failed

    def __del__(self):
        del self.text_detector
        del self.inpainter
        del self.bg_remover