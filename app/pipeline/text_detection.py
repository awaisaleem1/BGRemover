import cv2
import numpy as np
from pathlib import Path
import sys

from app.config import CRAFT_MODEL_PATH
from app.utils.image_io import resize_image
from app.utils.mask_utils import create_mask_from_polygons, dilate_mask

class TextDetector:
    def __init__(self, model_path=CRAFT_MODEL_PATH, device="auto"):
        """
        Initialize CRAFT text detector
        """
        self.model_path = model_path
        
        # Set device
        from app.config import get_device
        self.device = get_device(device)
        
        # Initialize CRAFT
        self.craft = None
        self._load_model()
        
    def _load_model(self):
        """Load CRAFT model"""
        try:
            # Try to import CRAFT
            try:
                from craft_text_detector import Craft
            except ImportError:
                print("CRAFT not installed. Please install with: pip install craft-text-detector")
                print("Using fallback text detection method...")
                self.craft = None
                return
            
            # Initialize CRAFT
            self.craft = Craft(
                output_dir=None,
                crop_type="poly",
                model_path=str(self.model_path),
                rect_th=0.7,
                link_th=0.4,
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
                canvas_size=1280,
                mag_ratio=1.5,
                device=self.device if self.device == "cuda" else "cpu"
            )
            
        except Exception as e:
            print(f"Error loading CRAFT model: {e}")
            self.craft = None
    
    def detect_text(self, image):
        """
        Detect text regions in image
        Returns binary mask where text regions are white (255)
        """
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize if too large
        image = resize_image(image, max_size=2048)
        
        # If CRAFT is not available, use simple edge-based detection
        if self.craft is None:
            return self._fallback_text_detection(image), []
        
        try:
            # Detect text using CRAFT
            prediction_result = self.craft.detect_text(image)
            
            # Get text regions
            if hasattr(prediction_result, 'polys'):
                polygons = prediction_result.polys
            else:
                polygons = []
            
            # Create mask from polygons
            mask = create_mask_from_polygons(polygons, image.shape)
            
            # Dilate mask slightly to ensure complete text coverage
            mask = dilate_mask(mask, kernel_size=3, iterations=1)
            
            # Apply Gaussian blur for smoother edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            return mask, polygons
            
        except Exception as e:
            print(f"CRAFT text detection failed: {e}")
            print("Using fallback text detection...")
            return self._fallback_text_detection(image), []
    
    def _fallback_text_detection(self, image):
        """Fallback text detection using simple edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Filter contours by area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area for text
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Dilate to cover text better
        mask = dilate_mask(mask, kernel_size=3, iterations=2)
        
        return mask
    
    def visualize_detection(self, image, mask, output_path=None):
        """
        Visualize text detection results
        """
        # Create visualization
        vis_image = image.copy()
        
        # Overlay mask
        overlay = vis_image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red for text regions
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image
    
    def __del__(self):
        """Clean up"""
        if hasattr(self, 'craft') and self.craft is not None:
            del self.craft