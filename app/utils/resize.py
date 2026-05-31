import cv2
import numpy as np


def pad_to_square(image, pad_color=(0, 0, 0, 0)):
    """Pad image to square shape."""
    h, w = image.shape[:2]

    if h == w:
        return image

    size = max(h, w)
    padded = np.full((size, size, image.shape[2]), pad_color, dtype=image.dtype)

    pad_h = (size - h) // 2
    pad_w = (size - w) // 2

    padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
    return padded


def resize_with_padding(image, target_size, pad_color=(0, 0, 0, 0)):
    """Resize image with padding to maintain aspect ratio."""
    h, w = image.shape[:2]

    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.full((target_size, target_size, image.shape[2]), pad_color, dtype=image.dtype)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, (pad_h, pad_w, scale)


def remove_padding(image, padding_info):
    """Remove padding added by resize_with_padding."""
    pad_h, pad_w, scale = padding_info
    h, w = image.shape[:2]

    unpadded_h = int(h - 2 * pad_h)
    unpadded_w = int(w - 2 * pad_w)

    if unpadded_h <= 0 or unpadded_w <= 0:
        return image

    unpadded = image[pad_h:pad_h + unpadded_h, pad_w:pad_w + unpadded_w]

    if scale != 1.0:
        original_h = int(unpadded_h / scale)
        original_w = int(unpadded_w / scale)
        unpadded = cv2.resize(unpadded, (original_w, original_h),
                              interpolation=cv2.INTER_CUBIC)

    return unpadded