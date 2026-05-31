import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path


def load_image(image_path):
    """Load image from file path or bytes — preserves alpha if present."""
    if isinstance(image_path, (str, Path)):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif isinstance(image_path, bytes):
        nparr = np.frombuffer(image_path, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Could not decode image bytes")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input must be file path or bytes")

    return image


def save_image(image_array, output_path, format="png", background="transparent"):
    """Save image — forces PNG when alpha channel is present."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(image_array.shape) == 2:
        pil_image = Image.fromarray(image_array, 'L')

    elif image_array.shape[2] == 4:
        pil_image = Image.fromarray(image_array, 'RGBA')
        format = "png"

    elif image_array.shape[2] == 3:
        if background == "transparent":
            alpha = np.ones(
                (image_array.shape[0], image_array.shape[1], 1),
                dtype=np.uint8
            ) * 255
            image_array = np.concatenate([image_array, alpha], axis=2)
            pil_image = Image.fromarray(image_array, 'RGBA')
            format = "png"
        else:
            pil_image = Image.fromarray(image_array, 'RGB')
    else:
        pil_image = Image.fromarray(image_array)

    if format.lower() in ("jpg", "jpeg"):
        pil_image = pil_image.convert("RGB")

    if format.lower() == "png":
        output_path = output_path.with_suffix('.png')
    elif format.lower() in ("jpg", "jpeg"):
        output_path = output_path.with_suffix('.jpg')

    pil_image.save(str(output_path), format=format.upper())
    return str(output_path)


def bytes_to_image(image_bytes):
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def image_to_bytes(image_array, format="PNG"):
    """Convert image array to bytes."""
    if len(image_array.shape) == 2:
        pil_image = Image.fromarray(image_array, 'L')
    elif image_array.shape[2] == 4:
        pil_image = Image.fromarray(image_array, 'RGBA')
    else:
        pil_image = Image.fromarray(image_array, 'RGB')

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def resize_image(image, max_size=1024):
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)