"""
Utility functions for image loading, saving, degradation, and color-space conversion.
"""

import numpy as np
from PIL import Image
from typing import Union


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a NumPy array (RGB, uint8).

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        np.ndarray of shape (H, W, 3) with dtype uint8 in RGB order.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save a NumPy array (RGB, uint8) to disk.

    Args:
        image: np.ndarray of shape (H, W, 3) with dtype uint8.
        path:  Destination file path (extension determines format).
    """
    img = numpy_to_pil(image)
    img.save(path)


def degrade_image(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Down-sample an image by *scale_factor* using bicubic interpolation to
    simulate a low-resolution capture.

    Args:
        image:        HR image as np.ndarray (H, W, 3), uint8.
        scale_factor: Integer down-sampling factor (e.g. 2, 4, 8).

    Returns:
        LR image as np.ndarray with shape (H // scale_factor, W // scale_factor, 3).
    """
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")

    pil_img = numpy_to_pil(image)
    h, w = image.shape[:2]
    new_size = (w // scale_factor, h // scale_factor)  # PIL uses (W, H)
    lr_img = pil_img.resize(new_size, Image.BICUBIC)
    return pil_to_numpy(lr_img)


# ---------- Conversion helpers ---------- #

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert an RGB uint8 NumPy array to a PIL Image.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to an RGB uint8 NumPy array.
    """
    return np.array(image.convert("RGB"), dtype=np.uint8)


def convert_color_space(
    image: np.ndarray,
    target: str = "ycbcr",
) -> np.ndarray:
    """
    Convert between RGB and YCbCr colour spaces.

    Args:
        image:  np.ndarray (H, W, 3), uint8.
        target: ``"ycbcr"`` to go from RGB → YCbCr, or ``"rgb"`` for the reverse.

    Returns:
        np.ndarray (H, W, 3) in the target colour space, float64 for YCbCr
        and uint8 for RGB.
    """
    target = target.lower()

    if target == "ycbcr":
        pil_img = numpy_to_pil(image)
        ycbcr = pil_img.convert("YCbCr")
        return np.array(ycbcr, dtype=np.float64)

    elif target == "rgb":
        # Expect float64 YCbCr input
        ycbcr_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(ycbcr_uint8, mode="YCbCr")
        rgb = pil_img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)

    else:
        raise ValueError(f"Unsupported target colour space: {target!r}")
