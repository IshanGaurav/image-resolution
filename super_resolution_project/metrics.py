"""
Image quality metrics for super-resolution evaluation.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(hr_image: np.ndarray, sr_image: np.ndarray) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio between the high-resolution
    ground truth and the super-resolved output.

    Both images must have the same shape and dtype uint8.

    Args:
        hr_image: Ground-truth HR image (H, W, 3), uint8.
        sr_image: Super-resolved image     (H, W, 3), uint8.

    Returns:
        PSNR value in dB (higher is better).
    """
    if hr_image.shape != sr_image.shape:
        raise ValueError(
            f"Shape mismatch: HR {hr_image.shape} vs SR {sr_image.shape}"
        )
    return float(
        peak_signal_noise_ratio(hr_image, sr_image, data_range=255)
    )


def calculate_ssim(hr_image: np.ndarray, sr_image: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between the
    high-resolution ground truth and the super-resolved output.

    Both images must have the same shape and dtype uint8.

    Args:
        hr_image: Ground-truth HR image (H, W, 3), uint8.
        sr_image: Super-resolved image     (H, W, 3), uint8.

    Returns:
        SSIM value in [0, 1] (higher is better).
    """
    if hr_image.shape != sr_image.shape:
        raise ValueError(
            f"Shape mismatch: HR {hr_image.shape} vs SR {sr_image.shape}"
        )
    return float(
        structural_similarity(
            hr_image,
            sr_image,
            data_range=255,
            channel_axis=2,      # colour images  (H, W, C)
        )
    )
