"""
Traditional interpolation-based upscaling models and a restricted stacking model.
"""

import numpy as np
from PIL import Image
from typing import Union

from .base_model import BaseModel


# ------------------------------------------------------------------ #
#  Single-stage interpolation models
# ------------------------------------------------------------------ #

class NearestModel(BaseModel):
    """Upscale using nearest-neighbour interpolation."""

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor

    def predict(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image, mode="RGB")
        h, w = image.shape[:2]
        new_size = (w * self.scale_factor, h * self.scale_factor)
        upscaled = pil_img.resize(new_size, Image.NEAREST)
        return np.array(upscaled, dtype=np.uint8)


class BilinearModel(BaseModel):
    """Upscale using bilinear interpolation."""

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor

    def predict(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image, mode="RGB")
        h, w = image.shape[:2]
        new_size = (w * self.scale_factor, h * self.scale_factor)
        upscaled = pil_img.resize(new_size, Image.BILINEAR)
        return np.array(upscaled, dtype=np.uint8)


class BicubicModel(BaseModel):
    """Upscale using bicubic interpolation."""

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor

    def predict(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image, mode="RGB")
        h, w = image.shape[:2]
        new_size = (w * self.scale_factor, h * self.scale_factor)
        upscaled = pil_img.resize(new_size, Image.BICUBIC)
        return np.array(upscaled, dtype=np.uint8)


# ------------------------------------------------------------------ #
#  Two-stage stacked interpolation model
# ------------------------------------------------------------------ #

# Only traditional interpolation types are allowed in the stack.
_ALLOWED_TYPES = (NearestModel, BilinearModel, BicubicModel)


class StackedInterpolationModel(BaseModel):
    """
    Sequentially applies two traditional interpolation models.

    The first model upscales by ``scale1`` and the second model upscales
    the intermediate result by ``scale2``, giving an overall factor of
    ``scale1 × scale2``.

    Only :class:`NearestModel`, :class:`BilinearModel`, and
    :class:`BicubicModel` are accepted — PyTorch / deep-learning models
    are explicitly rejected.
    """

    def __init__(
        self,
        model1: BaseModel,
        scale1: int,
        model2: BaseModel,
        scale2: int,
    ):
        # --- Validate that both models are traditional types ----------
        for idx, m in enumerate((model1, model2), start=1):
            if not isinstance(m, _ALLOWED_TYPES):
                raise TypeError(
                    f"Stage {idx} model must be one of "
                    f"{[c.__name__ for c in _ALLOWED_TYPES]}, "
                    f"got {type(m).__name__!r}."
                )

        # Override each model's scale factor to match the caller's intent
        model1.scale_factor = scale1
        model2.scale_factor = scale2

        self.model1 = model1
        self.model2 = model2
        self.scale1 = scale1
        self.scale2 = scale2

    # ---- BaseModel interface ----------------------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        intermediate = self.model1.predict(image)
        return self.model2.predict(intermediate)

    @property
    def name(self) -> str:
        return (
            f"Stacked({self.model1.name}×{self.scale1} → "
            f"{self.model2.name}×{self.scale2})"
        )
