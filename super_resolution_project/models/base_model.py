"""
Abstract base class for all super-resolution models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Every SR model must inherit from this class and implement ``predict``.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Up-scale the given low-resolution image.

        Args:
            image: LR input as np.ndarray (H, W, 3), uint8.

        Returns:
            SR output as np.ndarray (H', W', 3), uint8.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable model name (defaults to class name)."""
        return self.__class__.__name__
