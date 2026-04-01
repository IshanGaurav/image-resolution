"""
Model registry — re-exports every model class for convenient imports.

Usage::

    from models import BicubicModel, SRCNNModel
"""

from .base_model import BaseModel
from .interpolation import (
    NearestModel,
    BilinearModel,
    BicubicModel,
    StackedInterpolationModel,
)
from .srcnn import SRCNNModel
from .vdsr import VDSRModel

__all__ = [
    "BaseModel",
    "NearestModel",
    "BilinearModel",
    "BicubicModel",
    "StackedInterpolationModel",
    "SRCNNModel",
    "VDSRModel",
]
