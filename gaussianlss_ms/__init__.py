"""
GaussianLSS MindSpore Implementation

A MindSpore implementation of GaussianLSS for 3D object detection and BEV segmentation
using 3D Gaussian Splatting from multi-view camera images.
"""

__version__ = "1.0.0"
__author__ = "GaussianLSS MindSpore Team"

from . import data
from . import models
from . import losses
from . import metrics
from . import utils

__all__ = [
    "data",
    "models", 
    "losses",
    "metrics",
    "utils"
]