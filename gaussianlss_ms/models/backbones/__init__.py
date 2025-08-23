"""
Backbone networks for GaussianLSS MindSpore implementation.

This module provides various backbone architectures for feature extraction
from multi-view camera images.
"""

from .efficientnet import EfficientNetBackbone
from .resnet import ResNetBackbone

__all__ = [
    "EfficientNetBackbone",
    "ResNetBackbone"
]