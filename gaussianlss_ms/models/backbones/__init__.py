"""
Backbone networks for GaussianLSS MindSpore implementation.

This module provides various backbone architectures for feature extraction
from multi-view camera images.
"""

from .efficientnet import EfficientNetBackbone, create_efficientnet_backbone

# Create aliases for convenience
EfficientNet = EfficientNetBackbone

__all__ = [
    "EfficientNetBackbone",
    "EfficientNet",
    "create_efficientnet_backbone"
]