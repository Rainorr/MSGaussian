"""
ResNet backbone implementation for MindSpore.

This module provides ResNet architectures as an alternative backbone
for the GaussianLSS model.
"""

from typing import List, Tuple

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class BasicBlock(nn.Cell):
    """Basic ResNet block."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Cell = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def construct(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Cell):
    """Bottleneck ResNet block."""
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Cell = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
    
    def construct(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetBackbone(nn.Cell):
    """
    ResNet backbone for feature extraction.
    """
    
    def __init__(
        self,
        depth: int = 50,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1,
        norm_eval: bool = False
    ):
        """
        Initialize ResNet backbone.
        
        Args:
            depth: ResNet depth (18, 34, 50, 101, 152)
            out_indices: Indices of stages to output features
            frozen_stages: Number of stages to freeze
            norm_eval: Whether to set norm layers to eval mode
        """
        super().__init__()
        
        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # Get architecture configuration
        if depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        self.in_channels = 64
        
        # Stem layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Initialize weights
        self._init_weights()
        
        # Freeze stages
        self._freeze_stages()
    
    def _make_layer(
        self,
        block: nn.Cell,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.SequentialCell:
        """Create a ResNet layer."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    1,
                    stride=stride,
                    has_bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            ])
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.SequentialCell(layers)
    
    def construct(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass of ResNet backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors from specified stages
        """
        features = []
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if 0 in self.out_indices:
            features.append(x)
        
        x = self.maxpool(x)
        
        # Stage 1
        x = self.layer1(x)
        if 1 in self.out_indices:
            features.append(x)
        
        # Stage 2
        x = self.layer2(x)
        if 2 in self.out_indices:
            features.append(x)
        
        # Stage 3
        x = self.layer3(x)
        if 3 in self.out_indices:
            features.append(x)
        
        # Stage 4
        x = self.layer4(x)
        if 4 in self.out_indices:
            features.append(x)
        
        return features
    
    def _init_weights(self):
        """Initialize network weights."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    ms.common.initializer.initializer(
                        ms.common.initializer.HeNormal(),
                        cell.weight.shape,
                        cell.weight.dtype
                    )
                )
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(ms.common.initializer.initializer('zeros', cell.beta.shape))
    
    def _freeze_stages(self):
        """Freeze specified stages."""
        if self.frozen_stages >= 0:
            # Freeze stem
            for param in [self.conv1, self.bn1]:
                for p in param.get_parameters():
                    p.requires_grad = False
            
            # Freeze stages
            stages = [self.layer1, self.layer2, self.layer3, self.layer4]
            for i in range(min(self.frozen_stages, len(stages))):
                for param in stages[i].get_parameters():
                    param.requires_grad = False


def create_resnet_backbone(depth: int = 50, **kwargs) -> ResNetBackbone:
    """
    Factory function to create ResNet backbone.
    
    Args:
        depth: ResNet depth
        **kwargs: Additional arguments
        
    Returns:
        ResNet backbone instance
    """
    return ResNetBackbone(depth=depth, **kwargs)