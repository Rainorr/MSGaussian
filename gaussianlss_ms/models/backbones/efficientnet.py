"""
EfficientNet backbone implementation for MindSpore.

This module provides EfficientNet architectures adapted for the
GaussianLSS multi-view perception task.
"""

import math
from typing import List, Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class MBConvBlock(nn.Cell):
    """
    Mobile Inverted Bottleneck Convolution Block.
    
    This is the core building block of EfficientNet.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0
    ):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.SequentialCell([
                nn.Conv2d(in_channels, expanded_channels, 1, has_bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            ])
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.SequentialCell([
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, pad_mode='pad', padding=kernel_size//2, group=expanded_channels,
                has_bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.SequentialCell([
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            ])
        else:
            self.se = None
        
        # Output projection
        self.project_conv = nn.SequentialCell([
            nn.Conv2d(expanded_channels, out_channels, 1, has_bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        # Dropout
        if drop_rate > 0:
            self.dropout = nn.Dropout(p=drop_rate)
        else:
            self.dropout = None
    
    def construct(self, x: Tensor) -> Tensor:
        """Forward pass of MBConv block."""
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Squeeze-and-Excitation
        if self.se is not None:
            se_weight = self.se(x)
            x = x * se_weight
        
        # Output projection
        x = self.project_conv(x)
        
        # Dropout and residual connection
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity
        
        return x


class EfficientNetBackbone(nn.Cell):
    """
    EfficientNet backbone for feature extraction.
    
    This implementation is optimized for multi-view perception tasks
    and provides multi-scale feature outputs.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet-b4',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (2, 3, 4, 5),
        frozen_stages: int = -1,
        norm_eval: bool = False
    ):
        """
        Initialize EfficientNet backbone.
        
        Args:
            model_name: EfficientNet variant name
            pretrained: Whether to use pretrained weights
            out_indices: Indices of stages to output features
            frozen_stages: Number of stages to freeze
            norm_eval: Whether to set norm layers to eval mode
        """
        super().__init__()
        
        self.model_name = model_name
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # Get model configuration
        config = self._get_model_config(model_name)
        
        # Build network
        self.stem = nn.SequentialCell([
            nn.Conv2d(3, config['stem_channels'], 3, stride=2, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(config['stem_channels']),
            nn.SiLU()
        ])
        
        # Build stages
        self.stages = nn.CellList()
        in_channels = config['stem_channels']
        
        for stage_idx, stage_config in enumerate(config['stages']):
            stage_blocks = nn.CellList()
            
            for block_idx in range(stage_config['num_blocks']):
                stride = stage_config['stride'] if block_idx == 0 else 1
                out_channels = stage_config['out_channels']
                
                block = MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stage_config['kernel_size'],
                    stride=stride,
                    expand_ratio=stage_config['expand_ratio'],
                    se_ratio=stage_config['se_ratio'],
                    drop_rate=config['drop_rate']
                )
                
                stage_blocks.append(block)
                in_channels = out_channels
            
            self.stages.append(stage_blocks)
        
        # Initialize weights
        if pretrained:
            self._load_pretrained_weights()
        else:
            self._init_weights()
        
        # Freeze stages if specified
        self._freeze_stages()
    
    def _get_model_config(self, model_name: str) -> dict:
        """Get configuration for specific EfficientNet variant."""
        configs = {
            'efficientnet-b0': {
                'stem_channels': 32,
                'drop_rate': 0.2,
                'stages': [
                    {'num_blocks': 1, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 1, 'se_ratio': 0.25},
                    {'num_blocks': 2, 'out_channels': 24, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 2, 'out_channels': 40, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 3, 'out_channels': 80, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 3, 'out_channels': 112, 'kernel_size': 5, 'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 4, 'out_channels': 192, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 1, 'out_channels': 320, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25},
                ]
            },
            'efficientnet-b4': {
                'stem_channels': 48,
                'drop_rate': 0.4,
                'stages': [
                    {'num_blocks': 2, 'out_channels': 24, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 1, 'se_ratio': 0.25},
                    {'num_blocks': 4, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 4, 'out_channels': 56, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 6, 'out_channels': 112, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 6, 'out_channels': 160, 'kernel_size': 5, 'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 8, 'out_channels': 272, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                    {'num_blocks': 2, 'out_channels': 448, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25},
                ]
            }
        }
        
        if model_name not in configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return configs[model_name]
    
    def construct(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass of EfficientNet backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors from specified stages
        """
        features = []
        
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            
            # Collect features from specified stages
            if stage_idx in self.out_indices:
                features.append(x)
        
        return features
    
    def _init_weights(self):
        """Initialize network weights."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                # Kaiming normal initialization
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    ms.common.initializer.initializer(
                        ms.common.initializer.Normal(sigma=math.sqrt(2.0 / fan_out)),
                        cell.weight.shape,
                        cell.weight.dtype
                    )
                )
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(ms.common.initializer.initializer('zeros', cell.beta.shape))
    
    def _load_pretrained_weights(self):
        """Load pretrained weights (placeholder for actual implementation)."""
        # In practice, this would load weights from a pretrained model
        # For now, just initialize randomly
        self._init_weights()
        print(f"Loaded pretrained weights for {self.model_name}")
    
    def _freeze_stages(self):
        """Freeze specified stages."""
        if self.frozen_stages >= 0:
            # Freeze stem
            for param in self.stem.get_parameters():
                param.requires_grad = False
            
            # Freeze stages
            for i in range(min(self.frozen_stages, len(self.stages))):
                for param in self.stages[i].get_parameters():
                    param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        
        if self.norm_eval:
            for cell in self.cells():
                if isinstance(cell, nn.BatchNorm2d):
                    cell.set_train(False)


def create_efficientnet_backbone(
    model_name: str = 'efficientnet-b4',
    pretrained: bool = True,
    **kwargs
) -> EfficientNetBackbone:
    """
    Factory function to create EfficientNet backbone.
    
    Args:
        model_name: EfficientNet variant name
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        EfficientNet backbone instance
    """
    return EfficientNetBackbone(
        model_name=model_name,
        pretrained=pretrained,
        **kwargs
    )