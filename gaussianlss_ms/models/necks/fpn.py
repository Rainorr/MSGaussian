"""
Feature Pyramid Network (FPN) implementation for MindSpore.

This module implements a Feature Pyramid Network that combines multi-scale
features from the backbone network.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from typing import List, Tuple


class FPN(nn.Cell):
    """
    Feature Pyramid Network (FPN) for multi-scale feature fusion.
    
    Args:
        in_channels (List[int]): Number of input channels for each level
        out_channels (int): Number of output channels for all levels
        num_outs (int): Number of output feature levels
        start_level (int): Index of the start input backbone level
        end_level (int): Index of the end input backbone level
        add_extra_convs (bool): Whether to add extra conv layers
        relu_before_extra_convs (bool): Whether to add relu before extra convs
    """
    
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: bool = False,
                 relu_before_extra_convs: bool = False):
        super(FPN, self).__init__()
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        # Build lateral convs
        self.lateral_convs = nn.CellList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bias=True,
                pad_mode='pad'
            )
            self.lateral_convs.append(l_conv)
        
        # Build fpn convs
        self.fpn_convs = nn.CellList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                pad_mode='pad',
                padding=1,
                has_bias=True
            )
            self.fpn_convs.append(fpn_conv)
        
        # Add extra convs if necessary
        if self.add_extra_convs:
            extra_levels = num_outs - self.backbone_end_level + self.start_level
            if self.add_extra_convs == 'on_input':
                extra_source = in_channels[self.backbone_end_level - 1]
            elif self.add_extra_convs == 'on_lateral':
                extra_source = out_channels
            elif self.add_extra_convs == 'on_output':
                extra_source = out_channels
            else:
                raise ValueError(
                    f'add_extra_convs should be "on_input", "on_lateral" '
                    f'or "on_output", got {self.add_extra_convs}')
            
            self.extra_convs = nn.CellList()
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = extra_source
                else:
                    in_channels = out_channels
                extra_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    pad_mode='pad',
                    padding=1,
                    has_bias=True
                )
                self.extra_convs.append(extra_conv)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.XavierUniform(), m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Zero(), m.bias.shape, m.bias.dtype))
    
    def construct(self, inputs: Tuple[ms.Tensor]) -> Tuple[ms.Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            # it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += ops.ResizeNearestNeighbor(
                    size=(laterals[i - 1].shape[2], laterals[i - 1].shape[3])
                )(laterals[i])
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += ops.ResizeNearestNeighbor(
                    size=prev_shape
                )(laterals[i])
        
        # Build outputs
        # Part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # Part 2: add extra levels
        if self.num_outs > len(outs):
            # Use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(ops.MaxPool2d(kernel_size=1, stride=2)(outs[-1]))
            # Add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](ops.ReLU()(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))
        
        return tuple(outs)
    
    @property
    def upsample_cfg(self):
        """Upsample configuration for FPN."""
        return {'mode': 'nearest'}