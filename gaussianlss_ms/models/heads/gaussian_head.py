"""
Gaussian Head for GaussianLSS MindSpore implementation.

This module implements the head that predicts Gaussian parameters
for 3D object detection and BEV segmentation.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from typing import List, Tuple, Dict, Any


class GaussianHead(nn.Cell):
    """
    Gaussian Head for predicting 3D Gaussian parameters.
    
    This head predicts:
    - Gaussian centers (3D positions)
    - Gaussian covariances (3D shapes)
    - Gaussian opacities (visibility)
    - Gaussian colors (appearance)
    - Semantic labels
    
    Args:
        in_channels (int): Number of input channels
        feat_channels (int): Number of feature channels
        num_gaussians (int): Maximum number of Gaussians to predict
        depth_num (int): Number of depth bins
        num_classes (int): Number of semantic classes
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 feat_channels: int = 256,
                 num_gaussians: int = 1000,
                 depth_num: int = 64,
                 num_classes: int = 2):
        super(GaussianHead, self).__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_gaussians = num_gaussians
        self.depth_num = depth_num
        self.num_classes = num_classes
        
        # Shared feature extraction
        self.shared_conv = nn.SequentialCell([
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU()
        ])
        
        # Gaussian parameter prediction heads
        # Center prediction (x, y, z)
        self.center_head = nn.Conv2d(feat_channels, 3, 1)
        
        # Covariance prediction (6 parameters for 3x3 symmetric matrix)
        self.covariance_head = nn.Conv2d(feat_channels, 6, 1)
        
        # Opacity prediction (1 parameter)
        self.opacity_head = nn.Conv2d(feat_channels, 1, 1)
        
        # Color prediction (3 parameters for RGB)
        self.color_head = nn.Conv2d(feat_channels, 3, 1)
        
        # Depth distribution prediction
        self.depth_head = nn.Conv2d(feat_channels, depth_num, 1)
        
        # Semantic segmentation head
        self.seg_head = nn.Conv2d(feat_channels, num_classes, 1)
        
        # Object detection head (for center heatmap)
        self.heatmap_head = nn.Conv2d(feat_channels, num_classes, 1)
        
        # Offset prediction (for precise localization)
        self.offset_head = nn.Conv2d(feat_channels, 2, 1)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights of the head."""
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.XavierUniform(), m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Zero(), m.bias.shape, m.bias.dtype))
        
        # Special initialization for heatmap head (bias initialization)
        if self.heatmap_head.bias is not None:
            bias_init = ms.common.initializer.initializer(
                ms.common.initializer.Constant(-2.19), self.heatmap_head.bias.shape, self.heatmap_head.bias.dtype)
            self.heatmap_head.bias.set_data(bias_init)
    
    def construct(self, x: ms.Tensor) -> Dict[str, ms.Tensor]:
        """
        Forward pass of the Gaussian head.
        
        Args:
            x (ms.Tensor): Input feature tensor [B, C, H, W]
            
        Returns:
            Dict[str, ms.Tensor]: Dictionary containing predicted parameters
        """
        # Shared feature extraction
        feat = self.shared_conv(x)
        
        # Predict Gaussian parameters
        centers = self.center_head(feat)  # [B, 3, H, W]
        covariances = self.covariance_head(feat)  # [B, 6, H, W]
        opacities = self.opacity_head(feat)  # [B, 1, H, W]
        colors = self.color_head(feat)  # [B, 3, H, W]
        
        # Predict depth distribution
        depth_dist = self.depth_head(feat)  # [B, depth_num, H, W]
        
        # Predict segmentation
        seg_logits = self.seg_head(feat)  # [B, num_classes, H, W]
        
        # Predict detection heatmap and offsets
        heatmap = self.heatmap_head(feat)  # [B, num_classes, H, W]
        offsets = self.offset_head(feat)  # [B, 2, H, W]
        
        # Apply activations
        opacities = ops.Sigmoid()(opacities)
        colors = ops.Sigmoid()(colors)
        depth_dist = ops.Softmax(axis=1)(depth_dist)
        heatmap = ops.Sigmoid()(heatmap)
        
        return {
            'centers': centers,
            'covariances': covariances,
            'opacities': opacities,
            'colors': colors,
            'depth_dist': depth_dist,
            'seg_logits': seg_logits,
            'heatmap': heatmap,
            'offsets': offsets
        }
    
    def get_gaussians(self, predictions: Dict[str, ms.Tensor], 
                     threshold: float = 0.1) -> List[Dict[str, ms.Tensor]]:
        """
        Extract Gaussian parameters from predictions.
        
        Args:
            predictions (Dict[str, ms.Tensor]): Raw predictions from the head
            threshold (float): Opacity threshold for filtering Gaussians
            
        Returns:
            List[Dict[str, ms.Tensor]]: List of Gaussian parameters for each batch
        """
        batch_size = predictions['opacities'].shape[0]
        gaussians_list = []
        
        for b in range(batch_size):
            # Get predictions for this batch
            opacity = predictions['opacities'][b, 0]  # [H, W]
            
            # Find valid Gaussian locations
            valid_mask = opacity > threshold
            valid_indices = ops.nonzero(valid_mask)  # [N, 2]
            
            if valid_indices.shape[0] == 0:
                gaussians_list.append({
                    'centers': ops.zeros((0, 3), ms.float32),
                    'covariances': ops.zeros((0, 6), ms.float32),
                    'opacities': ops.zeros((0,), ms.float32),
                    'colors': ops.zeros((0, 3), ms.float32)
                })
                continue
            
            # Extract parameters for valid locations
            h_indices, w_indices = valid_indices[:, 0], valid_indices[:, 1]
            
            centers = predictions['centers'][b, :, h_indices, w_indices].T  # [N, 3]
            covariances = predictions['covariances'][b, :, h_indices, w_indices].T  # [N, 6]
            opacities = predictions['opacities'][b, 0, h_indices, w_indices]  # [N]
            colors = predictions['colors'][b, :, h_indices, w_indices].T  # [N, 3]
            
            gaussians_list.append({
                'centers': centers,
                'covariances': covariances,
                'opacities': opacities,
                'colors': colors
            })
        
        return gaussians_list