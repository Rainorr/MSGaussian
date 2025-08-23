"""
Main loss function for GaussianLSS model.

This module implements the composite loss function used for training
the GaussianLSS model, including segmentation and detection losses.
"""

from typing import Dict, Any, Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .focal_loss import FocalLoss
from .smooth_l1_loss import SmoothL1Loss


class GaussianLSSLoss(nn.Cell):
    """
    Composite loss function for GaussianLSS model.
    
    This loss combines multiple components:
    - BEV segmentation loss (focal loss)
    - Center point detection loss (focal loss)
    - Offset regression loss (smooth L1)
    - Visibility classification loss (cross entropy)
    """
    
    def __init__(
        self,
        seg_weight: float = 1.0,
        center_weight: float = 1.0,
        offset_weight: float = 0.5,
        visibility_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0,
        **kwargs
    ):
        """
        Initialize GaussianLSS loss function.
        
        Args:
            seg_weight: Weight for segmentation loss
            center_weight: Weight for center point detection loss
            offset_weight: Weight for offset regression loss
            visibility_weight: Weight for visibility classification loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            smooth_l1_beta: Beta parameter for smooth L1 loss
        """
        super().__init__()
        
        self.seg_weight = seg_weight
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        self.visibility_weight = visibility_weight
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.smooth_l1_loss = SmoothL1Loss(beta=smooth_l1_beta)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        
        # Operations
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
    
    def construct(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Compute composite loss.
        
        Args:
            predictions: Model predictions containing:
                - vehicle_seg: Vehicle segmentation predictions [B, H, W]
                - vehicle_center: Vehicle center predictions [B, H, W]
                - vehicle_offset: Vehicle offset predictions [B, H, W, 2]
                - ped_seg: Pedestrian segmentation predictions [B, H, W]
                - ped_center: Pedestrian center predictions [B, H, W]
                - ped_offset: Pedestrian offset predictions [B, H, W, 2]
            targets: Ground truth targets with same structure
            
        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Vehicle losses
        if 'vehicle_seg' in predictions and 'vehicle' in targets:
            vehicle_seg_loss = self.compute_segmentation_loss(
                predictions['vehicle_seg'],
                targets['vehicle']
            )
            losses['vehicle_seg_loss'] = vehicle_seg_loss
            total_loss += self.seg_weight * vehicle_seg_loss
        
        if 'vehicle_center' in predictions and 'vehicle_center' in targets:
            vehicle_center_loss = self.compute_center_loss(
                predictions['vehicle_center'],
                targets['vehicle_center']
            )
            losses['vehicle_center_loss'] = vehicle_center_loss
            total_loss += self.center_weight * vehicle_center_loss
        
        if 'vehicle_offset' in predictions and 'vehicle_offset' in targets:
            vehicle_offset_loss = self.compute_offset_loss(
                predictions['vehicle_offset'],
                targets['vehicle_offset'],
                targets.get('vehicle', None)
            )
            losses['vehicle_offset_loss'] = vehicle_offset_loss
            total_loss += self.offset_weight * vehicle_offset_loss
        
        # Pedestrian losses
        if 'ped_seg' in predictions and 'ped' in targets:
            ped_seg_loss = self.compute_segmentation_loss(
                predictions['ped_seg'],
                targets['ped']
            )
            losses['ped_seg_loss'] = ped_seg_loss
            total_loss += self.seg_weight * ped_seg_loss
        
        if 'ped_center' in predictions and 'ped_center' in targets:
            ped_center_loss = self.compute_center_loss(
                predictions['ped_center'],
                targets['ped_center']
            )
            losses['ped_center_loss'] = ped_center_loss
            total_loss += self.center_weight * ped_center_loss
        
        if 'ped_offset' in predictions and 'ped_offset' in targets:
            ped_offset_loss = self.compute_offset_loss(
                predictions['ped_offset'],
                targets['ped_offset'],
                targets.get('ped', None)
            )
            losses['ped_offset_loss'] = ped_offset_loss
            total_loss += self.offset_weight * ped_offset_loss
        
        # Visibility losses
        for obj_type in ['vehicle', 'ped']:
            vis_key = f'{obj_type}_visibility'
            if vis_key in predictions and vis_key in targets:
                vis_loss = self.compute_visibility_loss(
                    predictions[vis_key],
                    targets[vis_key]
                )
                losses[f'{obj_type}_visibility_loss'] = vis_loss
                total_loss += self.visibility_weight * vis_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def compute_segmentation_loss(
        self,
        pred_seg: Tensor,
        target_seg: Tensor
    ) -> Tensor:
        """
        Compute segmentation loss using focal loss.
        
        Args:
            pred_seg: Predicted segmentation [B, H, W]
            target_seg: Target segmentation [B, H, W]
            
        Returns:
            Segmentation loss
        """
        # Normalize target to [0, 1]
        target_normalized = target_seg / 255.0
        target_binary = (target_normalized > 0.5).astype(ms.int32)
        
        # Compute focal loss
        loss = self.focal_loss(pred_seg, target_binary)
        
        return loss
    
    def compute_center_loss(
        self,
        pred_center: Tensor,
        target_center: Tensor
    ) -> Tensor:
        """
        Compute center point detection loss.
        
        Args:
            pred_center: Predicted center heatmap [B, H, W]
            target_center: Target center heatmap [B, H, W]
            
        Returns:
            Center detection loss
        """
        # Use focal loss for center point detection
        # Target is already in [0, 1] range
        target_binary = (target_center > 0.1).astype(ms.int32)
        loss = self.focal_loss(pred_center, target_binary)
        
        return loss
    
    def compute_offset_loss(
        self,
        pred_offset: Tensor,
        target_offset: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute offset regression loss.
        
        Args:
            pred_offset: Predicted offsets [B, H, W, 2]
            target_offset: Target offsets [B, H, W, 2]
            mask: Optional mask for valid regions [B, H, W]
            
        Returns:
            Offset regression loss
        """
        if mask is not None:
            # Apply mask to focus on object regions
            mask_normalized = (mask / 255.0 > 0.5).astype(ms.float32)
            mask_expanded = mask_normalized.expand_dims(-1)  # [B, H, W, 1]
            
            # Compute masked loss
            diff = pred_offset - target_offset
            masked_diff = diff * mask_expanded
            loss = self.smooth_l1_loss(masked_diff, ops.zeros_like(masked_diff))
            
            # Normalize by number of valid pixels
            num_valid = self.sum(mask_normalized) + 1e-8
            loss = self.sum(loss) / num_valid
        else:
            # Compute loss over all pixels
            loss = self.smooth_l1_loss(pred_offset, target_offset)
        
        return loss
    
    def compute_visibility_loss(
        self,
        pred_visibility: Tensor,
        target_visibility: Tensor
    ) -> Tensor:
        """
        Compute visibility classification loss.
        
        Args:
            pred_visibility: Predicted visibility logits [B, H, W, num_classes]
            target_visibility: Target visibility labels [B, H, W]
            
        Returns:
            Visibility classification loss
        """
        # Reshape for cross entropy
        batch_size, height, width = target_visibility.shape
        pred_flat = pred_visibility.reshape(-1, pred_visibility.shape[-1])
        target_flat = target_visibility.reshape(-1)
        
        # Compute cross entropy loss
        loss = self.cross_entropy(pred_flat, target_flat)
        
        return loss


class SimplifiedGaussianLoss(nn.Cell):
    """
    Simplified version of GaussianLSS loss for quick prototyping.
    
    This version only includes the most essential loss components.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def construct(self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tensor:
        """Compute simplified loss."""
        total_loss = 0.0
        
        # Simple MSE loss on BEV features
        if 'bev_features' in predictions:
            # Dummy target - in practice would use real targets
            dummy_target = ops.zeros_like(predictions['bev_features'])
            loss = self.mse_loss(predictions['bev_features'], dummy_target)
            total_loss += loss
        
        return total_loss