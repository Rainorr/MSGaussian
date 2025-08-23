"""
Main metrics for GaussianLSS model evaluation.

This module implements comprehensive metrics for evaluating
the GaussianLSS model performance.
"""

from typing import Dict, Any, List

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .detection_metrics import DetectionMetrics
from .segmentation_metrics import SegmentationMetrics


class GaussianLSSMetrics(nn.Metric):
    """
    Comprehensive metrics for GaussianLSS model.
    
    This class combines multiple evaluation metrics including:
    - BEV segmentation metrics (IoU, precision, recall)
    - Object detection metrics (AP, mAP)
    - Center point detection metrics
    - Offset regression metrics
    """
    
    def __init__(self):
        super().__init__()
        
        # Sub-metrics
        self.detection_metrics = DetectionMetrics()
        self.segmentation_metrics = SegmentationMetrics()
        
        # Metric storage
        self.clear()
    
    def clear(self):
        """Clear all accumulated metrics."""
        self.detection_metrics.clear()
        self.segmentation_metrics.clear()
        
        # Additional metrics
        self.total_samples = 0
        self.center_errors = []
        self.offset_errors = []
    
    def update(self, *inputs):
        """
        Update metrics with new predictions and targets.
        
        Args:
            inputs: Tuple of (predictions, targets)
        """
        if len(inputs) != 2:
            raise ValueError("Expected (predictions, targets) as inputs")
        
        predictions, targets = inputs
        
        # Update segmentation metrics
        if 'vehicle_seg' in predictions and 'vehicle' in targets:
            self.segmentation_metrics.update(
                predictions['vehicle_seg'],
                targets['vehicle']
            )
        
        if 'ped_seg' in predictions and 'ped' in targets:
            self.segmentation_metrics.update(
                predictions['ped_seg'],
                targets['ped']
            )
        
        # Update detection metrics (simplified)
        # In practice, would need to convert predictions to detection format
        
        # Update center point metrics
        if 'vehicle_center' in predictions and 'vehicle_center' in targets:
            center_error = self._compute_center_error(
                predictions['vehicle_center'],
                targets['vehicle_center']
            )
            self.center_errors.append(center_error)
        
        # Update offset metrics
        if 'vehicle_offset' in predictions and 'vehicle_offset' in targets:
            offset_error = self._compute_offset_error(
                predictions['vehicle_offset'],
                targets['vehicle_offset']
            )
            self.offset_errors.append(offset_error)
        
        self.total_samples += 1
    
    def eval(self) -> Dict[str, float]:
        """
        Compute final metric values.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Get segmentation metrics
        seg_metrics = self.segmentation_metrics.eval()
        for key, value in seg_metrics.items():
            metrics[f'seg_{key}'] = value
        
        # Get detection metrics
        det_metrics = self.detection_metrics.eval()
        for key, value in det_metrics.items():
            metrics[f'det_{key}'] = value
        
        # Compute center point metrics
        if self.center_errors:
            center_errors = np.array(self.center_errors)
            metrics['center_mae'] = np.mean(center_errors)
            metrics['center_rmse'] = np.sqrt(np.mean(center_errors ** 2))
        
        # Compute offset metrics
        if self.offset_errors:
            offset_errors = np.array(self.offset_errors)
            metrics['offset_mae'] = np.mean(offset_errors)
            metrics['offset_rmse'] = np.sqrt(np.mean(offset_errors ** 2))
        
        # Overall metrics
        metrics['total_samples'] = self.total_samples
        
        return metrics
    
    def _compute_center_error(
        self,
        pred_center: Tensor,
        target_center: Tensor
    ) -> float:
        """
        Compute center point detection error.
        
        Args:
            pred_center: Predicted center heatmap [B, H, W]
            target_center: Target center heatmap [B, H, W]
            
        Returns:
            Mean absolute error
        """
        # Convert to numpy for computation
        pred_np = pred_center.asnumpy()
        target_np = target_center.asnumpy()
        
        # Compute MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        return mae
    
    def _compute_offset_error(
        self,
        pred_offset: Tensor,
        target_offset: Tensor
    ) -> float:
        """
        Compute offset regression error.
        
        Args:
            pred_offset: Predicted offsets [B, H, W, 2]
            target_offset: Target offsets [B, H, W, 2]
            
        Returns:
            Mean absolute error
        """
        # Convert to numpy for computation
        pred_np = pred_offset.asnumpy()
        target_np = target_offset.asnumpy()
        
        # Compute MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        return mae


class SimplifiedMetrics(nn.Metric):
    """
    Simplified metrics for quick evaluation during development.
    """
    
    def __init__(self):
        super().__init__()
        self.clear()
    
    def clear(self):
        """Clear accumulated metrics."""
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, *inputs):
        """Update with loss value."""
        if len(inputs) == 1:
            loss = inputs[0]
            if isinstance(loss, Tensor):
                loss = loss.asnumpy()
            self.total_loss += float(loss)
            self.num_batches += 1
    
    def eval(self) -> Dict[str, float]:
        """Compute average loss."""
        if self.num_batches == 0:
            return {'avg_loss': 0.0}
        
        return {
            'avg_loss': self.total_loss / self.num_batches,
            'num_batches': self.num_batches
        }


class BEVMetrics(nn.Metric):
    """
    Specialized metrics for BEV (Bird's Eye View) evaluation.
    
    This class focuses on metrics specific to BEV representation
    quality and spatial accuracy.
    """
    
    def __init__(self, grid_size: tuple = (200, 200)):
        super().__init__()
        self.grid_size = grid_size
        self.clear()
    
    def clear(self):
        """Clear accumulated metrics."""
        self.spatial_errors = []
        self.coverage_scores = []
        self.num_samples = 0
    
    def update(self, *inputs):
        """
        Update BEV metrics.
        
        Args:
            inputs: Tuple of (bev_predictions, bev_targets)
        """
        if len(inputs) != 2:
            return
        
        bev_pred, bev_target = inputs
        
        # Compute spatial accuracy
        spatial_error = self._compute_spatial_error(bev_pred, bev_target)
        self.spatial_errors.append(spatial_error)
        
        # Compute coverage score
        coverage = self._compute_coverage_score(bev_pred, bev_target)
        self.coverage_scores.append(coverage)
        
        self.num_samples += 1
    
    def eval(self) -> Dict[str, float]:
        """Compute BEV metrics."""
        if self.num_samples == 0:
            return {}
        
        metrics = {
            'bev_spatial_error': np.mean(self.spatial_errors),
            'bev_coverage_score': np.mean(self.coverage_scores),
            'bev_samples': self.num_samples
        }
        
        return metrics
    
    def _compute_spatial_error(self, pred: Tensor, target: Tensor) -> float:
        """Compute spatial error in BEV space."""
        # Simplified spatial error computation
        pred_np = pred.asnumpy()
        target_np = target.asnumpy()
        
        # Compute pixel-wise error
        error = np.mean(np.abs(pred_np - target_np))
        
        return error
    
    def _compute_coverage_score(self, pred: Tensor, target: Tensor) -> float:
        """Compute coverage score (how well prediction covers target)."""
        # Simplified coverage computation
        pred_np = pred.asnumpy()
        target_np = target.asnumpy()
        
        # Binarize predictions and targets
        pred_binary = (pred_np > 0.5).astype(np.float32)
        target_binary = (target_np > 0.5).astype(np.float32)
        
        # Compute intersection over union
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(np.maximum(pred_binary, target_binary))
        
        if union == 0:
            return 1.0  # Perfect score if both are empty
        
        iou = intersection / union
        return iou