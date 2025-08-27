"""
Segmentation metrics for GaussianLSS MindSpore implementation.

This module implements metrics for BEV segmentation evaluation.
"""

import mindspore as ms
import mindspore.ops as ops
import numpy as np
from typing import Dict, List, Optional


class SegmentationMetrics:
    """
    Segmentation metrics for BEV segmentation evaluation.
    
    Computes metrics like IoU, Dice coefficient, pixel accuracy, etc.
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 ignore_index: int = 255,
                 threshold: float = 0.5):
        """
        Initialize segmentation metrics.
        
        Args:
            num_classes (int): Number of segmentation classes
            ignore_index (int): Index to ignore in evaluation
            threshold (float): Threshold for binary classification
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        
        # Accumulate confusion matrix
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, predictions: ms.Tensor, targets: ms.Tensor):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions (ms.Tensor): Model predictions [B, C, H, W] or [B, H, W]
            targets (ms.Tensor): Ground truth targets [B, H, W]
        """
        # Convert to numpy
        if isinstance(predictions, ms.Tensor):
            predictions = predictions.asnumpy()
        if isinstance(targets, ms.Tensor):
            targets = targets.asnumpy()
        
        # Handle different prediction formats
        if predictions.ndim == 4:  # [B, C, H, W]
            if predictions.shape[1] == 1:  # Binary segmentation
                predictions = (predictions[:, 0] > self.threshold).astype(np.int64)
            else:  # Multi-class segmentation
                predictions = np.argmax(predictions, dim=1)
        elif predictions.ndim == 3:  # [B, H, W]
            predictions = predictions.astype(np.int64)
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove ignored pixels
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1
        
        self.total_samples += len(predictions)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute segmentation metrics.
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        if self.total_samples == 0:
            return {}
        
        metrics = {}
        
        # Compute per-class IoU
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(dim=1) + 
                self.confusion_matrix.sum(dim=0) - 
                intersection)
        
        # Avoid division by zero
        valid_classes = union > 0
        iou_per_class = np.zeros(self.num_classes)
        iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
        
        # Mean IoU
        metrics['mIoU'] = float(np.mean(iou_per_class[valid_classes]))
        
        # Per-class IoU
        for i in range(self.num_classes):
            metrics[f'IoU_class_{i}'] = float(iou_per_class[i])
        
        # Pixel accuracy
        correct_pixels = np.sum(intersection)
        total_pixels = np.sum(self.confusion_matrix)
        metrics['pixel_accuracy'] = float(correct_pixels / max(total_pixels, 1))
        
        # Mean pixel accuracy
        per_class_accuracy = intersection / np.maximum(self.confusion_matrix.sum(dim=1), 1)
        metrics['mean_pixel_accuracy'] = float(np.mean(per_class_accuracy[valid_classes]))
        
        # Frequency weighted IoU
        freq = self.confusion_matrix.sum(dim=1) / max(total_pixels, 1)
        metrics['freq_weighted_IoU'] = float(np.sum(freq[valid_classes] * iou_per_class[valid_classes]))
        
        # Dice coefficient (F1 score)
        dice_per_class = 2 * intersection / np.maximum(
            self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0), 1)
        metrics['mean_dice'] = float(np.mean(dice_per_class[valid_classes]))
        
        # Per-class Dice
        for i in range(self.num_classes):
            metrics[f'dice_class_{i}'] = float(dice_per_class[i])
        
        # Precision and Recall per class
        precision_per_class = intersection / np.maximum(self.confusion_matrix.sum(dim=0), 1)
        recall_per_class = intersection / np.maximum(self.confusion_matrix.sum(dim=1), 1)
        
        metrics['mean_precision'] = float(np.mean(precision_per_class[valid_classes]))
        metrics['mean_recall'] = float(np.mean(recall_per_class[valid_classes]))
        
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = float(precision_per_class[i])
            metrics[f'recall_class_{i}'] = float(recall_per_class[i])
        
        return metrics
    
    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get the confusion matrix.
        
        Returns:
            np.ndarray: Confusion matrix [num_classes, num_classes]
        """
        return self.confusion_matrix.copy()
    
    def compute_iou_from_confusion_matrix(self, confusion_matrix: np.ndarray) -> np.ndarray:
        """
        Compute IoU from confusion matrix.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            
        Returns:
            np.ndarray: IoU per class
        """
        intersection = np.diag(confusion_matrix)
        union = (confusion_matrix.sum(dim=1) + 
                confusion_matrix.sum(dim=0) - 
                intersection)
        
        # Avoid division by zero
        iou = np.zeros(len(intersection))
        valid_classes = union > 0
        iou[valid_classes] = intersection[valid_classes] / union[valid_classes]
        
        return iou


class BinarySegmentationMetrics(SegmentationMetrics):
    """
    Specialized metrics for binary segmentation.
    """
    
    def __init__(self, threshold: float = 0.5, ignore_index: int = 255):
        super().__init__(num_classes=2, ignore_index=ignore_index, threshold=threshold)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute binary segmentation metrics.
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        metrics = super().compute()
        
        if self.total_samples == 0:
            return metrics
        
        # Binary-specific metrics
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        
        # Sensitivity (True Positive Rate, Recall)
        metrics['sensitivity'] = float(tp / max(tp + fn, 1))
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = float(tn / max(tn + fp, 1))
        
        # Precision (Positive Predictive Value)
        metrics['precision'] = float(tp / max(tp + fp, 1))
        
        # F1 Score
        precision = metrics['precision']
        recall = metrics['sensitivity']
        metrics['f1_score'] = float(2 * precision * recall / max(precision + recall, 1e-8))
        
        # Matthews Correlation Coefficient
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = float(numerator / max(denominator, 1e-8))
        
        # Balanced Accuracy
        metrics['balanced_accuracy'] = float((metrics['sensitivity'] + metrics['specificity']) / 2)
        
        return metrics