"""
Focal Loss implementation for MindSpore.

Focal Loss addresses class imbalance by down-weighting easy examples
and focusing on hard examples during training.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from typing import Optional


class FocalLoss(nn.Cell):
    """
    Focal Loss implementation for binary and multi-class classification.
    
    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing learning on hard negatives.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
        ignore_index: Specifies a target value that is ignored
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # MindSpore operations
        self.softmax = ops.Softmax(axis=1)
        self.log_softmax = ops.LogSoftmax(axis=1)
        self.pow = ops.Pow()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.gather = ops.GatherD()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()
        
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predictions of shape (N, C) where N is batch size, C is number of classes
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        log_pt = self.log_softmax(inputs)
        
        # Get log probabilities for target classes
        targets_expanded = self.expand_dims(targets, 1)
        log_pt = self.gather(log_pt, 1, targets_expanded)
        log_pt = self.squeeze(log_pt, 1)
        
        # Compute probabilities
        pt = ops.exp(log_pt)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = self.pow(1 - pt, self.gamma)
        
        # Apply alpha weighting
        alpha_weight = self.alpha
        
        # Compute focal loss
        focal_loss = -alpha_weight * focal_weight * log_pt
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(focal_loss)
        elif self.reduction == 'sum':
            return self.sum(focal_loss)
        else:
            return focal_loss


class BinaryFocalLoss(nn.Cell):
    """
    Binary Focal Loss for binary classification tasks.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # MindSpore operations
        self.sigmoid = ops.Sigmoid()
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Binary Focal Loss.
        
        Args:
            inputs: Predictions of shape (N,) or (N, 1)
            targets: Ground truth labels of shape (N,) with values 0 or 1
            
        Returns:
            Binary focal loss value
        """
        # Apply sigmoid to get probabilities
        p = self.sigmoid(inputs)
        
        # Compute binary cross entropy components
        ce_loss = targets * self.log(p + 1e-8) + (1 - targets) * self.log(1 - p + 1e-8)
        
        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute focal weight
        focal_weight = self.pow(1 - p_t, self.gamma)
        
        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = -alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(focal_loss)
        elif self.reduction == 'sum':
            return self.sum(focal_loss)
        else:
            return focal_loss


class SigmoidFocalLoss(nn.Cell):
    """
    Sigmoid Focal Loss for dense object detection.
    
    This is commonly used in object detection frameworks like RetinaNet.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # MindSpore operations
        self.sigmoid = ops.Sigmoid()
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.maximum = ops.Maximum()
        
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Sigmoid Focal Loss.
        
        Args:
            inputs: Predictions of shape (N, C, H, W) or (N, C)
            targets: Ground truth labels of same shape as inputs
            
        Returns:
            Sigmoid focal loss value
        """
        # Apply sigmoid to get probabilities
        p = self.sigmoid(inputs)
        
        # Compute binary cross entropy
        ce_loss = -(targets * self.log(self.maximum(p, Tensor(1e-8, ms.float32))) + 
                   (1 - targets) * self.log(self.maximum(1 - p, Tensor(1e-8, ms.float32))))
        
        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute focal weight
        focal_weight = self.pow(1 - p_t, self.gamma)
        
        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(focal_loss)
        elif self.reduction == 'sum':
            return self.sum(focal_loss)
        else:
            return focal_loss