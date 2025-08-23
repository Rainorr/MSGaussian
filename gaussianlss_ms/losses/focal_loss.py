"""
Focal Loss implementation for MindSpore.

Focal Loss is designed to address class imbalance in dense object detection
by down-weighting easy examples and focusing on hard examples.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class FocalLoss(nn.Cell):
    """
    Focal Loss implementation.
    
    Reference: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Operations
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.pow = ops.Pow()
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
    
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [N, ...] 
            targets: Ground truth binary labels [N, ...] (0 or 1)
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        p = self.sigmoid(inputs)
        
        # Convert targets to float
        targets_float = targets.astype(ms.float32)
        
        # Compute cross entropy components
        # For positive examples: -log(p)
        # For negative examples: -log(1-p)
        ce_loss = -(targets_float * self.log(p + 1e-8) + 
                   (1 - targets_float) * self.log(1 - p + 1e-8))
        
        # Compute focal weight
        # For positive examples: (1-p)^gamma
        # For negative examples: p^gamma
        p_t = targets_float * p + (1 - targets_float) * (1 - p)
        focal_weight = self.pow(1 - p_t, self.gamma)
        
        # Compute alpha weight
        # For positive examples: alpha
        # For negative examples: (1-alpha)
        alpha_weight = targets_float * self.alpha + (1 - targets_float) * (1 - self.alpha)
        
        # Combine all components
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(focal_loss)
        elif self.reduction == 'sum':
            return self.sum(focal_loss)
        else:
            return focal_loss


class SigmoidFocalLoss(nn.Cell):
    """
    Sigmoid Focal Loss for multi-class classification.
    
    This version applies sigmoid activation and focal loss
    independently to each class.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        self.reduction = reduction
        
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
    
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute sigmoid focal loss for multi-class case.
        
        Args:
            inputs: Predicted logits [N, C, ...]
            targets: Ground truth labels [N, C, ...] (0 or 1 for each class)
            
        Returns:
            Focal loss value
        """
        # Compute focal loss for each class
        loss = self.focal_loss(inputs, targets)
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(loss)
        elif self.reduction == 'sum':
            return self.sum(loss)
        else:
            return loss