"""
Smooth L1 Loss implementation for MindSpore.

Smooth L1 Loss is less sensitive to outliers than L2 loss and is commonly
used in object detection for bounding box regression.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from typing import Optional


class SmoothL1Loss(nn.Cell):
    """
    Smooth L1 Loss implementation.
    
    Smooth L1 Loss combines the advantages of L1 and L2 losses:
    - For small errors (|x| < beta), it behaves like L2 loss (quadratic)
    - For large errors (|x| >= beta), it behaves like L1 loss (linear)
    
    This makes it less sensitive to outliers than L2 loss while still being
    differentiable everywhere.
    
    Args:
        beta: Threshold for switching between L1 and L2 loss (default: 1.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean'
    ):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        
        # MindSpore operations
        self.abs = ops.Abs()
        self.square = ops.Square()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.select = ops.Select()
        
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Smooth L1 Loss.
        
        Args:
            inputs: Predicted values of shape (N, ...)
            targets: Ground truth values of same shape as inputs
            
        Returns:
            Smooth L1 loss value
        """
        # Compute absolute difference
        diff = inputs - targets
        abs_diff = self.abs(diff)
        
        # Create condition mask: |diff| < beta
        condition = abs_diff < self.beta
        
        # Compute L2 loss component: 0.5 * diff^2 / beta
        l2_loss = 0.5 * self.square(diff) / self.beta
        
        # Compute L1 loss component: |diff| - 0.5 * beta
        l1_loss = abs_diff - 0.5 * self.beta
        
        # Select between L2 and L1 based on condition
        loss = self.select(condition, l2_loss, l1_loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(loss)
        elif self.reduction == 'sum':
            return self.sum(loss)
        else:
            return loss


class HuberLoss(nn.Cell):
    """
    Huber Loss implementation (alternative name for Smooth L1 Loss).
    
    This is essentially the same as Smooth L1 Loss but with different
    parameterization commonly used in reinforcement learning.
    
    Args:
        delta: Threshold for switching between quadratic and linear loss
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = 'mean'
    ):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        
        # MindSpore operations
        self.abs = ops.Abs()
        self.square = ops.Square()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.select = ops.Select()
        
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Huber Loss.
        
        Args:
            inputs: Predicted values of shape (N, ...)
            targets: Ground truth values of same shape as inputs
            
        Returns:
            Huber loss value
        """
        # Compute absolute difference
        diff = inputs - targets
        abs_diff = self.abs(diff)
        
        # Create condition mask: |diff| <= delta
        condition = abs_diff <= self.delta
        
        # Compute quadratic loss component: 0.5 * diff^2
        quadratic_loss = 0.5 * self.square(diff)
        
        # Compute linear loss component: delta * (|diff| - 0.5 * delta)
        linear_loss = self.delta * (abs_diff - 0.5 * self.delta)
        
        # Select between quadratic and linear based on condition
        loss = self.select(condition, quadratic_loss, linear_loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(loss)
        elif self.reduction == 'sum':
            return self.sum(loss)
        else:
            return loss


class WeightedSmoothL1Loss(nn.Cell):
    """
    Weighted Smooth L1 Loss for handling class imbalance or importance weighting.
    
    Args:
        beta: Threshold for switching between L1 and L2 loss (default: 1.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean'
    ):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        
        # MindSpore operations
        self.abs = ops.Abs()
        self.square = ops.Square()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.select = ops.Select()
        
    def construct(self, inputs: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        """
        Forward pass of Weighted Smooth L1 Loss.
        
        Args:
            inputs: Predicted values of shape (N, ...)
            targets: Ground truth values of same shape as inputs
            weights: Weight values of same shape as inputs
            
        Returns:
            Weighted smooth L1 loss value
        """
        # Compute absolute difference
        diff = inputs - targets
        abs_diff = self.abs(diff)
        
        # Create condition mask: |diff| < beta
        condition = abs_diff < self.beta
        
        # Compute L2 loss component: 0.5 * diff^2 / beta
        l2_loss = 0.5 * self.square(diff) / self.beta
        
        # Compute L1 loss component: |diff| - 0.5 * beta
        l1_loss = abs_diff - 0.5 * self.beta
        
        # Select between L2 and L1 based on condition
        loss = self.select(condition, l2_loss, l1_loss)
        
        # Apply weights
        weighted_loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(weighted_loss)
        elif self.reduction == 'sum':
            return self.sum(weighted_loss)
        else:
            return weighted_loss