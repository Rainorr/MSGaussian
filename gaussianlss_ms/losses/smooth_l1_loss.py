"""
Smooth L1 Loss implementation for MindSpore.

Smooth L1 Loss is commonly used for regression tasks in object detection,
providing a balance between L1 and L2 losses.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class SmoothL1Loss(nn.Cell):
    """
    Smooth L1 Loss implementation.
    
    The loss is defined as:
    - 0.5 * (x)^2 / beta, if |x| < beta
    - |x| - 0.5 * beta, otherwise
    
    where x = input - target
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Smooth L1 Loss.
        
        Args:
            beta: Threshold for switching between L1 and L2 loss
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.beta = beta
        self.reduction = reduction
        
        # Operations
        self.abs = ops.Abs()
        self.square = ops.Square()
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
    
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute smooth L1 loss.
        
        Args:
            inputs: Predicted values [N, ...]
            targets: Ground truth values [N, ...]
            
        Returns:
            Smooth L1 loss value
        """
        # Compute absolute difference
        diff = inputs - targets
        abs_diff = self.abs(diff)
        
        # Apply smooth L1 formula
        # Use conditional logic: if |x| < beta, use quadratic, else linear
        quadratic = 0.5 * self.square(diff) / self.beta
        linear = abs_diff - 0.5 * self.beta
        
        # Create mask for switching between quadratic and linear
        mask = (abs_diff < self.beta).astype(ms.float32)
        
        # Combine quadratic and linear parts
        loss = mask * quadratic + (1 - mask) * linear
        
        # Apply reduction
        if self.reduction == 'mean':
            return self.mean(loss)
        elif self.reduction == 'sum':
            return self.sum(loss)
        else:
            return loss


class WeightedSmoothL1Loss(nn.Cell):
    """
    Weighted Smooth L1 Loss with per-sample weighting.
    
    This version allows applying different weights to different samples
    or spatial locations.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.smooth_l1 = SmoothL1Loss(beta=beta, reduction='none')
        self.reduction = reduction
        
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
    
    def construct(
        self,
        inputs: Tensor,
        targets: Tensor,
        weights: Tensor
    ) -> Tensor:
        """
        Compute weighted smooth L1 loss.
        
        Args:
            inputs: Predicted values [N, ...]
            targets: Ground truth values [N, ...]
            weights: Per-sample weights [N, ...] (same shape as inputs)
            
        Returns:
            Weighted smooth L1 loss value
        """
        # Compute unweighted loss
        loss = self.smooth_l1(inputs, targets)
        
        # Apply weights
        weighted_loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            # Normalize by sum of weights to maintain proper scaling
            weight_sum = self.sum(weights) + 1e-8
            return self.sum(weighted_loss) / weight_sum
        elif self.reduction == 'sum':
            return self.sum(weighted_loss)
        else:
            return weighted_loss


class HuberLoss(nn.Cell):
    """
    Huber Loss (alternative name for Smooth L1 Loss).
    
    This is essentially the same as Smooth L1 Loss but with
    a different parameterization (delta instead of beta).
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Huber Loss.
        
        Args:
            delta: Threshold parameter (equivalent to beta in Smooth L1)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        # Huber loss is the same as Smooth L1 with beta = delta
        self.smooth_l1 = SmoothL1Loss(beta=delta, reduction=reduction)
    
    def construct(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute Huber loss."""
        return self.smooth_l1(inputs, targets)