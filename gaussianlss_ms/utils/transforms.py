"""
Transform utilities for GaussianLSS MindSpore implementation.
"""

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from typing import Tuple, List, Optional


def normalize_tensor(
    tensor: Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tensor:
    """
    Normalize tensor with given mean and std.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (N, C, H, W)
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized tensor
    """
    mean_tensor = Tensor(mean, ms.float32).reshape(-1, 1, 1)
    std_tensor = Tensor(std, ms.float32).reshape(-1, 1, 1)
    
    if len(tensor.shape) == 4:  # Batch dimension
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    return (tensor - mean_tensor) / std_tensor


def denormalize_tensor(
    tensor: Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tensor:
    """
    Denormalize tensor with given mean and std.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (N, C, H, W)
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Denormalized tensor
    """
    mean_tensor = Tensor(mean, ms.float32).reshape(-1, 1, 1)
    std_tensor = Tensor(std, ms.float32).reshape(-1, 1, 1)
    
    if len(tensor.shape) == 4:  # Batch dimension
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    return tensor * std_tensor + mean_tensor


def resize_tensor(
    tensor: Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear'
) -> Tensor:
    """
    Resize tensor to given size.
    
    Args:
        tensor: Input tensor of shape (N, C, H, W)
        size: Target size (H, W)
        mode: Interpolation mode
        
    Returns:
        Resized tensor
    """
    resize_op = ops.ResizeBilinear(size, align_corners=False)
    return resize_op(tensor)


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert MindSpore tensor to numpy array."""
    return tensor.asnumpy()


def numpy_to_tensor(array: np.ndarray, dtype: ms.dtype = ms.float32) -> Tensor:
    """Convert numpy array to MindSpore tensor."""
    return Tensor(array, dtype)