"""
Model module wrapper for GaussianLSS MindSpore implementation.

This module provides a high-level interface for training and inference.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from typing import Dict, Any, Optional, List

from .gaussianlss import GaussianLSS
from ..losses import GaussianLSSLoss
from ..metrics import GaussianLSSMetrics


class GaussianLSSModule(nn.Cell):
    """
    High-level module wrapper for GaussianLSS model.
    
    This module combines the model, loss, and metrics for easy training and evaluation.
    
    Args:
        model_config (Dict[str, Any]): Model configuration
        loss_config (Dict[str, Any]): Loss configuration
        optimizer_config (Dict[str, Any]): Optimizer configuration
    """
    
    def __init__(self,
                 model_config: Dict[str, Any],
                 loss_config: Dict[str, Any],
                 optimizer_config: Optional[Dict[str, Any]] = None):
        super(GaussianLSSModule, self).__init__()
        
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config or {}
        
        # Initialize model
        self.model = GaussianLSS(**model_config)
        
        # Initialize loss
        self.loss_fn = GaussianLSSLoss(**loss_config)
        
        # Initialize metrics
        self.metrics = GaussianLSSMetrics()
        
        # Training mode flag
        self.training = True
    
    def construct(self, batch: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        """
        Forward pass of the module.
        
        Args:
            batch (Dict[str, ms.Tensor]): Input batch containing images and targets
            
        Returns:
            Dict[str, ms.Tensor]: Model outputs and losses (if training)
        """
        # Extract inputs
        images = batch['images']  # [B, N, C, H, W] where N is number of cameras
        
        # Forward pass through model
        outputs = self.model(images)
        
        # Compute losses if training
        if self.training and 'targets' in batch:
            targets = batch['targets']
            losses = self.loss_fn(outputs, targets)
            outputs.update(losses)
        
        return outputs
    
    def training_step(self, batch: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        """
        Training step.
        
        Args:
            batch (Dict[str, ms.Tensor]): Training batch
            
        Returns:
            Dict[str, ms.Tensor]: Losses and metrics
        """
        self.training = True
        outputs = self.construct(batch)
        
        # Compute metrics
        if 'targets' in batch:
            metrics = self.metrics.compute(outputs, batch['targets'])
            outputs.update(metrics)
        
        return outputs
    
    def validation_step(self, batch: Dict[str, ms.Tensor]) -> Dict[str, ms.Tensor]:
        """
        Validation step.
        
        Args:
            batch (Dict[str, ms.Tensor]): Validation batch
            
        Returns:
            Dict[str, ms.Tensor]: Predictions and metrics
        """
        self.training = False
        outputs = self.construct(batch)
        
        # Compute metrics
        if 'targets' in batch:
            metrics = self.metrics.compute(outputs, batch['targets'])
            outputs.update(metrics)
        
        return outputs
    
    def predict(self, images: ms.Tensor) -> Dict[str, ms.Tensor]:
        """
        Inference/prediction.
        
        Args:
            images (ms.Tensor): Input images [B, N, C, H, W]
            
        Returns:
            Dict[str, ms.Tensor]: Model predictions
        """
        self.training = False
        batch = {'images': images}
        return self.construct(batch)
    
    def configure_optimizers(self) -> nn.Optimizer:
        """
        Configure optimizer for training.
        
        Returns:
            nn.Optimizer: Configured optimizer
        """
        optimizer_name = self.optimizer_config.get('name', 'adam').lower()
        lr = self.optimizer_config.get('lr', 1e-4)
        weight_decay = self.optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            beta1 = self.optimizer_config.get('beta1', 0.9)
            beta2 = self.optimizer_config.get('beta2', 0.999)
            eps = self.optimizer_config.get('eps', 1e-8)
            
            optimizer = nn.Adam(
                self.trainable_params(),
                learning_rate=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.optimizer_config.get('momentum', 0.9)
            
            optimizer = nn.SGD(
                self.trainable_params(),
                learning_rate=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def get_model_summary(self) -> str:
        """
        Get model summary string.
        
        Returns:
            str: Model summary
        """
        total_params = sum(p.size for p in self.trainable_params())
        
        summary = f"""
GaussianLSS Model Summary:
========================
Total Parameters: {total_params:,}
Model Config: {self.model_config}
Loss Config: {self.loss_config}
Optimizer Config: {self.optimizer_config}
"""
        return summary