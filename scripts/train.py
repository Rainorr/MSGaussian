#!/usr/bin/env python3
"""
Training script for GaussianLSS MindSpore implementation.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Training loop with validation
- Checkpointing and logging
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Model, CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train.callback import Callback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from gaussianlss_ms.data import DataModule
from gaussianlss_ms.models import GaussianLSS, EfficientNetBackbone
from gaussianlss_ms.losses import GaussianLSSLoss
from gaussianlss_ms.metrics import GaussianLSSMetrics


def setup_logging(log_dir: Path, log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> GaussianLSS:
    """Create GaussianLSS model from configuration."""
    model_config = config['model']
    
    # Create backbone
    backbone = EfficientNetBackbone(
        model_name=model_config['backbone']['name'],
        pretrained=model_config['backbone']['pretrained'],
        out_indices=model_config['backbone']['out_indices']
    )
    
    # Create neck (placeholder - would implement FPN or similar)
    neck = nn.Identity()
    
    # Create head (placeholder - would implement Gaussian parameter prediction)
    head = nn.Identity()
    
    # Create decoder (placeholder - would implement final prediction layers)
    decoder = nn.Identity()
    
    # Create main model
    model = GaussianLSS(
        embed_dims=model_config['embed_dims'],
        backbone=backbone,
        neck=neck,
        head=head,
        decoder=decoder,
        **model_config.get('gaussian_params', {})
    )
    
    return model


class ValidationCallback(Callback):
    """Custom callback for validation during training."""
    
    def __init__(self, model, val_dataset, metrics, logger, val_interval=1):
        super().__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.metrics = metrics
        self.logger = logger
        self.val_interval = val_interval
        self.best_metric = float('inf')
    
    def epoch_end(self, run_context):
        """Run validation at the end of each epoch."""
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        
        if epoch % self.val_interval == 0:
            self.logger.info(f"Running validation at epoch {epoch}")
            
            # Run validation
            val_metrics = self.run_validation()
            
            # Log metrics
            for name, value in val_metrics.items():
                self.logger.info(f"Validation {name}: {value:.4f}")
            
            # Check if this is the best model
            current_metric = val_metrics.get('loss', float('inf'))
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.logger.info(f"New best validation metric: {current_metric:.4f}")
    
    def run_validation(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        # Set model to eval mode
        self.model.set_train(False)
        
        total_loss = 0.0
        num_batches = 0
        
        try:
            for batch in self.val_dataset.create_dict_iterator():
                # Forward pass
                outputs = self.model(batch['image'], batch['lidar2img'])
                
                # Compute loss (placeholder)
                loss = ops.mean(outputs['bev_features'])  # Dummy loss
                
                total_loss += loss.asnumpy()
                num_batches += 1
        
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {'loss': float('inf')}
        
        finally:
            # Set model back to train mode
            self.model.set_train(True)
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}


def train(config: Dict[str, Any], logger: logging.Logger):
    """Main training function."""
    
    # Set MindSpore context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config['device']['target'],
        device_id=config['device']['id']
    )
    
    logger.info(f"Using device: {config['device']['target']} {config['device']['id']}")
    
    # Create data module
    data_config = config['data']
    data_module = DataModule(
        dataset_dir=data_config['dataset_dir'],
        labels_dir=data_config['labels_dir'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        **data_config.get('transform_config', {})
    )
    
    # Setup datasets
    data_module.setup('fit')
    train_dataset = data_module.train_dataloader()
    val_dataset = data_module.val_dataloader()
    
    logger.info(f"Train dataset size: {train_dataset.get_dataset_size()}")
    logger.info(f"Validation dataset size: {val_dataset.get_dataset_size()}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.size for p in model.get_parameters())} parameters")
    
    # Create loss function
    loss_fn = GaussianLSSLoss(**config.get('loss', {}))
    
    # Create optimizer
    optimizer_config = config['optimizer']
    if optimizer_config['name'] == 'adam':
        optimizer = nn.Adam(
            model.trainable_params(),
            learning_rate=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    elif optimizer_config['name'] == 'sgd':
        optimizer = nn.SGD(
            model.trainable_params(),
            learning_rate=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    # Create metrics
    metrics = GaussianLSSMetrics()
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=config['training']['save_interval'],
        keep_checkpoint_max=config['training']['keep_checkpoint_max']
    )
    ckpt_callback = ModelCheckpoint(
        prefix='gaussianlss',
        directory=config['training']['checkpoint_dir'],
        config=ckpt_config
    )
    callbacks.append(ckpt_callback)
    
    # Monitoring callbacks
    callbacks.append(LossMonitor(per_print_times=config['training']['log_interval']))
    callbacks.append(TimeMonitor())
    
    # Validation callback
    val_callback = ValidationCallback(
        model=model,
        val_dataset=val_dataset,
        metrics=metrics,
        logger=logger,
        val_interval=config['training']['val_interval']
    )
    callbacks.append(val_callback)
    
    # Create MindSpore Model
    ms_model = Model(
        network=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics
    )
    
    # Start training
    logger.info("Starting training...")
    
    try:
        ms_model.train(
            epoch=config['training']['epochs'],
            train_dataset=train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=config['training'].get('dataset_sink_mode', True)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save final model
    final_ckpt_path = Path(config['training']['checkpoint_dir']) / 'final_model.ckpt'
    ms.save_checkpoint(model, str(final_ckpt_path))
    logger.info(f"Final model saved to: {final_ckpt_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train GaussianLSS MindSpore model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir, args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Create output directories
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    try:
        train(config, logger)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()