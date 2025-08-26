#!/usr/bin/env python3
"""
GaussianLSS MindSpore Training Script - Linux Optimized Version
"""

import os
import sys
import time
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from gaussianlss_ms.models.gaussianlss import GaussianLSS
from gaussianlss_ms.data import DataModule

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: str = "training.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class GaussianLSSLoss(nn.Cell):
    """Optimized loss function for GaussianLSS."""
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        self.loss_weights = loss_weights or {
            'heatmap': 1.0,
            'offset': 0.1,
            'depth': 0.1,
            'rotation': 0.1,
            'size': 0.1
        }
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def construct(self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tensor:
        """Compute weighted loss."""
        total_loss = Tensor(0.0, ms.float32)
        
        # Main heatmap loss
        if 'heatmap' in predictions:
            # Use dummy target for now - in real training this would be ground truth
            target_shape = predictions['heatmap'].shape
            dummy_target = ops.zeros(target_shape, predictions['heatmap'].dtype)
            heatmap_loss = self.mse_loss(predictions['heatmap'], dummy_target)
            total_loss += heatmap_loss * self.loss_weights['heatmap']
        
        # Additional losses can be added here when ground truth is available
        
        return total_loss

class TrainingNetwork(nn.Cell):
    """Training network wrapper."""
    
    def __init__(self, network, loss_fn):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        
    def construct(self, *inputs):
        """Forward pass with loss computation."""
        batch = inputs[0] if len(inputs) == 1 else inputs
        predictions = self.network(batch)
        loss = self.loss_fn(predictions, batch)
        return loss

class GaussianLSSTrainer:
    """Main trainer class for GaussianLSS."""
    
    def __init__(self, config_path: str = "configs/gaussianlss.yaml"):
        """Initialize trainer."""
        self.config_path = config_path
        self.config = None
        self.model = None
        self.data_module = None
        self.loss_fn = None
        self.optimizer = None
        self.train_network = None
        self.logger = None
        
    def load_config(self):
        """Load configuration."""
        self.logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.logger.info("Configuration loaded successfully")
        
    def setup_mindspore(self):
        """Setup MindSpore context for Linux."""
        self.logger.info("Setting up MindSpore context for Linux")
        
        # Optimize for Linux environment
        ms.set_context(
            mode=ms.GRAPH_MODE,  # Use GRAPH mode for better performance
            device_target="CPU",  # Can be changed to GPU if available
            save_graphs=False,
            max_call_depth=1000,
            enable_graph_kernel=False,  # Disable for stability
            graph_kernel_flags="--disable_expand_ops=Softmax,Dropout "
                              "--disable_cluster_ops=ReduceMax,ReduceMin "
                              "--composite_op_limit_size=50"
        )
        
        # Set environment variables for better performance
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
        
        self.logger.info("MindSpore context configured for Linux")
        
    def create_model(self):
        """Create and initialize model."""
        self.logger.info("Creating GaussianLSS model")
        self.model = GaussianLSS(**self.config['model'])
        
        # Model statistics
        total_params = sum(p.size for p in self.model.get_parameters())
        trainable_params = sum(p.size for p in self.model.trainable_params())
        
        self.logger.info(f"Model created:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
    def create_data_module(self):
        """Create data module."""
        self.logger.info("Creating data module")
        self.data_module = DataModule(**self.config['data'])
        self.data_module.setup('fit')
        
        train_size = self.data_module._train_dataset.get_dataset_size()
        val_size = self.data_module._val_dataset.get_dataset_size()
        
        self.logger.info(f"Data module created:")
        self.logger.info(f"  Training batches: {train_size}")
        self.logger.info(f"  Validation batches: {val_size}")
        
    def create_loss_and_optimizer(self):
        """Create loss function and optimizer."""
        self.logger.info("Creating loss function and optimizer")
        
        # Loss function
        loss_weights = self.config.get('loss', {})
        self.loss_fn = GaussianLSSLoss(loss_weights)
        
        # Optimizer configuration
        opt_config = self.config.get('optimizer', {})
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=lr,
            weight_decay=weight_decay,
            beta1=opt_config.get('beta1', 0.9),
            beta2=opt_config.get('beta2', 0.999),
            eps=opt_config.get('eps', 1e-8)
        )
        
        # Training network
        self.train_network = TrainingNetwork(self.model, self.loss_fn)
        
        self.logger.info(f"Optimizer created with lr={lr}, weight_decay={weight_decay}")
        
    def setup_callbacks(self):
        """Setup training callbacks."""
        train_config = self.config.get('training', {})
        checkpoint_dir = train_config.get('checkpoint_dir', 'checkpoints')
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint callback
        config_ck = CheckpointConfig(
            save_checkpoint_steps=train_config.get('save_interval', 100),
            keep_checkpoint_max=train_config.get('keep_checkpoint_max', 5)
        )
        ckpoint_cb = ModelCheckpoint(
            prefix="gaussianlss",
            directory=checkpoint_dir,
            config=config_ck
        )
        
        # Loss monitor
        loss_cb = LossMonitor(per_print_times=train_config.get('log_interval', 10))
        
        # Time monitor
        time_cb = TimeMonitor(data_size=self.data_module._train_dataset.get_dataset_size())
        
        self.logger.info(f"Callbacks configured:")
        self.logger.info(f"  Checkpoint directory: {checkpoint_dir}")
        self.logger.info(f"  Save interval: {train_config.get('save_interval', 100)} steps")
        
        return [ckpoint_cb, loss_cb, time_cb]
        
    def train(self, epochs: int):
        """Run training."""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Get data loaders
        train_loader = self.data_module.train_dataloader()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Create MindSpore Model
        model = Model(
            network=self.train_network,
            optimizer=self.optimizer,
            metrics=None
        )
        
        # Training loop
        start_time = time.time()
        
        try:
            model.train(
                epoch=epochs,
                train_dataset=train_loader,
                callbacks=callbacks,
                dataset_sink_mode=False  # Better for debugging
            )
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Run validation
            self.validate()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def validate(self):
        """Run validation."""
        self.logger.info("Running validation")
        
        self.model.set_train(False)
        val_loader = self.data_module.val_dataloader()
        
        total_loss = 0.0
        num_batches = 0
        
        try:
            data_iter = val_loader.create_dict_iterator()
            
            for batch_idx, batch in enumerate(data_iter):
                predictions = self.model(batch)
                loss = self.loss_fn(predictions, batch)
                
                total_loss += loss.asnumpy()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    self.logger.info(f"Validation batch {batch_idx}: loss = {loss.asnumpy():.6f}")
                    
            avg_loss = total_loss / max(num_batches, 1)
            self.logger.info(f"Validation completed: avg_loss = {avg_loss:.6f}")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            
    def run(self, epochs: int, log_level: str = "INFO"):
        """Run complete training pipeline."""
        # Setup logging
        self.logger = setup_logging(log_level)
        
        self.logger.info("=" * 80)
        self.logger.info("GAUSSIANLSS MINDSPORE TRAINING - LINUX OPTIMIZED")
        self.logger.info("=" * 80)
        
        try:
            # Pipeline steps
            self.load_config()
            self.setup_mindspore()
            self.create_model()
            self.create_data_module()
            self.create_loss_and_optimizer()
            
            # Run training
            success = self.train(epochs)
            
            if success:
                self.logger.info("=" * 80)
                self.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
                self.logger.info("=" * 80)
                return True
            else:
                self.logger.error("Training failed!")
                return False
                
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GaussianLSS MindSpore Training - Linux Optimized')
    parser.add_argument('--config', type=str, default='configs/gaussianlss.yaml',
                       help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    print("GaussianLSS MindSpore Training - Linux Optimized")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Log level: {args.log_level}")
    print("=" * 60)
    
    # Create and run trainer
    trainer = GaussianLSSTrainer(args.config)
    success = trainer.run(args.epochs, args.log_level)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())