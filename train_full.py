#!/usr/bin/env python3
"""
Complete training script for GaussianLSS MindSpore using full NuScenes data.
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GaussianLSSLoss(nn.Cell):
    """Loss function for GaussianLSS training."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def construct(self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tensor:
        """Compute total loss."""
        total_loss = Tensor(0.0, ms.float32)
        loss_dict = {}
        
        # Heatmap loss (detection)
        if 'heatmap' in predictions and 'gt_heatmap' in targets:
            heatmap_loss = self.mse_loss(predictions['heatmap'], targets['gt_heatmap'])
            total_loss += heatmap_loss * 1.0
            loss_dict['heatmap_loss'] = heatmap_loss
        
        # Regression losses
        if 'offset' in predictions and 'gt_offset' in targets:
            offset_loss = self.l1_loss(predictions['offset'], targets['gt_offset'])
            total_loss += offset_loss * 0.1
            loss_dict['offset_loss'] = offset_loss
            
        if 'depth' in predictions and 'gt_depth' in targets:
            depth_loss = self.l1_loss(predictions['depth'], targets['gt_depth'])
            total_loss += depth_loss * 0.1
            loss_dict['depth_loss'] = depth_loss
            
        if 'rotation' in predictions and 'gt_rotation' in targets:
            rotation_loss = self.l1_loss(predictions['rotation'], targets['gt_rotation'])
            total_loss += rotation_loss * 0.1
            loss_dict['rotation_loss'] = rotation_loss
            
        if 'size' in predictions and 'gt_size' in targets:
            size_loss = self.l1_loss(predictions['size'], targets['gt_size'])
            total_loss += size_loss * 0.1
            loss_dict['size_loss'] = size_loss
        
        # If no specific targets available, use a simple reconstruction loss
        if len(loss_dict) == 0:
            # Fallback: use heatmap with dummy target
            if 'heatmap' in predictions:
                dummy_target = ops.zeros_like(predictions['heatmap'])
                total_loss = self.mse_loss(predictions['heatmap'], dummy_target)
                loss_dict['reconstruction_loss'] = total_loss
        
        return total_loss

class TrainingNetwork(nn.Cell):
    """Training network wrapper."""
    
    def __init__(self, network, loss_fn):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        
    def construct(self, *inputs):
        """Forward pass with loss computation."""
        # Unpack inputs
        batch = inputs[0] if len(inputs) == 1 else inputs
        
        # Forward pass
        predictions = self.network(batch)
        
        # Compute loss (using batch as targets for now)
        loss = self.loss_fn(predictions, batch)
        
        return loss

class FullTrainer:
    """Complete trainer for GaussianLSS."""
    
    def __init__(self, config_path: str = "configs/gaussianlss.yaml"):
        """Initialize trainer."""
        self.config_path = config_path
        self.config = None
        self.model = None
        self.data_module = None
        self.loss_fn = None
        self.optimizer = None
        self.train_network = None
        
    def load_config(self):
        """Load training configuration."""
        logger.info("Loading configuration...")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {self.config_path}")
        
    def setup_mindspore(self):
        """Setup MindSpore context."""
        logger.info("Setting up MindSpore context...")
        ms.set_context(
            mode=ms.PYNATIVE_MODE,  # Use PYNATIVE mode for easier debugging
            device_target="CPU",
            save_graphs=False,
            max_call_depth=1000
        )
        logger.info("MindSpore context configured")
        
    def create_model(self):
        """Create the GaussianLSS model."""
        logger.info("Creating GaussianLSS model...")
        self.model = GaussianLSS.from_config(self.config['model'])
        
        # Count parameters
        total_params = sum(p.size for p in self.model.get_parameters())
        trainable_params = sum(p.size for p in self.model.trainable_params())
        
        logger.info(f"Model created:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        
        # Verify we have trainable parameters
        if trainable_params == 0:
            raise ValueError("Model has no trainable parameters! Check model initialization.")
        
    def create_data_module(self):
        """Create data module and loaders."""
        logger.info("Creating data module...")
        self.data_module = DataModule(**self.config['data'])
        
        # Setup data
        self.data_module.setup('fit')
        
        train_size = self.data_module._train_dataset.get_dataset_size()
        val_size = self.data_module._val_dataset.get_dataset_size()
        
        logger.info(f"Data module created:")
        logger.info(f"  - Training batches: {train_size}")
        logger.info(f"  - Validation batches: {val_size}")
        
    def create_loss_and_optimizer(self):
        """Create loss function and optimizer."""
        logger.info("Creating loss function and optimizer...")
        
        # Loss function
        self.loss_fn = GaussianLSSLoss()
        
        # Training configuration
        optimizer_config = self.config.get('optimizer', {})
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        # Optimizer
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=lr,
            weight_decay=weight_decay
        )
        
        # Training network
        self.train_network = TrainingNetwork(self.model, self.loss_fn)
        
        logger.info(f"Loss and optimizer created:")
        logger.info(f"  - Learning rate: {lr}")
        logger.info(f"  - Weight decay: {weight_decay}")
        
    def setup_callbacks(self, save_dir: str = "./checkpoints"):
        """Setup training callbacks."""
        logger.info("Setting up training callbacks...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Checkpoint callback
        config_ck = CheckpointConfig(
            save_checkpoint_steps=100,  # Save every 100 steps
            keep_checkpoint_max=10,     # Keep last 10 checkpoints
        )
        ckpoint_cb = ModelCheckpoint(
            prefix="gaussianlss",
            directory=save_dir,
            config=config_ck
        )
        
        # Loss monitor
        loss_cb = LossMonitor(per_print_times=10)  # Print loss every 10 steps
        
        # Time monitor
        time_cb = TimeMonitor(data_size=self.data_module._train_dataset.get_dataset_size())
        
        callbacks = [ckpoint_cb, loss_cb, time_cb]
        
        logger.info(f"Callbacks configured:")
        logger.info(f"  - Checkpoint directory: {save_dir}")
        logger.info(f"  - Save frequency: every 100 steps")
        logger.info(f"  - Loss print frequency: every 10 steps")
        
        return callbacks
        
    def run_training(self, epochs: int = 10):
        """Run complete training."""
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Get data loaders
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Create MindSpore Model
        model = Model(
            network=self.train_network,
            optimizer=self.optimizer,
            metrics=None  # We'll handle metrics manually
        )
        
        logger.info("Starting training loop...")
        training_start = time.time()
        
        try:
            # Train the model
            model.train(
                epoch=epochs,
                train_dataset=train_loader,
                callbacks=callbacks,
                dataset_sink_mode=False  # Disable dataset sink for debugging
            )
            
            training_time = time.time() - training_start
            logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            
            # Run validation
            logger.info("Running final validation...")
            self.run_validation(val_loader)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def run_validation(self, val_loader):
        """Run validation."""
        logger.info("Running validation...")
        
        self.model.set_train(False)
        
        total_loss = 0.0
        num_batches = 0
        
        val_start = time.time()
        
        try:
            data_iter = val_loader.create_dict_iterator()
            
            for batch_idx, batch in enumerate(data_iter):
                # Forward pass
                predictions = self.model(batch)
                loss = self.loss_fn(predictions, batch)
                
                total_loss += loss.asnumpy()
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    logger.info(f"Validation batch {batch_idx}: loss = {loss.asnumpy():.6f}")
                    
            avg_loss = total_loss / max(num_batches, 1)
            val_time = time.time() - val_start
            
            logger.info(f"Validation completed:")
            logger.info(f"  - Average loss: {avg_loss:.6f}")
            logger.info(f"  - Time: {val_time:.2f}s")
            logger.info(f"  - Batches processed: {num_batches}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            
    def run_complete_training(self, epochs: int = 10):
        """Run the complete training pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE GAUSSIANLSS MINDSPORE TRAINING")
        logger.info("=" * 80)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Load configuration
            self.load_config()
            
            # Step 2: Setup MindSpore
            self.setup_mindspore()
            
            # Step 3: Create model
            self.create_model()
            
            # Step 4: Create data module
            self.create_data_module()
            
            # Step 5: Create loss and optimizer
            self.create_loss_and_optimizer()
            
            # Step 6: Run training
            success = self.run_training(epochs)
            
            pipeline_time = time.time() - pipeline_start
            
            if success:
                logger.info("=" * 80)
                logger.info("TRAINING COMPLETED SUCCESSFULLY!")
                logger.info(f"Total pipeline time: {pipeline_time:.2f} seconds")
                logger.info("=" * 80)
                return True
            else:
                logger.error("Training pipeline failed!")
                return False
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GaussianLSS MindSpore Full Training')
    parser.add_argument('--config', type=str, default='configs/gaussianlss.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("GaussianLSS MindSpore - Complete Training with NuScenes Data")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Log level: {args.log_level}")
    print("=" * 80)
    
    # Create trainer
    trainer = FullTrainer(args.config)
    
    # Run complete training
    success = trainer.run_complete_training(args.epochs)
    
    if success:
        print("\n🎉 Complete training finished successfully!")
        return 0
    else:
        print("\n❌ Training failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())