#!/usr/bin/env python3
"""
GaussianLSS MindSpore - 极简训练脚本
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor

from gaussianlss_ms.models.gaussianlss import GaussianLSS
from gaussianlss_ms.data import DataModule

class SimpleLoss(nn.Cell):
    """极简损失函数"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def construct(self, predictions, targets):
        # 简单的MSE损失
        if isinstance(predictions, dict) and 'heatmap' in predictions:
            target_shape = predictions['heatmap'].shape
            dummy_target = ops.zeros(target_shape, predictions['heatmap'].dtype)
            return self.mse_loss(predictions['heatmap'], dummy_target)
        return Tensor(0.0, ms.float32)

class SimpleTrainer:
    """极简训练器"""
    
    def __init__(self, config_path="configs/gaussianlss.yaml"):
        self.config_path = config_path
        self.config = None
        self.model = None
        self.data_module = None
        self.loss_fn = None
        self.optimizer = None
        
    def setup(self):
        """初始化设置"""
        print("Setting up MindSpore...")
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        
        print("Loading configuration...")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        print("Creating model...")
        self.model = GaussianLSS(**self.config['model'])
        
        print("Creating data module...")
        self.data_module = DataModule(**self.config['data'])
        self.data_module.setup('fit')
        
        print("Setting up loss and optimizer...")
        self.loss_fn = SimpleLoss()
        
        lr = self.config.get('optimizer', {}).get('lr', 1e-4)
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=lr)
        
    def train_step(self, batch):
        """单步训练"""
        def forward_fn():
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, batch)
            return loss
            
        grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters)
        loss, grads = grad_fn()
        self.optimizer(grads)
        return loss
        
    def train(self, epochs=5):
        """训练循环"""
        print(f"Starting training for {epochs} epochs...")
        
        train_loader = self.data_module.train_dataloader()
        data_iter = train_loader.create_dict_iterator()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            try:
                for batch_idx, batch in enumerate(data_iter):
                    loss = self.train_step(batch)
                    epoch_loss += loss.asnumpy()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}: loss = {loss.asnumpy():.6f}")
                        
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"  Epoch {epoch + 1} average loss: {avg_loss:.6f}")
                
            except Exception as e:
                print(f"Training error: {e}")
                break
                
        print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='GaussianLSS Simple Training')
    parser.add_argument('--config', type=str, default='configs/gaussianlss.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    
    args = parser.parse_args()
    
    print("GaussianLSS MindSpore - Simple Training")
    print("=" * 40)
    
    trainer = SimpleTrainer(args.config)
    trainer.setup()
    trainer.train(args.epochs)

if __name__ == "__main__":
    main()