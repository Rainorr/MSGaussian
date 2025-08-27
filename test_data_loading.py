#!/usr/bin/env python3
"""
Simple test script to verify data loading works correctly.
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_loading():
    """Test basic data loading functionality."""
    print("Testing basic data loading...")
    
    try:
        import mindspore as ms
        from gaussianlss_ms.data import DataModule
        import yaml
        
        # Set MindSpore context
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        
        # Load config
        with open("configs/gaussianlss.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Dataset dir: {config['data']['dataset_dir']}")
        print(f"Labels dir: {config['data']['labels_dir']}")
        
        # Create data module
        data_module = DataModule(**config['data'])
        data_module.setup('fit')
        
        # Get one sample
        train_dataset = data_module._train_dataset
        data_iter = train_dataset.create_dict_iterator()
        sample = next(data_iter)
        
        print("✅ SUCCESS: Data loading works!")
        print(f"Sample keys: {list(sample.keys())}")
        
        if 'images' in sample:
            print(f"Images shape: {sample['images'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_loading()
    sys.exit(0 if success else 1)
