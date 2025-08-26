#!/usr/bin/env python3
"""
Quick test script for GaussianLSS MindSpore - Linux Optimized
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all core imports."""
    print("Testing imports...")
    
    try:
        import mindspore as ms
        print(f"✓ MindSpore {ms.__version__}")
        
        import yaml
        print("✓ PyYAML")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        from gaussianlss_ms.models.gaussianlss import GaussianLSS
        print("✓ GaussianLSS model")
        
        from gaussianlss_ms.data import DataModule
        print("✓ DataModule")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open("configs/gaussianlss.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration loaded")
        print(f"  - Model embed_dims: {config['model']['embed_dims']}")
        print(f"  - Training epochs: {config['training']['epochs']}")
        print(f"  - Batch size: {config['data']['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        import mindspore as ms
        import yaml
        
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        
        with open("configs/gaussianlss.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        from gaussianlss_ms.models.gaussianlss import GaussianLSS
        model = GaussianLSS(**config['model'])
        
        total_params = sum(p.size for p in model.get_parameters())
        print(f"✓ Model created with {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_data_module():
    """Test data module creation."""
    print("\nTesting data module...")
    
    try:
        import yaml
        
        with open("configs/gaussianlss.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        from gaussianlss_ms.data import DataModule
        data_module = DataModule(**config['data'])
        
        print("✓ DataModule created")
        
        # Check if data exists
        import os
        if os.path.exists(config['data']['dataset_dir']):
            print("✓ Dataset directory exists")
        else:
            print("⚠ Dataset directory not found (expected for fresh setup)")
            
        if os.path.exists(config['data']['labels_dir']):
            print("✓ Processed data directory exists")
        else:
            print("⚠ Processed data directory not found (run prepare_data.py)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data module test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("GaussianLSS MindSpore - Linux Optimized Test")
    print("=" * 50)
    
    start_time = time.time()
    
    tests = [
        test_imports,
        test_config,
        test_model_creation,
        test_data_module
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == len(tests):
        print("🎉 All tests passed! Project is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())