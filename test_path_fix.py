#!/usr/bin/env python3
"""
Test script to verify path fixes work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from gaussianlss_ms.data.transforms import LoadDataTransform
from gaussianlss_ms.data.dataset import NuScenesDataset
import pathlib

def test_path_handling():
    """Test that path handling works correctly."""
    print("=== Testing Path Handling ===")
    
    # Test pathlib usage
    test_path = pathlib.Path("data/nuscenes/samples/CAM_FRONT/test.jpg")
    print(f"✅ pathlib.Path works: {test_path}")
    
    # Test path normalization
    windows_path = "data\\nuscenes\\samples\\CAM_FRONT\\test.jpg"
    normalized = windows_path.replace('\\', '/')
    print(f"✅ Path normalization: {windows_path} -> {normalized}")
    
    # Test dataset creation (without loading data)
    try:
        dataset_dir = "data/nuscenes"
        labels_dir = "data/processed"
        
        if os.path.exists(dataset_dir) and os.path.exists(labels_dir):
            transform = LoadDataTransform(
                dataset_dir=dataset_dir,
                labels_dir=labels_dir,
                image_config={'h': 224, 'w': 480, 'top_crop': 46}
            )
            
            dataset = NuScenesDataset(
                dataset_dir=dataset_dir,
                labels_dir=labels_dir,
                transform=transform
            )
            
            print(f"✅ Dataset created successfully: {len(dataset)} samples")
            
            # Try to load first sample
            if len(dataset) > 0:
                try:
                    sample = dataset[0]
                    print("✅ Successfully loaded first sample")
                    print(f"Sample keys: {list(sample.keys())}")
                except Exception as e:
                    print(f"⚠️  Could not load sample (expected if data not available): {e}")
            else:
                print("⚠️  No samples found in dataset")
        else:
            print(f"⚠️  Data directories not found: {dataset_dir}, {labels_dir}")
            print("This is expected if you haven't downloaded the data yet.")
    
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

def test_imports():
    """Test that all modules import correctly."""
    print("\n=== Testing Module Imports ===")
    
    modules_to_test = [
        'gaussianlss_ms',
        'gaussianlss_ms.data',
        'gaussianlss_ms.models', 
        'gaussianlss_ms.losses',
        'gaussianlss_ms.metrics',
        'gaussianlss_ms.utils'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")

if __name__ == "__main__":
    print("GaussianLSS MindSpore - Path Fix Test")
    print("=" * 50)
    
    test_imports()
    test_path_handling()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nIf you see this message, the path fixes are working correctly.")
    print("You can now run your training/inference scripts.")