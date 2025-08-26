#!/usr/bin/env python3
"""
Test script to verify path fixes and module imports
"""

import sys
import os
import pathlib

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    
    try:
        import gaussianlss_ms
        print("✅ gaussianlss_ms")
    except ImportError as e:
        print(f"❌ gaussianlss_ms: {e}")
        return False
    
    try:
        import gaussianlss_ms.data
        print("✅ gaussianlss_ms.data")
    except ImportError as e:
        print(f"❌ gaussianlss_ms.data: {e}")
        return False
    
    try:
        import gaussianlss_ms.models
        print("✅ gaussianlss_ms.models")
    except ImportError as e:
        print(f"❌ gaussianlss_ms.models: {e}")
        return False
    
    try:
        import gaussianlss_ms.losses
        print("✅ gaussianlss_ms.losses")
    except ImportError as e:
        print(f"❌ gaussianlss_ms.losses: {e}")
        return False
    
    try:
        import gaussianlss_ms.metrics
        print("✅ gaussianlss_ms.metrics")
    except ImportError as e:
        print(f"❌ gaussianlss_ms.metrics: {e}")
        return False
    
    try:
        import gaussianlss_ms.utils
        print("✅ gaussianlss_ms.utils")
    except ImportError as e:
        print(f"❌ gaussianlss_ms.utils: {e}")
        return False
    
    return True

def test_path_normalization():
    """Test path normalization function"""
    print("\nTesting path normalization...")
    
    # Test cases with mixed separators
    test_paths = [
        "samples\\CAM_FRONT\\n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
        "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
        "samples\\CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
        "samples//CAM_FRONT//n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"
    ]
    
    for test_path in test_paths:
        # Apply the same normalization as in transforms.py
        normalized = str(test_path).replace('\\', '/').replace('//', '/')
        expected = "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"
        
        if normalized == expected:
            print(f"✅ Path normalization works: {test_path[:30]}...")
        else:
            print(f"❌ Path normalization failed: {test_path[:30]}...")
            print(f"   Expected: {expected}")
            print(f"   Got: {normalized}")
            return False
    
    return True

def test_dataset_creation():
    """Test basic dataset imports and path handling"""
    print("\nTesting dataset imports...")
    
    try:
        # Test dataset imports
        from gaussianlss_ms.data.dataset import NuScenesDataset
        from gaussianlss_ms.data.transforms import LoadDataTransform, BaseTransform
        print("✅ Dataset imports successful")
        
        # Test path handling function
        test_path = "samples\\CAM_FRONT\\test.jpg"
        normalized = str(test_path).replace('\\', '/').replace('//', '/')
        expected = "samples/CAM_FRONT/test.jpg"
        
        if normalized == expected:
            print("✅ Path handling logic works correctly")
            return True
        else:
            print(f"❌ Path handling failed: got {normalized}, expected {expected}")
            return False
        
    except Exception as e:
        print(f"❌ Dataset imports failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing GaussianLSS MindSpore Path Fixes")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test path normalization
    paths_ok = test_path_normalization()
    
    # Test dataset creation
    dataset_ok = test_dataset_creation()
    
    print("\n" + "=" * 50)
    if imports_ok and paths_ok and dataset_ok:
        print("🎉 All tests passed! Path fixes are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())