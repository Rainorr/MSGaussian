#!/usr/bin/env python3
"""
Debug script to check path issues in the dataset.
"""

import json
import pathlib
import os
from gaussianlss_ms.data.dataset import NuScenesDataset
from gaussianlss_ms.data.transforms import LoadDataTransform


def check_data_structure():
    """Check the data directory structure."""
    print("🔍 Checking data directory structure...")
    
    # Check main directories
    data_dir = pathlib.Path("data")
    nuscenes_dir = data_dir / "nuscenes"
    processed_dir = data_dir / "processed"
    
    print(f"📁 Data directory exists: {data_dir.exists()}")
    print(f"📁 NuScenes directory exists: {nuscenes_dir.exists()}")
    print(f"📁 Processed directory exists: {processed_dir.exists()}")
    
    if nuscenes_dir.exists():
        print(f"📁 NuScenes contents:")
        for item in sorted(nuscenes_dir.iterdir()):
            print(f"   - {item.name}")
    
    if processed_dir.exists():
        print(f"📁 Processed contents:")
        for item in sorted(processed_dir.iterdir()):
            print(f"   - {item.name}")


def check_sample_paths():
    """Check paths in a sample file."""
    print("\n🔍 Checking sample file paths...")
    
    processed_dir = pathlib.Path("data/processed")
    sample_files = list(processed_dir.glob("*.json"))
    
    if not sample_files:
        print("❌ No sample files found")
        return
    
    # Check first sample file
    sample_file = sample_files[0]
    print(f"📄 Checking: {sample_file}")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    if data:
        sample = data[0]  # First sample
        print(f"🏷️  Token: {sample['token']}")
        print(f"🎬 Scene: {sample['scene']}")
        print(f"🗺️  Map: {sample['map_name']}")
        
        print(f"📸 Images:")
        for i, img_path in enumerate(sample['images']):
            print(f"   {i}: {img_path}")
            
            # Check if file exists
            full_path = pathlib.Path("data/nuscenes") / img_path
            exists = full_path.exists()
            print(f"      Exists: {exists}")
            if not exists:
                # Try with different separators
                alt_path = pathlib.Path("data/nuscenes") / img_path.replace('/', os.sep)
                alt_exists = alt_path.exists()
                print(f"      Alt path exists: {alt_exists}")


def test_dataset_loading():
    """Test dataset loading."""
    print("\n🔍 Testing dataset loading...")
    
    try:
        # Create transform
        transform = LoadDataTransform(
            dataset_dir="data/nuscenes",
            labels_dir="data/processed",
            image_config={'h': 224, 'w': 480, 'top_crop': 46}
        )
        
        # Create dataset
        dataset = NuScenesDataset(
            dataset_dir="data/nuscenes",
            labels_dir="data/processed",
            transform=transform
        )
        
        print(f"✅ Dataset created successfully")
        print(f"📊 Dataset length: {len(dataset)}")
        
        # Try to load first sample
        if len(dataset) > 0:
            print("🔄 Loading first sample...")
            sample = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"🏷️  Keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🐛 GaussianLSS Path Debug Tool")
    print("=" * 50)
    
    check_data_structure()
    check_sample_paths()
    test_dataset_loading()