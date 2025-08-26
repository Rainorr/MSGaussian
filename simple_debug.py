#!/usr/bin/env python3
"""
Simple debug script to check path issues.
"""

import json
import pathlib
import os


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
            if item.name == "samples":
                samples_dir = item
                print(f"     📁 Samples subdirs:")
                for subdir in sorted(samples_dir.iterdir()):
                    if subdir.is_dir():
                        count = len(list(subdir.glob("*.jpg")))
                        print(f"       - {subdir.name}: {count} images")
    
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
                
                # Check if directory exists
                parent_dir = full_path.parent
                print(f"      Parent dir exists: {parent_dir.exists()}")
                
                if parent_dir.exists():
                    # List files in parent directory
                    files = list(parent_dir.glob("*.jpg"))
                    print(f"      Files in parent: {len(files)}")
                    if files:
                        print(f"      First few files:")
                        for f in files[:3]:
                            print(f"        - {f.name}")


if __name__ == "__main__":
    print("🐛 Simple Path Debug Tool")
    print("=" * 50)
    
    check_data_structure()
    check_sample_paths()