#!/usr/bin/env python3
"""
Fix Windows path separators in preprocessed data files.

This script fixes the path separator issue in already processed data files
by converting Windows backslashes to Unix forward slashes.
"""

import json
import pathlib
import argparse
from typing import Dict, Any


def fix_paths_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively fix path separators in dictionary."""
    if isinstance(data, dict):
        return {key: fix_paths_in_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [fix_paths_in_dict(item) for item in data]
    elif isinstance(data, str) and ('\\' in data):
        # Fix path separators
        return data.replace('\\', '/')
    else:
        return data


def fix_sample_file(file_path: pathlib.Path) -> bool:
    """Fix a single sample file."""
    try:
        print(f"Processing: {file_path}")
        
        # Load data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Fix paths
        fixed_data = fix_paths_in_dict(data)
        
        # Check if any changes were made
        if data != fixed_data:
            # Save fixed data
            with open(file_path, 'w') as f:
                json.dump(fixed_data, f, indent=2)
            print(f"  ✅ Fixed paths in {file_path}")
            return True
        else:
            print(f"  ⏭️  No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix Windows path separators in data files")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/processed",
        help="Directory containing processed data files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    
    args = parser.parse_args()
    
    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print(f"🔍 Scanning for JSON files in: {data_dir}")
    
    # Find all JSON files
    json_files = list(data_dir.rglob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found")
        return
    
    print(f"📁 Found {len(json_files)} JSON files")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    fixed_count = 0
    
    for json_file in json_files:
        if args.dry_run:
            # Just check if file needs fixing
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                fixed_data = fix_paths_in_dict(data)
                if data != fixed_data:
                    print(f"  🔧 Would fix: {json_file}")
                    fixed_count += 1
                else:
                    print(f"  ✅ OK: {json_file}")
            except Exception as e:
                print(f"  ❌ Error reading {json_file}: {e}")
        else:
            # Actually fix the file
            if fix_sample_file(json_file):
                fixed_count += 1
    
    if args.dry_run:
        print(f"\n📊 Summary: {fixed_count} files would be fixed")
    else:
        print(f"\n📊 Summary: {fixed_count} files were fixed")


if __name__ == "__main__":
    main()