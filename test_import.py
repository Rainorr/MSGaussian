#!/usr/bin/env python3
"""
Test import to find the source of pathlib.Pure error.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import pathlib
    print("✅ pathlib imported successfully")
    print(f"pathlib attributes: {[attr for attr in dir(pathlib) if 'Pure' in attr]}")
except Exception as e:
    print(f"❌ Error importing pathlib: {e}")

try:
    from pathlib import Path
    print("✅ Path imported successfully")
except Exception as e:
    print(f"❌ Error importing Path: {e}")

try:
    from pathlib import Pure
    print("✅ Pure imported successfully")
except Exception as e:
    print(f"❌ Error importing Pure: {e}")

# Test our modules one by one
modules_to_test = [
    'gaussianlss_ms.data.transforms',
    'gaussianlss_ms.data.dataset', 
    'gaussianlss_ms.data.data_module',
    'gaussianlss_ms.models',
    'gaussianlss_ms'
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        import traceback
        traceback.print_exc()
        break