#!/usr/bin/env python3
"""测试MindSpore GaussianLSS安装"""

def test_mindspore():
    try:
        import mindspore as ms
        print(f"✅ MindSpore {ms.__version__} installed successfully")
        
        # 测试基本功能
        import mindspore.numpy as mnp
        x = mnp.array([1, 2, 3])
        print(f"✅ MindSpore basic operations work: {x}")
        return True
    except ImportError as e:
        print(f"❌ MindSpore import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MindSpore basic test failed: {e}")
        return False

def test_dependencies():
    deps = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('tqdm', 'tqdm'),
        ('einops', 'einops'),
        ('yaml', 'pyyaml')
    ]
    success = True
    
    for dep, pkg_name in deps:
        try:
            if dep == 'cv2':
                import cv2
                print(f"✅ OpenCV {cv2.__version__}")
            elif dep == 'PIL':
                import PIL
                print(f"✅ Pillow {PIL.__version__}")
            elif dep == 'yaml':
                import yaml
                print(f"✅ PyYAML")
            else:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {dep} {version}")
        except ImportError as e:
            print(f"❌ {dep} ({pkg_name}) import failed: {e}")
            success = False
    
    return success

def test_gaussianlss():
    try:
        import gaussianlss_ms
        print("✅ GaussianLSS MindSpore package imported successfully")
        
        # 测试主要模块
        from gaussianlss_ms.models import GaussianLSS
        from gaussianlss_ms.data import NuScenesDataset
        from gaussianlss_ms.losses import GaussianLSSLoss
        print("✅ Main modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ GaussianLSS MindSpore import failed: {e}")
        print("   This is expected if you haven't installed the package yet")
        return False
    except Exception as e:
        print(f"❌ GaussianLSS MindSpore test failed: {e}")
        return False

def test_mindspore_gpu():
    try:
        import mindspore as ms
        context = ms.get_context()
        device_target = context.get('device_target', 'Unknown')
        print(f"✅ MindSpore device target: {device_target}")
        
        if device_target == 'GPU':
            print("✅ GPU support detected")
        else:
            print("ℹ️  Using CPU mode")
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def print_system_info():
    import sys
    import platform
    
    print("🖥️  System Information:")
    print(f"   Python: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Architecture: {platform.architecture()}")

if __name__ == "__main__":
    print("🧪 Testing MindSpore GaussianLSS Installation")
    print("=" * 50)
    
    print_system_info()
    print()
    
    ms_ok = test_mindspore()
    deps_ok = test_dependencies()
    gpu_ok = test_mindspore_gpu()
    pkg_ok = test_gaussianlss()
    
    print("\n📊 Installation Summary:")
    print(f"MindSpore: {'✅' if ms_ok else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"GPU Support: {'✅' if gpu_ok else '❌'}")
    print(f"GaussianLSS Package: {'✅' if pkg_ok else '❌'}")
    
    if ms_ok and deps_ok:
        print("\n🎉 Core installation successful! You can now use GaussianLSS MindSpore.")
        if not pkg_ok:
            print("💡 To install the GaussianLSS package, run: pip install -e .")
    else:
        print("\n⚠️  Please fix the above issues before proceeding.")
        print("\n🔧 Quick fixes:")
        if not ms_ok:
            print("   - Install MindSpore: pip install mindspore==2.6.0")
        if not deps_ok:
            print("   - Install dependencies: pip install -r requirements_mindspore26.txt")