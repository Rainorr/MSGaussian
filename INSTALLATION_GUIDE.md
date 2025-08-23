# 🚀 MindSpore 2.6 安装指南

## 📋 环境要求

- Python 3.7-3.9 (推荐 3.8)
- MindSpore 2.6.0+
- CUDA 11.6+ (GPU版本)

## 🔧 安装步骤

### 步骤1: 创建虚拟环境

```bash
# 使用conda (推荐)
conda create -n gaussianlss_ms python=3.8
conda activate gaussianlss_ms

# 或使用venv
python -m venv gaussianlss_ms
source gaussianlss_ms/bin/activate  # Linux/Mac
# gaussianlss_ms\Scripts\activate  # Windows
```

### 步骤2: 安装MindSpore 2.6

```bash
# CPU版本
pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU版本 (CUDA 11.6)
pip install mindspore-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或从官方源安装
pip install mindspore==2.6.0
```

### 步骤3: 安装核心依赖

```bash
# 使用兼容的requirements文件
pip install -r requirements_mindspore26.txt

# 或手动安装核心依赖
pip install numpy==1.24.3
pip install opencv-python
pip install Pillow
pip install tqdm
pip install einops
pip install pyyaml
```

### 步骤4: 安装项目

```bash
# 开发模式安装
pip install -e .

# 或普通安装
pip install .
```

### 步骤5: 验证安装

```bash
python -c "import mindspore; print(mindspore.__version__)"
python -c "import gaussianlss_ms; print('GaussianLSS MindSpore installed successfully!')"
```

## 🐛 常见问题解决

### 问题1: pytest安装失败

**错误**: `ERROR: Could not find a version that satisfies the requirement pytest>=6.2.0`

**解决方案**:
```bash
# 方法1: 分别安装开发依赖
pip install -r requirements_dev.txt

# 方法2: 跳过开发依赖
pip install -r requirements_mindspore26.txt

# 方法3: 手动安装pytest
pip install pytest --upgrade
```

### 问题2: numpy版本冲突

**错误**: `numpy version conflict`

**解决方案**:
```bash
# 卸载现有numpy
pip uninstall numpy -y

# 安装兼容版本
pip install numpy==1.24.3
```

### 问题3: MindSpore导入失败

**错误**: `ImportError: No module named 'mindspore'`

**解决方案**:
```bash
# 检查Python版本
python --version

# 重新安装MindSpore
pip uninstall mindspore mindspore-gpu -y
pip install mindspore==2.6.0
```

### 问题4: CUDA版本不匹配

**错误**: `CUDA version mismatch`

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的MindSpore
# CUDA 11.6
pip install mindspore-gpu==2.6.0

# 或使用CPU版本
pip install mindspore==2.6.0
```

## 📦 最小安装 (仅核心功能)

如果您只需要核心功能，可以使用最小安装：

```bash
pip install mindspore==2.6.0
pip install numpy==1.24.3
pip install opencv-python
pip install tqdm
pip install einops
pip install pyyaml
```

## 🧪 测试安装

创建测试文件 `test_installation.py`:

```python
#!/usr/bin/env python3
"""测试MindSpore GaussianLSS安装"""

def test_mindspore():
    try:
        import mindspore as ms
        print(f"✅ MindSpore {ms.__version__} installed successfully")
        return True
    except ImportError as e:
        print(f"❌ MindSpore import failed: {e}")
        return False

def test_dependencies():
    deps = ['numpy', 'cv2', 'PIL', 'tqdm', 'einops', 'yaml']
    success = True
    
    for dep in deps:
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
            print(f"❌ {dep} import failed: {e}")
            success = False
    
    return success

def test_gaussianlss():
    try:
        import gaussianlss_ms
        print("✅ GaussianLSS MindSpore package imported successfully")
        return True
    except ImportError as e:
        print(f"❌ GaussianLSS MindSpore import failed: {e}")
        print("   This is expected if you haven't installed the package yet")
        return False

if __name__ == "__main__":
    print("🧪 Testing MindSpore GaussianLSS Installation")
    print("=" * 50)
    
    ms_ok = test_mindspore()
    deps_ok = test_dependencies()
    pkg_ok = test_gaussianlss()
    
    print("\n📊 Installation Summary:")
    print(f"MindSpore: {'✅' if ms_ok else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"GaussianLSS Package: {'✅' if pkg_ok else '❌'}")
    
    if ms_ok and deps_ok:
        print("\n🎉 Core installation successful! You can now use GaussianLSS MindSpore.")
    else:
        print("\n⚠️  Please fix the above issues before proceeding.")
```

运行测试：
```bash
python test_installation.py
```

## 🔗 有用链接

- [MindSpore 2.6 官方文档](https://www.mindspore.cn/docs/zh-CN/r2.6/index.html)
- [MindSpore 安装指南](https://www.mindspore.cn/install)
- [MindSpore GitHub](https://github.com/mindspore-ai/mindspore)

## 💡 安装建议

1. **使用虚拟环境**: 避免依赖冲突
2. **检查Python版本**: 确保兼容性
3. **使用国内镜像**: 加速下载
4. **分步安装**: 先装MindSpore，再装其他依赖
5. **验证安装**: 每步都进行验证

如果仍有问题，请提供具体的错误信息！