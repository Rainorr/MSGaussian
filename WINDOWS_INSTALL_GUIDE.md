# 🪟 Windows 安装指南 - GaussianLSS MindSpore

## 🎯 Windows 专用安装指南

### 系统要求

- **操作系统**: Windows 10/11 (64位)
- **Python**: 3.7-3.9 (推荐 3.8)
- **内存**: 至少 8GB RAM
- **存储**: 至少 5GB 可用空间

### 📦 方法1: 一键安装 (推荐)

1. **下载项目**:
```cmd
git clone https://github.com/Rainorr/gaussian_mindspore.git
cd gaussian_mindspore
```

2. **运行安装脚本**:
```cmd
quick_install.bat
```

### 🔧 方法2: 手动安装

#### 步骤1: 检查Python环境

```cmd
python --version
```
确保显示 Python 3.7-3.9 版本

#### 步骤2: 创建虚拟环境 (推荐)

```cmd
# 使用venv
python -m venv gaussianlss_env
gaussianlss_env\Scripts\activate

# 或使用conda
conda create -n gaussianlss_env python=3.8
conda activate gaussianlss_env
```

#### 步骤3: 安装MindSpore 2.6

```cmd
# CPU版本 (推荐新手)
pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU版本 (需要CUDA)
pip install mindspore-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 步骤4: 验证MindSpore安装

```cmd
python -c "import mindspore; print('MindSpore version:', mindspore.__version__)"
```

#### 步骤5: 安装项目依赖

```cmd
pip install -r requirements_mindspore26.txt
```

#### 步骤6: 安装项目

```cmd
pip install -e .
```

#### 步骤7: 验证安装

```cmd
python test_installation.py
```

### 🐛 Windows 常见问题解决

#### 问题1: numpy版本问题

**现象**: MindSpore安装了numpy 1.26，但requirements要求<1.25
**解决**: 这是正常的！MindSpore 2.6支持numpy 1.26

```cmd
# 如果遇到版本冲突，直接使用MindSpore安装的版本
pip install mindspore==2.6.0
# numpy会自动安装兼容版本
```

#### 问题2: pytest安装失败

**错误**: `ERROR: Could not find a version that satisfies the requirement pytest>=6.2.0`

**解决方案**:
```cmd
# 方法A: 跳过开发依赖
pip install mindspore==2.6.0
pip install opencv-python tqdm einops pyyaml Pillow
pip install -e .

# 方法B: 单独安装pytest
pip install pytest --upgrade

# 方法C: 使用开发依赖文件
pip install -r requirements_dev.txt
```

#### 问题3: 权限问题

**错误**: `Permission denied` 或 `Access is denied`

**解决方案**:
```cmd
# 以管理员身份运行命令提示符
# 或使用用户安装
pip install --user mindspore==2.6.0
```

#### 问题4: 网络连接问题

**错误**: `Connection timeout` 或下载缓慢

**解决方案**:
```cmd
# 使用国内镜像源
pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip install mindspore==2.6.0 -i https://mirrors.aliyun.com/pypi/simple/

# 或配置永久镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 问题5: CUDA相关问题 (GPU版本)

**错误**: CUDA版本不匹配

**解决方案**:
```cmd
# 检查CUDA版本
nvidia-smi

# 如果没有CUDA或版本不匹配，使用CPU版本
pip uninstall mindspore-gpu
pip install mindspore==2.6.0
```

#### 问题6: Visual Studio Build Tools

**错误**: `Microsoft Visual C++ 14.0 is required`

**解决方案**:
1. 下载并安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 或安装 Visual Studio Community
3. 重启命令提示符后重新安装

### 🧪 Windows 测试脚本

创建 `test_windows.py`:

```python
import sys
import platform

def test_windows_environment():
    print("🪟 Windows 环境测试")
    print("=" * 30)
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.architecture()}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 测试MindSpore
    try:
        import mindspore as ms
        print(f"✅ MindSpore {ms.__version__}")
        
        # 测试设备
        context = ms.get_context()
        device = context.get('device_target', 'Unknown')
        print(f"✅ 设备: {device}")
        
    except ImportError as e:
        print(f"❌ MindSpore导入失败: {e}")
    
    # 测试numpy
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")

if __name__ == "__main__":
    test_windows_environment()
```

运行测试:
```cmd
python test_windows.py
```

### 📝 Windows 安装检查清单

- [ ] Python 3.7-3.9 已安装
- [ ] 虚拟环境已创建并激活
- [ ] MindSpore 2.6 安装成功
- [ ] numpy 版本兼容 (1.21+ 即可，1.26也没问题)
- [ ] 核心依赖安装完成
- [ ] 项目安装成功
- [ ] 测试脚本运行通过

### 🎉 安装成功后

```cmd
# 运行演示
python test_migration_demo.py

# 查看项目信息
python -c "import gaussianlss_ms; print('项目安装成功!')"
```

### 💡 Windows 优化建议

1. **使用Anaconda**: 更好的包管理
2. **配置镜像源**: 加速下载
3. **使用虚拟环境**: 避免依赖冲突
4. **定期更新pip**: `python -m pip install --upgrade pip`

### 📞 需要帮助？

如果遇到问题：
1. 检查Python和pip版本
2. 尝试使用管理员权限
3. 清理pip缓存: `pip cache purge`
4. 重新创建虚拟环境

---

**Windows专用提示**: numpy 1.26版本是MindSpore 2.6官方支持的，无需降级！