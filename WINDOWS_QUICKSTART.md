# 🪟 Windows 快速开始指南

## 🚀 5分钟快速安装

### 前提条件
- Windows 10/11 (64位)
- Python 3.7-3.9 已安装

### 一键安装

1. **下载项目**:
```cmd
git clone https://github.com/Rainorr/gaussian_mindspore.git
cd gaussian_mindspore
```

2. **运行安装脚本**:
```cmd
quick_install.bat
```

就这么简单！脚本会自动：
- ✅ 安装MindSpore 2.6
- ✅ 安装所有依赖 (包括numpy 1.26，这是正常的！)
- ✅ 安装项目
- ✅ 运行测试验证

## ❓ 关于numpy版本的说明

**您的观察是正确的！** 

- **MindSpore 2.6 支持 numpy 1.26** ✅
- **我之前推荐 1.24 是过于保守了** 
- **让MindSpore自动选择numpy版本是最好的做法**

### 为什么会这样？

1. **MindSpore官方测试**: MindSpore 2.6经过了与numpy 1.26的兼容性测试
2. **自动依赖管理**: MindSpore会自动安装它测试过的numpy版本
3. **最新功能支持**: numpy 1.26包含了一些性能改进

## 🔧 如果遇到问题

### 问题1: pytest安装失败
```cmd
# 解决方案：跳过开发依赖
pip install mindspore==2.6.0
pip install opencv-python tqdm einops pyyaml Pillow
pip install -e .
```

### 问题2: 权限问题
```cmd
# 以管理员身份运行命令提示符
# 右键点击"命令提示符" -> "以管理员身份运行"
```

### 问题3: 网络慢
```cmd
# 使用国内镜像
pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## ✅ 验证安装

运行测试：
```cmd
python test_installation.py
```

应该看到：
```
✅ MindSpore 2.6.0 installed successfully
✅ NumPy 1.26.x (这个版本号是正常的!)
✅ OpenCV x.x.x
✅ 其他依赖...
🎉 Core installation successful!
```

## 🎯 下一步

```cmd
# 运行演示
python test_migration_demo.py

# 查看项目结构
dir gaussianlss_ms
```

## 💡 重要提醒

**numpy 1.26 是正确的选择！**
- ✅ MindSpore 2.6官方支持
- ✅ 性能更好
- ✅ 功能更完整
- ❌ 不需要降级到1.24

---

**Windows用户专用提示**: 如果您看到numpy 1.26，说明安装是正确的！