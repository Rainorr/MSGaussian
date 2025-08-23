@echo off
echo 🚀 GaussianLSS MindSpore 2.6 快速安装脚本 (Windows)
echo ==========================================

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.7-3.9
    pause
    exit /b 1
)

echo 📋 检测Python版本...
python --version

REM 升级pip
echo 📦 升级pip...
python -m pip install --upgrade pip

REM 安装MindSpore 2.6
echo 🧠 安装MindSpore 2.6...
echo ℹ️  MindSpore会自动安装兼容的numpy版本(如1.26)，这是正常的
python -m pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 验证MindSpore
echo ✅ 验证MindSpore安装...
python -c "import mindspore; print(f'MindSpore {mindspore.__version__} installed successfully')"
if errorlevel 1 (
    echo ❌ MindSpore安装失败，尝试从官方源安装...
    python -m pip install mindspore==2.6.0
)

REM 显示numpy版本
echo 📊 检查numpy版本...
python -c "import numpy; print(f'NumPy version: {numpy.__version__} (兼容MindSpore 2.6)')"

REM 安装核心依赖
echo 📚 安装核心依赖...
python -m pip install -r requirements_mindspore26.txt

REM 安装项目
echo 🔧 安装GaussianLSS项目...
python -m pip install -e .

REM 运行测试
echo 🧪 运行安装测试...
python test_installation.py

echo.
echo 🎉 安装完成！
echo.
echo 📝 下一步：
echo 1. 如果需要开发工具: pip install -r requirements_dev.txt
echo 2. 如果需要NuScenes数据集: pip install nuscenes-devkit
echo 3. 运行演示: python test_migration_demo.py
echo.
echo 📖 更多信息请查看 INSTALLATION_GUIDE.md
pause