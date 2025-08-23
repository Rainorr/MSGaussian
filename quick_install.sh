#!/bin/bash

echo "🚀 GaussianLSS MindSpore 2.6 快速安装脚本"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 检测到Python版本: $python_version"

if [[ "$python_version" < "3.7" ]] || [[ "$python_version" > "3.9" ]]; then
    echo "⚠️  警告: 推荐使用Python 3.7-3.9版本"
fi

# 升级pip
echo "📦 升级pip..."
python3 -m pip install --upgrade pip

# 安装MindSpore 2.6
echo "🧠 安装MindSpore 2.6..."
python3 -m pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 检查MindSpore安装
echo "✅ 验证MindSpore安装..."
python3 -c "import mindspore; print(f'MindSpore {mindspore.__version__} installed successfully')" || {
    echo "❌ MindSpore安装失败，尝试从官方源安装..."
    python3 -m pip install mindspore==2.6.0
}

# 安装核心依赖
echo "📚 安装核心依赖..."
python3 -m pip install -r requirements_mindspore26.txt

# 安装项目
echo "🔧 安装GaussianLSS项目..."
python3 -m pip install -e .

# 运行测试
echo "🧪 运行安装测试..."
python3 test_installation.py

echo ""
echo "🎉 安装完成！"
echo ""
echo "📝 下一步："
echo "1. 如果需要开发工具: pip install -r requirements_dev.txt"
echo "2. 如果需要NuScenes数据集: pip install nuscenes-devkit"
echo "3. 运行演示: python test_migration_demo.py"
echo ""
echo "📖 更多信息请查看 INSTALLATION_GUIDE.md"