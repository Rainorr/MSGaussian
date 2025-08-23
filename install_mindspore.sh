#!/bin/bash

echo "🚀 Installing MindSpore for GaussianLSS Migration"
echo "================================================"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv_mindspore
source venv_mindspore/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install MindSpore CPU version (for testing)
echo "🧠 Installing MindSpore..."
pip install mindspore

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install numpy opencv-python Pillow imageio pandas h5py
pip install matplotlib seaborn plotly tqdm einops shapely
pip install omegaconf hydra-core pyyaml

# Install NuScenes devkit (optional, for data processing)
echo "🚗 Installing NuScenes devkit..."
pip install nuscenes-devkit || echo "⚠️  NuScenes devkit installation failed (optional)"

# Install development tools
echo "🛠️  Installing development tools..."
pip install pytest black flake8 isort

echo "✅ Installation complete!"
echo ""
echo "To activate the environment:"
echo "source venv_mindspore/bin/activate"
echo ""
echo "To test the installation:"
echo "python test_migration.py"