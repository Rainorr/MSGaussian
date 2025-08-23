# GaussianLSS MindSpore Implementation

This is a MindSpore implementation of GaussianLSS, migrated from the original PyTorch version.

## 🚀 Key Features

- **MindSpore Framework**: Leverages MindSpore's efficient computation graph
- **3D Gaussian Splatting**: Custom MindSpore implementation of Gaussian rasterization
- **Multi-view Fusion**: Processes 6-camera surround view images
- **BEV Generation**: Produces bird's-eye-view representations for autonomous driving

## 📁 Project Structure

```
GaussianLSS_MindSpore/
├── gaussianlss_ms/           # Main package
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model architectures
│   ├── losses/               # Loss functions
│   ├── metrics/              # Evaluation metrics
│   └── utils/                # Utility functions
├── scripts/                  # Training and evaluation scripts
├── configs/                  # Configuration files
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🔧 Installation

```bash
# Install MindSpore (CPU/GPU version)
pip install mindspore

# Install other dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Installation

### 方法1: 快速安装 (推荐)

```bash
# Clone the repository
git clone https://github.com/Rainorr/gaussian_mindspore.git
cd gaussian_mindspore

# 运行快速安装脚本
./quick_install.sh          # Linux/Mac
quick_install.bat           # Windows
```

**Windows用户**: 请查看 [WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md) 获取详细指导

### 方法2: 手动安装 (MindSpore 2.6)

```bash
# 1. 安装MindSpore 2.6
pip install mindspore==2.6.0

# 2. 安装核心依赖
pip install -r requirements_mindspore26.txt

# 3. 安装项目
pip install -e .

# 4. 验证安装
python test_installation.py
```

### 方法3: 最小安装

```bash
pip install mindspore==2.6.0
# numpy会自动安装兼容版本(如1.26)，这是正常的！
pip install opencv-python tqdm einops pyyaml
pip install -e .
```

### 常见问题

**关于numpy版本**: MindSpore 2.6会自动安装numpy 1.26，这是官方支持的版本，无需担心！

如果遇到 `pytest` 安装错误：
```bash
# 跳过开发依赖，只安装核心功能
pip install -r requirements_mindspore26.txt
pip install -e .
```

详细安装指南请查看 [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

## 🏃‍♂️ Quick Start

```bash
# 验证安装
python test_installation.py

# 运行演示
python test_migration_demo.py

# Training (需要数据集)
python scripts/train.py --config configs/gaussianlss.yaml

# Evaluation
python scripts/eval.py --config configs/gaussianlss.yaml --checkpoint path/to/checkpoint
```

## 📊 Migration Status

- [x] Project structure setup
- [x] Data loading pipeline
- [x] Basic model architecture
- [ ] Gaussian rasterization implementation
- [ ] Training loop
- [ ] Loss functions
- [ ] Evaluation metrics
- [ ] Visualization tools

## 🔄 Migration Notes

### Key Differences from PyTorch Version:

1. **Framework**: PyTorch → MindSpore
2. **Data Pipeline**: PyTorch DataLoader → MindSpore Dataset API
3. **Training Loop**: PyTorch Lightning → MindSpore Model API
4. **Gaussian Rasterization**: Custom CUDA extension → MindSpore ops

### Challenges Addressed:

- **Gaussian Rasterization**: Implemented using MindSpore's custom operators
- **Multi-GPU Training**: Leverages MindSpore's parallel training capabilities
- **Memory Optimization**: Uses MindSpore's graph optimization features

## 📈 Performance

Expected improvements with MindSpore:
- Better memory efficiency through graph optimization
- Faster training with MindSpore's parallel execution
- Enhanced deployment capabilities

## 🤝 Contributing

This is a migration project. Please refer to the original PyTorch implementation for the core algorithm details.

## 📄 License

Same as the original GaussianLSS project.