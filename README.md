# GaussianLSS MindSpore

GaussianLSS的MindSpore实现，用于基于高斯分布的大规模场景理解。

## 项目结构

```
GaussianLSS_MindSpore/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── train.py                     # 训练脚本
├── configs/
│   └── gaussianlss.yaml        # 配置文件
├── data/
│   ├── nuscenes/               # NuScenes数据集
│   └── processed/              # 预处理数据
├── scripts/
│   └── prepare_data.py         # 数据预处理脚本
├── gaussianlss_ms/             # 核心代码
│   ├── __init__.py
│   ├── data/                   # 数据处理模块
│   ├── models/                 # 模型定义
│   ├── losses/                 # 损失函数
│   ├── metrics/                # 评估指标
│   └── utils/                  # 工具函数
└── venv_mindspore/             # MindSpore虚拟环境
```

## 环境要求

- Python 3.11+
- MindSpore 2.6.0+
- CUDA (可选，用于GPU训练)

## 快速开始

### 1. 激活环境

```bash
cd GaussianLSS_MindSpore
source venv_mindspore/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 数据预处理

```bash
python scripts/prepare_data.py
```

### 4. 开始训练

```bash
python train.py --epochs 10
```

## 配置说明

主要配置文件：`configs/gaussianlss.yaml`

- `data`: 数据相关配置
- `model`: 模型结构配置
- `training`: 训练参数配置
- `optimizer`: 优化器配置

## 核心模块

### 模型 (gaussianlss_ms/models/)
- `gaussianlss.py`: 主模型
- `backbone.py`: 骨干网络
- `neck.py`: 特征融合网络
- `head.py`: 检测头

### 数据 (gaussianlss_ms/data/)
- `dataset.py`: 数据集定义
- `transforms.py`: 数据变换
- `data_module.py`: 数据模块

### 损失函数 (gaussianlss_ms/losses/)
- `gaussian_loss.py`: 高斯损失函数

## 训练参数

常用训练参数：

```bash
python train.py \
    --epochs 20 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --log-level INFO
```

## 数据格式

支持NuScenes数据集格式，预处理后的数据包含：
- 图像数据
- 相机参数
- 标注信息
- BEV标签

## 注意事项

1. 确保有足够的内存和存储空间
2. 建议使用GPU进行训练以提高速度
3. 数据预处理可能需要较长时间
4. 训练过程中会自动保存检查点

## 许可证

本项目基于原始GaussianLSS项目，遵循相应的开源许可证。# MSGaussian
