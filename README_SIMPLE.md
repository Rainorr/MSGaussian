# GaussianLSS MindSpore - 极简版

基于MindSpore的GaussianLSS 3D目标检测实现，专为Linux环境优化。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
```bash
python scripts/prepare_data.py
```

### 3. 开始训练
```bash
# 使用极简训练脚本
python train_simple.py --epochs 5

# 使用完整训练脚本
python train.py --epochs 10
```

## 项目结构
```
GaussianLSS_MindSpore/
├── gaussianlss_ms/          # 核心代码
│   ├── models/              # 模型定义
│   ├── data/                # 数据处理
│   ├── losses/              # 损失函数
│   └── metrics/             # 评估指标
├── configs/                 # 配置文件
├── scripts/                 # 工具脚本
├── train.py                 # 完整训练脚本
├── train_simple.py          # 极简训练脚本
└── test.py                  # 测试脚本
```

## 系统要求
- Python 3.11+
- MindSpore 2.6.0
- Linux系统

## 使用说明

### 测试环境
```bash
python test.py
```

### 自定义配置
```bash
python train_simple.py --config configs/simple.yaml --epochs 10
```

### 便捷脚本
```bash
./run.sh test      # 运行测试
./run.sh train 10  # 训练10个epoch
```

## 核心特性
- ✅ MindSpore 2.6.0支持
- ✅ Linux环境优化
- ✅ NuScenes数据集支持
- ✅ 3D目标检测
- ✅ 极简代码结构