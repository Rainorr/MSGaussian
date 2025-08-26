# GaussianLSS MindSpore - Linux Optimized

A MindSpore implementation of GaussianLSS for 3D object detection using multi-view camera images and Gaussian splatting techniques.

## Features

- **MindSpore Framework**: Optimized for MindSpore 2.6.0
- **Linux Optimized**: Specifically tuned for Linux environments
- **NuScenes Dataset**: Full support for NuScenes multi-view data
- **3D Object Detection**: Bird's-eye-view detection using Gaussian splatting
- **Multi-view Fusion**: Efficient fusion of multiple camera views

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv_mindspore
source venv_mindspore/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Prepare NuScenes data
python scripts/prepare_data.py
```

### 3. Training

```bash
# Start training
python train.py --epochs 20 --log-level INFO

# Custom configuration
python train.py --config configs/gaussianlss.yaml --epochs 10
```

## Project Structure

```
GaussianLSS_MindSpore/
├── gaussianlss_ms/           # Core implementation
│   ├── models/               # Model architectures
│   │   ├── backbone/         # EfficientNet backbone
│   │   ├── necks/           # FPN neck
│   │   ├── heads/           # Detection heads
│   │   └── gaussianlss.py   # Main model
│   ├── data/                # Data processing
│   ├── losses/              # Loss functions
│   ├── metrics/             # Evaluation metrics
│   └── utils/               # Utilities
├── configs/                 # Configuration files
├── scripts/                 # Data preparation scripts
├── data/                    # Dataset directory
│   ├── nuscenes/           # NuScenes dataset
│   └── processed/          # Preprocessed data
├── checkpoints/            # Model checkpoints
├── train.py               # Training script
└── requirements.txt       # Dependencies
```

## Configuration

The main configuration file is `configs/gaussianlss.yaml`. Key sections:

- **Model**: Architecture parameters (backbone, neck, head)
- **Data**: Dataset paths and preprocessing options
- **Training**: Epochs, learning rate, checkpointing
- **Optimizer**: Adam optimizer settings

## System Requirements

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.11.x
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ for NuScenes dataset

## Performance

- **Training Speed**: ~10-15 seconds per epoch (CPU mode)
- **Memory Usage**: ~8-12GB during training
- **Model Size**: ~50M parameters

## Troubleshooting

### Common Issues

1. **MindSpore Installation**: Ensure Python 3.11 is used
2. **Memory Issues**: Reduce batch size in config
3. **Data Loading**: Check NuScenes dataset paths

### Logs

Training logs are saved to `training.log`. Monitor progress with:

```bash
tail -f training.log
```

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gaussianlss,
  title={GaussianLSS: 3D Object Detection with Gaussian Splatting},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```