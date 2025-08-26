# Linux路径分隔符修复指南

## 问题描述

在Linux系统中运行时出现路径错误：
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/ma-user/work/MSGaussian/data/nuscenes/samples\CAM_FRONT_LEFT\n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg'
```

问题原因：路径中混合了Linux的正斜杠(`/`)和Windows的反斜杠(`\`)。

## 解决方案

### 方案1：使用修复脚本（推荐）

1. 下载并运行路径修复脚本：

```bash
# 在项目根目录下运行
python fix_paths.py --data-dir data/processed
```

### 方案2：手动修复代码

在 `gaussianlss_ms/data/transforms.py` 文件的第196行附近，确保有以下代码：

```python
# 原代码
for image_path, I_original, extrinsic in zip(
    sample.images, sample.intrinsics, sample.extrinsics
):
    # 修复：标准化路径分隔符
    normalized_path = image_path.replace('\\', '/')
    # Load and resize image
    image = Image.open(self.dataset_dir / normalized_path)
```

### 方案3：环境变量修复

如果问题持续存在，可以设置环境变量：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 方案4：重新生成数据

如果上述方案都不工作，重新生成预处理数据：

```bash
# 删除旧数据
rm -rf data/processed/*

# 重新生成数据
python scripts/prepare_data.py \
    --dataset-dir data/nuscenes \
    --output-dir data/processed \
    --version v1.0-mini
```

## 验证修复

运行以下命令验证修复是否成功：

```bash
python simple_debug.py
```

应该看到所有图像文件都显示 `Exists: True`。

## 预防措施

为了避免将来出现类似问题：

1. **使用pathlib**：在Python代码中使用 `pathlib.Path` 而不是字符串拼接
2. **标准化路径**：在处理路径时总是调用 `.replace('\\', '/')`
3. **跨平台测试**：在不同操作系统上测试代码

## 常见问题

### Q: 为什么会出现这个问题？
A: 通常是因为数据在Windows系统上生成，然后在Linux系统上使用，导致路径分隔符不匹配。

### Q: 修复后还是有问题怎么办？
A: 检查你的数据目录路径是否正确，确保 `data/nuscenes/samples/` 目录存在且包含图像文件。

### Q: 可以直接修改数据文件吗？
A: 可以，但不推荐。最好是在代码中处理路径标准化。