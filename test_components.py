#!/usr/bin/env python3
"""
Test individual components of GaussianLSS model.
"""

import sys
import yaml
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

def test_backbone():
    """Test backbone only."""
    print("Testing backbone...")
    
    from gaussianlss_ms.models.backbones.efficientnet import EfficientNetBackbone
    
    # Set MindSpore context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    # Create backbone
    backbone = EfficientNetBackbone(
        model_name='efficientnet-b4',
        pretrained=True,
        out_indices=(2, 3, 4),
        norm_eval=False
    )
    
    # Test with small input
    x = Tensor(ops.randn(1, 3, 112, 240), ms.float32)
    
    try:
        features = backbone(x)
        print(f"✓ Backbone output shapes: {[f.shape for f in features]}")
        return True
    except Exception as e:
        print(f"❌ Backbone failed: {e}")
        return False

def test_head():
    """Test head only."""
    print("Testing head...")
    
    from gaussianlss_ms.models.heads.gaussian_head import GaussianHead
    
    # Create head
    head = GaussianHead(
        in_channels=160,  # EfficientNet-B4 feature channels
        feat_channels=256,
        num_classes=2,
        depth_num=64
    )
    
    # Test with small input
    x = Tensor(ops.randn(1, 160, 14, 30), ms.float32)
    
    try:
        outputs = head(x)
        print(f"✓ Head output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"  - {key}: {value.shape}")
        return True
    except Exception as e:
        print(f"❌ Head failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss():
    """Test loss function."""
    print("Testing loss function...")
    
    from train_full import GaussianLSSLoss
    
    loss_fn = GaussianLSSLoss()
    
    # Create dummy predictions and targets
    predictions = {
        'predictions': {
            'heatmap': Tensor(ops.randn(1, 2, 28, 60), ms.float32),
            'offset': Tensor(ops.randn(1, 2, 28, 60), ms.float32),
            'depth': Tensor(ops.randn(1, 1, 28, 60), ms.float32),
            'rotation': Tensor(ops.randn(1, 2, 28, 60), ms.float32),
            'size': Tensor(ops.randn(1, 3, 28, 60), ms.float32)
        }
    }
    
    targets = {
        'gt_heatmap': Tensor(ops.randn(1, 2, 28, 60), ms.float32),
        'gt_offset': Tensor(ops.randn(1, 2, 28, 60), ms.float32),
        'gt_depth': Tensor(ops.randn(1, 1, 28, 60), ms.float32),
        'gt_rotation': Tensor(ops.randn(1, 2, 28, 60), ms.float32),
        'gt_size': Tensor(ops.randn(1, 3, 28, 60), ms.float32)
    }
    
    try:
        loss = loss_fn(predictions, targets)
        print(f"✓ Loss computation successful: {loss.asnumpy():.6f}")
        return True
    except Exception as e:
        print(f"❌ Loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("GaussianLSS Component Tests")
    print("=" * 30)
    
    success1 = test_backbone()
    success2 = test_head()
    success3 = test_loss()
    
    if success1 and success2 and success3:
        print("\n🎉 All component tests passed!")
        print("Components are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Some component tests failed!")
        sys.exit(1)