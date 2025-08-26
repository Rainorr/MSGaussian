"""
Visualization utilities for GaussianLSS MindSpore implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import cv2


def visualize_bev(
    bev_map: np.ndarray,
    title: str = "BEV Map",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize bird's-eye-view map.
    
    Args:
        bev_map: BEV map array of shape (H, W) or (H, W, C)
        title: Plot title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    
    if len(bev_map.shape) == 3:
        plt.imshow(bev_map)
    else:
        plt.imshow(bev_map, cmap='viridis')
    
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()


def plot_detections(
    image: np.ndarray,
    detections: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot detections on image.
    
    Args:
        image: Input image array
        detections: List of detection dictionaries
        save_path: Path to save the visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, 
                               fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            
            # Add label if available
            if 'label' in det:
                ax.text(x, y-5, det['label'], 
                       color='red', fontsize=12, weight='bold')
    
    ax.set_title("Detections")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()


def save_predictions(
    predictions: Dict,
    save_dir: str,
    prefix: str = "pred"
) -> None:
    """
    Save predictions to files.
    
    Args:
        predictions: Dictionary of predictions
        save_dir: Directory to save predictions
        prefix: File prefix
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            save_path = os.path.join(save_dir, f"{prefix}_{key}.npy")
            np.save(save_path, value)
        else:
            # Convert tensor to numpy if needed
            try:
                array = value.asnumpy()
                save_path = os.path.join(save_dir, f"{prefix}_{key}.npy")
                np.save(save_path, array)
            except:
                print(f"Could not save {key}: unsupported type {type(value)}")


def create_color_map(num_classes: int) -> np.ndarray:
    """Create color map for visualization."""
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    return (colors[:, :3] * 255).astype(np.uint8)