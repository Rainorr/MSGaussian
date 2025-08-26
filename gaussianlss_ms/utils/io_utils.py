"""
I/O utilities for GaussianLSS MindSpore implementation.
"""

import json
import yaml
import pickle
import mindspore as ms
from mindspore import save_checkpoint, load_checkpoint
from typing import Dict, Any, Optional
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (.json or .yaml)
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif save_path.endswith(('.yaml', '.yml')):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {save_path}")


def save_checkpoint(
    network: ms.nn.Cell,
    save_path: str,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        network: MindSpore network
        save_path: Path to save checkpoint
        epoch: Current epoch number
        metrics: Training metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save network parameters
    save_checkpoint(network, save_path)
    
    # Save additional metadata
    if epoch is not None or metrics is not None:
        metadata = {}
        if epoch is not None:
            metadata['epoch'] = epoch
        if metrics is not None:
            metadata['metrics'] = metrics
        
        metadata_path = save_path.replace('.ckpt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_checkpoint_with_metadata(
    network: ms.nn.Cell,
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Load model checkpoint with metadata.
    
    Args:
        network: MindSpore network
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing loaded metadata
    """
    # Load network parameters
    param_dict = load_checkpoint(checkpoint_path)
    ms.load_param_into_net(network, param_dict)
    
    # Load metadata if available
    metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
    metadata = {}
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return metadata


def save_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = 'json'
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
        format: Save format ('json', 'pickle')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):
                serializable_results[key] = value.tolist()
            elif hasattr(value, 'asnumpy'):
                serializable_results[key] = value.asnumpy().tolist()
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results dictionary
    """
    if results_path.endswith('.json'):
        with open(results_path, 'r') as f:
            results = json.load(f)
    elif results_path.endswith('.pickle') or results_path.endswith('.pkl'):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported results format: {results_path}")
    
    return results