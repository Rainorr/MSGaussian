"""
NuScenes dataset implementation for MindSpore.

This module provides dataset classes for loading and processing
NuScenes data for the GaussianLSS model.
"""

import json
import pathlib
from typing import List, Dict, Any, Optional, Union

import numpy as np
import mindspore as ms
import mindspore.dataset as ds

from .transforms import Sample, LoadDataTransform
from .common import get_camera_names


class NuScenesDataset:
    """
    NuScenes dataset for GaussianLSS MindSpore implementation.
    
    This dataset loads multi-view camera images and corresponding
    3D bounding box annotations from the NuScenes dataset.
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, pathlib.Path],
        labels_dir: Union[str, pathlib.Path],
        split: str = 'train',
        version: str = 'v1.0-mini',
        transform: Optional[LoadDataTransform] = None,
        **kwargs
    ):
        """
        Initialize NuScenes dataset.
        
        Args:
            dataset_dir: Path to NuScenes dataset directory
            labels_dir: Path to preprocessed labels directory
            split: Dataset split ('train', 'val', 'test')
            version: NuScenes version ('v1.0-mini', 'v1.0-trainval')
            transform: Data transformation pipeline
        """
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.split = split
        self.version = version
        self.transform = transform
        
        # Load scene data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Sample]:
        """Load sample data from preprocessed JSON files."""
        samples = []
        
        # Find all scene JSON files
        scene_files = list(self.labels_dir.glob("scene-*.json"))
        
        for scene_file in scene_files:
            scene_name = scene_file.stem
            
            # Load scene data
            with open(scene_file, 'r') as f:
                scene_data = json.load(f)
            
            # Create samples from scene data
            for sample_data in scene_data:
                sample = Sample(
                    token=sample_data['token'],
                    scene=sample_data['scene'],
                    map_name=sample_data['map_name'],
                    intrinsics=[np.array(intr) for intr in sample_data['intrinsics']],
                    extrinsics=[np.array(extr) for extr in sample_data['extrinsics']],
                    images=sample_data['images'],
                    view=np.array(sample_data['view']),
                    gt_box=sample_data['gt_box'],
                    **{k: v for k, v in sample_data.items() 
                       if k not in ['token', 'scene', 'map_name', 'intrinsics', 
                                   'extrinsics', 'images', 'view', 'gt_box']}
                )
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing processed sample data
        """
        sample = self.samples[idx]
        
        if self.transform is not None:
            return self.transform(sample)
        else:
            return {
                'token': sample.token,
                'scene': sample.scene,
                'map_name': sample.map_name,
                'images': sample.images,
                'intrinsics': sample.intrinsics,
                'extrinsics': sample.extrinsics,
                'view': sample.view,
                'gt_box': sample.gt_box
            }


def create_nuscenes_dataset(
    dataset_dir: Union[str, pathlib.Path],
    labels_dir: Union[str, pathlib.Path],
    split: str = 'train',
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    transform_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ds.Dataset:
    """
    Create MindSpore Dataset for NuScenes data.
    
    Args:
        dataset_dir: Path to NuScenes dataset directory
        labels_dir: Path to preprocessed labels directory
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        transform_config: Configuration for data transformations
        
    Returns:
        MindSpore Dataset object
    """
    # Default transform configuration
    if transform_config is None:
        transform_config = {
            'image_config': {
                'h': 224,
                'w': 480,
                'top_crop': 46
            },
            'vehicle': True,
            'ped': True,
            'image_data': True
        }
    
    # Create transform
    transform = LoadDataTransform(
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
        **transform_config
    )
    
    # Create dataset
    nuscenes_dataset = NuScenesDataset(
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
        split=split,
        transform=transform,
        **kwargs
    )
    
    # Create MindSpore dataset
    def generator():
        """Generator function for MindSpore dataset."""
        for i in range(len(nuscenes_dataset)):
            yield nuscenes_dataset[i]
    
    # Define column names and types
    column_names = ['image', 'lidar2img', 'vehicle', 'vehicle_center', 
                   'vehicle_offset', 'ped', 'ped_center', 'ped_offset']
    
    # Create dataset from generator
    dataset = ds.GeneratorDataset(
        source=generator,
        column_names=column_names,
        num_parallel_workers=num_workers,
        shuffle=shuffle
    )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset


class NuScenesGeneratedDataset(NuScenesDataset):
    """
    Dataset for loading preprocessed NuScenes data.
    
    This dataset loads data that has been preprocessed and saved
    to disk for faster training.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as base dataset."""
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load preprocessed data sample."""
        sample = self.samples[idx]
        
        # For generated dataset, we assume data is already preprocessed
        # and stored in a format that can be quickly loaded
        if self.transform is not None:
            return self.transform(sample)
        else:
            # Return raw sample data
            return super().__getitem__(idx)


def get_dataset_splits(
    dataset_dir: Union[str, pathlib.Path],
    labels_dir: Union[str, pathlib.Path],
        splits=None,
    **kwargs
) -> Dict[str, ds.Dataset]:
    """
    Create datasets for multiple splits.
    
    Args:
        dataset_dir: Path to NuScenes dataset directory
        labels_dir: Path to preprocessed labels directory
        splits: List of dataset splits to create
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Dict mapping split names to MindSpore Dataset objects
    """
    if splits is None:
        splits = ['train', 'val']
    datasets = {}
    
    for split in splits:
        datasets[split] = create_nuscenes_dataset(
            dataset_dir=dataset_dir,
            labels_dir=labels_dir,
            split=split,
            shuffle=(split == 'train'),  # Only shuffle training data
            **kwargs
        )
    
    return datasets


def create_nuscenes_dataset(
    dataset_dir: Union[str, pathlib.Path],
    labels_dir: Union[str, pathlib.Path],
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 1,
    shuffle: bool = True,
    transform_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ds.Dataset:
    """
    Create a MindSpore Dataset for NuScenes data.
    
    Args:
        dataset_dir: Path to NuScenes dataset directory
        labels_dir: Path to preprocessed labels directory
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        transform_config: Configuration for data transforms
        **kwargs: Additional arguments
        
    Returns:
        MindSpore Dataset object
    """
    # Create transform if config provided
    transform = None
    if transform_config:
        transform = LoadDataTransform(
            dataset_dir=dataset_dir,
            labels_dir=labels_dir,
            image_config=transform_config.get('image', {}),
            bev_config=transform_config.get('bev', {}),
            augment_config=transform_config.get('augment', {}),
            **transform_config.get('options', {})
        )
    
    # Create the dataset instance
    dataset = NuScenesDataset(
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
        split=split,
        transform=transform,
        **kwargs
    )
    
    # Convert to MindSpore dataset
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]
    
    # Create MindSpore dataset from generator
    ms_dataset = ds.GeneratorDataset(
        source=generator,
        column_names=['token', 'scene', 'map_name', 'images', 'intrinsics', 'extrinsics', 'view', 'gt_box'],
        shuffle=shuffle,
        num_parallel_workers=num_workers
    )
    
    # Batch the dataset
    if batch_size > 1:
        ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    
    return ms_dataset