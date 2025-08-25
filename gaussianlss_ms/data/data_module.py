"""
Data module for managing dataset creation and loading.

This module provides a high-level interface for creating and managing
datasets for training and evaluation.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import mindspore.dataset as ds

from .dataset import create_nuscenes_dataset, get_dataset_splits


class DataModule:
    """
    Data module for GaussianLSS MindSpore implementation.
    
    This class manages dataset creation, loading, and preprocessing
    for training and evaluation.
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        labels_dir: Union[str, Path],
        batch_size: int = 2,
        num_workers: int = 4,
        image_config: Optional[Dict[str, Any]] = None,
        augment_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize data module.
        
        Args:
            dataset_dir: Path to NuScenes dataset directory
            labels_dir: Path to preprocessed labels directory
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            image_config: Image preprocessing configuration
            augment_config: Data augmentation configuration
        """
        self.dataset_dir = Path.cwd() / 'data' / 'nuscenes'
        self.labels_dir = Path.cwd() / 'data' / 'nuscenes' / 'labels'
        print(self.labels_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Default configurations
        self.image_config = image_config or {
            'h': 224,
            'w': 480,
            'top_crop': 46
        }
        
        self.augment_config = augment_config or {}
        
        # Transform configuration
        self.transform_config = {
            'image_config': self.image_config,
            'augment_config': self.augment_config,
            'vehicle': kwargs.get('vehicle', True),
            'ped': kwargs.get('ped', True),
            'image_data': kwargs.get('image_data', True)
        }
        
        # Store additional kwargs
        self.dataset_kwargs = kwargs
        
        # Datasets will be created on demand
        self.train_dataset = None
        self.val_dataset = None
        self._test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for different stages.
        
        Args:
            stage: Stage name ('fit', 'validate', 'test', or None for all)
        """
        if stage == 'fit' or stage is None:
            # Create training and validation datasets
            # Prepare kwargs, avoiding duplicate parameters
            kwargs = {
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'transform_config': self.transform_config,
                **self.dataset_kwargs
            }
            
            datasets = get_dataset_splits(
                dataset_dir=self.dataset_dir,
                labels_dir=self.labels_dir,
                splits=['train', 'val'],
                **kwargs
            )
            
            self.train_dataset = datasets['train']
            self.val_dataset = datasets['val']



    
    def train_dataloader(self) -> ds.Dataset:
        """Get training dataloader."""
        if self.train_dataset is None:
            self.setup('fit')
        return self.train_dataset
    
    def val_dataloader(self) -> ds.Dataset:
        """Get validation dataloader."""
        if self.val_dataset is None:
            self.setup('fit')
        return self.val_dataset
    
    def test_dataloader(self) -> ds.Dataset:
        """Get test dataloader."""
        if self._test_dataset is None:
            self.setup('test')
        return self._test_dataset
    
    def predict_dataloader(self) -> ds.Dataset:
        """Get prediction dataloader (same as test)."""
        return self.test_dataloader()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the datasets.
        
        Returns:
            Dict containing dataset statistics and configuration
        """
        info = {
            'dataset_dir': str(self.dataset_dir),
            'labels_dir': str(self.labels_dir),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'image_config': self.image_config,
            'augment_config': self.augment_config,
            'transform_config': self.transform_config
        }
        
        # Add dataset sizes if available
        if self.train_dataset is not None:
            info['train_size'] = self.train_dataset.get_dataset_size()
        
        if self.val_dataset is not None:
            info['val_size'] = self.val_dataset.get_dataset_size()
        
        if self._test_dataset is not None:
            info['test_size'] = self._test_dataset.get_dataset_size()
        
        return info
    
    def create_single_dataset(
        self,
        split: str,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        **kwargs
    ) -> ds.Dataset:
        """
        Create a single dataset for specific requirements.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            batch_size: Override default batch size
            shuffle: Override default shuffle setting
            **kwargs: Additional dataset arguments
            
        Returns:
            MindSpore Dataset object
        """
        # Use provided values or defaults
        batch_size = batch_size or self.batch_size
        if shuffle is None:
            shuffle = (split == 'train')
        
        # Merge kwargs with defaults
        dataset_kwargs = {**self.dataset_kwargs, **kwargs}
        
        return create_nuscenes_dataset(
            dataset_dir=self.dataset_dir,
            labels_dir=self.labels_dir,
            split=split,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            transform_config=self.transform_config,
            **dataset_kwargs
        )

    @property
    def train_dataset(self):
        return self.train_dataset

    @property
    def val_dataset(self):
        return self.val_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @val_dataset.setter
    def val_dataset(self, value):
        self._val_dataset = value


def create_data_module(config: Dict[str, Any]) -> DataModule:
    """
    Create data module from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataModule instance
    """
    return DataModule(**config)