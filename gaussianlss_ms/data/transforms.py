"""
Data transformation utilities for GaussianLSS MindSpore implementation.

This module provides:
- Sample data structure
- Image preprocessing and augmentation
- BEV label generation from 3D bounding boxes
- Camera parameter handling
"""

import pathlib
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

from .common import INTERPOLATION, sincos2quaternion, get_camera_names
from nuscenes.utils.data_classes import Box

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class Sample(dict):
    """
    Data sample container for NuScenes data.
    
    This class extends dict to provide convenient access to sample data
    including images, camera parameters, and ground truth labels.
    """
    
    def __init__(
        self,
        token: str,
        scene: str, 
        map_name: str,
        intrinsics: List[np.ndarray],
        extrinsics: List[np.ndarray],
        images: List[str],
        view: np.ndarray,
        gt_box: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Sample identifiers
        self.token = token
        self.scene = scene
        self.map_name = map_name
        
        # Camera parameters
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.images = images
        self.view = view
        
        # Ground truth
        self.gt_box = gt_box
        
        # Additional data
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, key):
        """Allow attribute-style access to dictionary items."""
        try:
            return self[key]
        except KeyError:
            return super().__getitem__(key)


class BaseTransform:
    """Base class for data transformations."""
    
    def __init__(self):
        # MindSpore tensor conversion
        self.to_tensor = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, sample: Sample) -> Dict[str, Any]:
        raise NotImplementedError


class LoadDataTransform(BaseTransform):
    """
    Load and preprocess data for training/inference.
    
    This transform:
    1. Loads multi-view images
    2. Applies camera parameter transformations
    3. Generates BEV labels from 3D bounding boxes
    4. Applies data augmentation (if enabled)
    """
    
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        labels_dir: pathlib.Path,
        image_config: Dict[str, Any],
        bev_config: Dict[str, Any] = None,
        augment_config: Dict[str, Any] = None,
        vehicle: bool = True,
        ped: bool = True,
        image_data: bool = True
    ):
        super().__init__()
        
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.bev_config = bev_config or {}
        self.augment_config = augment_config or {}
        
        self.vehicle = vehicle
        self.ped = ped
        self.image_data = image_data
        
        # Image preprocessing parameters
        self.img_h = image_config.get('h', 224)
        self.img_w = image_config.get('w', 480)
        self.top_crop = image_config.get('top_crop', 46)
    
    def __call__(self, sample: Sample) -> Dict[str, Any]:
        """
        Process a data sample.
        
        Args:
            sample: Input data sample
            
        Returns:
            Dict containing processed data
        """
        result = {}
        
        # Load and process images
        if self.image_data:
            camera_data = self.get_cameras(sample)
            result.update(camera_data)
        
        # Generate BEV labels
        if self.vehicle:
            vehicle_bev, vehicle_center, vehicle_offset, vehicle_vis = \
                self.get_bev_from_gtbbox(sample, mode='vehicle')
            result.update({
                'vehicle': vehicle_bev,
                'vehicle_center': vehicle_center,
                'vehicle_offset': vehicle_offset,
                'vehicle_visibility': vehicle_vis
            })
        
        if self.ped:
            ped_bev, ped_center, ped_offset, ped_vis = \
                self.get_bev_from_gtbbox(sample, mode='ped')
            result.update({
                'ped': ped_bev,
                'ped_center': ped_center,
                'ped_offset': ped_offset,
                'ped_visibility': ped_vis
            })
        
        # Add metadata
        result.update({
            'token': sample.token,
            'scene': sample.scene,
            'map_name': sample.map_name
        })
        
        return result
    
    def get_cameras(self, sample: Sample) -> Dict[str, ms.Tensor]:
        """
        Load and preprocess multi-view camera images.
        
        Args:
            sample: Input data sample
            
        Returns:
            Dict containing processed camera data
        """
        images = []
        intrinsics = []
        lidar2img = []
        
        for image_path, I_original, extrinsic in zip(
            sample.images, sample.intrinsics, sample.extrinsics
        ):
            # Load and resize image
            image = Image.open(self.dataset_dir / image_path)
            h_resize = self.img_h + self.top_crop
            w_resize = self.img_w
            
            # Resize and crop
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, self.top_crop, w_resize, self.img_h + self.top_crop))
            
            # Convert to tensor
            image_tensor = self.to_tensor(image_new)
            images.append(image_tensor)
            
            # Adjust intrinsic matrix for resizing and cropping
            I = np.float32(I_original.copy())
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width  
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= self.top_crop
            
            intrinsics.append(ms.Tensor(I, dtype=ms.float32))
            
            # Compute lidar2img transformation
            viewpad = np.eye(4)
            viewpad[:I.shape[0], :I.shape[1]] = I
            lidar2img_matrix = viewpad @ extrinsic
            lidar2img.append(ms.Tensor(lidar2img_matrix, dtype=ms.float32))
        
        return {
            'image': ms.ops.stack(images, axis=0),
            'intrinsics': ms.ops.stack(intrinsics, axis=0),
            'lidar2img': ms.ops.stack(lidar2img, axis=0)
        }
    
    def get_bev_from_gtbbox(
        self, 
        sample: Sample, 
        mode: str = 'vehicle'
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """
        Generate BEV labels from 3D bounding boxes.
        
        Args:
            sample: Input data sample
            mode: 'vehicle' or 'ped' for different object types
            
        Returns:
            Tuple of (bev_mask, center_score, center_offset, visibility)
        """
        # Load ground truth boxes
        scene_dir = self.labels_dir / sample.scene
        gt_box_data = np.load(scene_dir / sample.gt_box, allow_pickle=True)['gt_box']
        V = sample.view
        
        # Initialize BEV grids (200x200 for 100m x 100m area)
        bev = np.zeros((200, 200), dtype=np.uint8)
        center_score = np.zeros((200, 200), dtype=np.float32)
        center_offset = np.zeros((200, 200, 2), dtype=np.float32)
        visibility = np.full((200, 200), 255, dtype=np.uint8)
        
        # Create coordinate grid for offset calculation
        coords = np.stack(np.meshgrid(np.arange(200), np.arange(200)), -1).astype(np.float32)
        sigma = 1.0  # Gaussian kernel standard deviation
        
        # Process each bounding box
        for box_data in gt_box_data:
            if len(box_data) == 0:
                continue
                
            class_idx = int(box_data[7])
            visibility_token = box_data[8]
            
            # Filter by object type
            if mode == 'vehicle' and class_idx == 5:  # Skip pedestrians
                continue
            elif mode == 'ped' and class_idx != 5:  # Only pedestrians
                continue
            
            # Extract box parameters
            translation = [box_data[0], box_data[1], box_data[4]]  # [x, y, z]
            size = [box_data[2], box_data[3], box_data[5]]  # [l, w, h]
            yaw = -box_data[6] - np.pi / 2  # Adjust yaw angle
            
            # Create 3D bounding box
            box = Box(translation, size, sincos2quaternion(np.sin(yaw), np.cos(yaw)))
            points = box.bottom_corners()  # Get bottom face corners
            
            # Project to BEV coordinates
            homog_points = np.ones((4, 4))
            homog_points[:3, :] = points
            bev_points = (V @ homog_points)[:2]  # Project to BEV
            
            # Fill polygon in BEV mask
            cv2.fillPoly(bev, [bev_points.round().astype(np.int32).T], 1, INTERPOLATION)
            
            # Compute center point in BEV
            center = points.mean(-1)[:, None]
            homog_center = np.ones((4, 1))
            homog_center[:3, :] = center
            bev_center = (V @ homog_center)[:2, 0].astype(np.float32)
            
            # Generate center point labels
            buf = np.zeros((200, 200), dtype=np.uint8)
            cv2.fillPoly(buf, [bev_points.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0
            
            # Compute offset vectors
            center_off = bev_center[None] - coords
            center_offset[mask] = center_off[mask]
            
            # Generate Gaussian heatmap for center
            gaussian_map = np.exp(-(center_off ** 2).sum(-1) / (2 * sigma ** 2))
            center_score = np.maximum(center_score, gaussian_map)
            
            # Set visibility
            visibility[mask] = visibility_token
        
        # Convert to MindSpore tensors
        bev_tensor = ms.Tensor(255 * bev, dtype=ms.float32)
        center_score_tensor = ms.Tensor(center_score, dtype=ms.float32)
        center_offset_tensor = ms.Tensor(center_offset, dtype=ms.float32)
        visibility_tensor = ms.Tensor(visibility, dtype=ms.uint8)
        
        return bev_tensor, center_score_tensor, center_offset_tensor, visibility_tensor


class SaveDataTransform(BaseTransform):
    """
    Transform for saving preprocessed data.
    
    This is used during data generation phase to save processed
    samples for faster loading during training.
    """
    
    def __init__(self, save_dir: pathlib.Path):
        super().__init__()
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, sample: Sample) -> Dict[str, Any]:
        """Save sample data to disk."""
        # Implementation for saving data
        # This would save processed images, labels, etc.
        pass