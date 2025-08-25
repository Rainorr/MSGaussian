#!/usr/bin/env python3
"""
Data preparation script for GaussianLSS MindSpore implementation.

This script preprocesses NuScenes data and creates the necessary JSON files
for training and validation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from gaussianlss_ms.data.transforms import Sample
from gaussianlss_ms.data.common import get_camera_names


def get_camera_intrinsics(nusc: NuScenes, cam_token: str) -> np.ndarray:
    """Get camera intrinsic matrix."""
    cam = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    return np.array(cs_record['camera_intrinsic'])


def get_camera_extrinsics(nusc: NuScenes, cam_token: str, ego_pose_token: str) -> np.ndarray:
    """Get camera extrinsic matrix (world to camera)."""
    cam = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    
    # Camera to ego transformation
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
    cam_to_ego[:3, 3] = cs_record['translation']
    
    # Ego to world transformation
    ego_to_world = np.eye(4)
    ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
    ego_to_world[:3, 3] = ego_pose['translation']
    
    # World to camera transformation
    world_to_cam = np.linalg.inv(cam_to_ego @ ego_to_world)
    
    return world_to_cam


def get_boxes_in_camera(nusc: NuScenes, sample_token: str, cam_token: str) -> List[Dict[str, Any]]:
    """Get 3D boxes visible in camera."""
    sample = nusc.get('sample', sample_token)
    cam = nusc.get('sample_data', cam_token)
    
    # Get boxes
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # Skip if not a vehicle or pedestrian
        if not any(name in ann['category_name'] for name in ['vehicle', 'human.pedestrian']):
            continue
        
        # Create box
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        
        # Check if box is in camera view
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam['ego_pose_token'])
        
        # Transform box to camera coordinates
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        # Check if box is in front of camera
        if box.center[2] > 0:
            # Get camera intrinsics
            intrinsics = np.array(cs_record['camera_intrinsic'])
            
            # Project box to image
            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, intrinsics, normalize=True)[:2, :]
            
            # Check if any corner is in image
            if np.any((corners_2d[0, :] >= 0) & (corners_2d[0, :] < 1600) &
                     (corners_2d[1, :] >= 0) & (corners_2d[1, :] < 900)):
                
                # Determine class
                if 'vehicle' in ann['category_name']:
                    class_id = 0
                elif 'human.pedestrian' in ann['category_name']:
                    class_id = 1
                else:
                    continue
                
                boxes.append({
                    'center': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.q.tolist(),
                    'class_id': class_id,
                    'category': ann['category_name'],
                    'corners_2d': corners_2d.tolist()
                })
    
    return boxes


def process_sample(nusc: NuScenes, sample_token: str) -> Dict[str, Any]:
    """Process a single sample."""
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    
    # Get camera data
    camera_names = get_camera_names()
    intrinsics = []
    extrinsics = []
    images = []
    gt_boxes = []
    
    for cam_name in camera_names:
        if cam_name in sample['data']:
            cam_token = sample['data'][cam_name]
            cam = nusc.get('sample_data', cam_token)
            
            # Get intrinsics and extrinsics
            intrinsics.append(get_camera_intrinsics(nusc, cam_token).tolist())
            extrinsics.append(get_camera_extrinsics(nusc, cam_token, cam['ego_pose_token']).tolist())
            
            # Get image path (relative to dataset root)
            images.append(cam['filename'])
            
            # Get boxes in this camera
            boxes = get_boxes_in_camera(nusc, sample_token, cam_token)
            gt_boxes.append(boxes)
    
    # Create sample data
    sample_data = {
        'token': sample_token,
        'scene': scene['name'],
        'map_name': nusc.get('log', scene['log_token'])['location'],
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'images': images,
        'view': np.eye(4).tolist(),  # Identity for now
        'gt_box': gt_boxes
    }
    
    return sample_data


def prepare_nuscenes_data(dataset_dir: str, output_dir: str, version: str = 'v1.0-mini'):
    """Prepare NuScenes data for training."""
    print(f"Loading NuScenes {version} from {dataset_dir}")
    
    # Initialize NuScenes
    nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=True)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each scene
    for scene in nusc.scene:
        print(f"Processing scene: {scene['name']}")
        
        scene_samples = []
        sample_token = scene['first_sample_token']
        
        while sample_token:
            try:
                sample_data = process_sample(nusc, sample_token)
                scene_samples.append(sample_data)
                
                # Get next sample
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next']
                
            except Exception as e:
                print(f"Error processing sample {sample_token}: {e}")
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next']
                continue
        
        # Save scene data
        if scene_samples:
            scene_file = output_path / f"scene-{scene['name']}.json"
            with open(scene_file, 'w') as f:
                json.dump(scene_samples, f, indent=2)
            
            print(f"Saved {len(scene_samples)} samples to {scene_file}")
    
    print(f"Data preparation complete. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare NuScenes data for GaussianLSS training')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to NuScenes dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory for processed data')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       help='NuScenes version (default: v1.0-mini)')
    
    args = parser.parse_args()
    
    prepare_nuscenes_data(args.dataset_dir, args.output_dir, args.version)


if __name__ == '__main__':
    main()