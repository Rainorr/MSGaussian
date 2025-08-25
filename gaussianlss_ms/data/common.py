"""
Common utilities for data processing in MindSpore implementation.
"""

import numpy as np
import cv2

# Interpolation method for OpenCV operations
INTERPOLATION = cv2.INTER_LINEAR

def sincos2quaternion(sin_yaw, cos_yaw):
    """
    Convert sin/cos yaw to quaternion representation.
    
    Args:
        sin_yaw (float): Sine of yaw angle
        cos_yaw (float): Cosine of yaw angle
        
    Returns:
        list: Quaternion [w, x, y, z]
    """
    # For rotation around Z-axis (yaw)
    # q = [cos(θ/2), 0, 0, sin(θ/2)]
    half_angle_cos = np.sqrt((1 + cos_yaw) / 2)
    half_angle_sin = np.sqrt((1 - cos_yaw) / 2)
    
    # Determine sign of sin component based on original sin_yaw
    if sin_yaw < 0:
        half_angle_sin = -half_angle_sin
        
    return [half_angle_cos, 0, 0, half_angle_sin]

def get_camera_names():
    """Get standard NuScenes camera names."""
    return [
        'CAM_FRONT_LEFT',
        'CAM_FRONT', 
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT'
    ]

def get_nusc_map_names():
    """Get NuScenes map names."""
    return [
        'boston-seaport',
        'singapore-onenorth', 
        'singapore-hollandvillage',
        'singapore-queenstown'
    ]

def get_map_layers():
    """Get map layer names for NuScenes."""
    return [
        'lane', 'road_segment',
        'road_divider', 'lane_divider', 
        'ped_crossing', 'walkway', 'carpark_area'
    ]

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def rotation_matrix_from_yaw(yaw):
    """
    Create 2D rotation matrix from yaw angle.
    
    Args:
        yaw (float): Yaw angle in radians
        
    Returns:
        np.ndarray: 2x2 rotation matrix
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    return np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

def transform_points_2d(points, translation, rotation_matrix):
    """
    Transform 2D points with translation and rotation.
    
    Args:
        points (np.ndarray): Points to transform [N, 2]
        translation (np.ndarray): Translation vector [2]
        rotation_matrix (np.ndarray): 2x2 rotation matrix
        
    Returns:
        np.ndarray: Transformed points [N, 2]
    """
    # Apply rotation then translation
    rotated = points @ rotation_matrix.T
    return rotated + translation[None, :]