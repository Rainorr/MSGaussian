"""
Detection metrics for GaussianLSS MindSpore implementation.

This module implements metrics for 3D object detection evaluation.
"""

import mindspore as ms
import mindspore.ops as ops
import numpy as np
from typing import Dict, List, Tuple, Any


class DetectionMetrics:
    """
    Detection metrics for 3D object detection evaluation.
    
    Computes metrics like Average Precision (AP), Average Recall (AR),
    and other detection-specific metrics.
    """
    
    def __init__(self,
                 score_threshold: float = 0.1,
                 nms_threshold: float = 0.5,
                 max_detections: int = 100,
                 distance_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]):
        """
        Initialize detection metrics.
        
        Args:
            score_threshold (float): Minimum score threshold for detections
            nms_threshold (float): NMS IoU threshold
            max_detections (int): Maximum number of detections per image
            distance_thresholds (List[float]): Distance thresholds for AP calculation
        """
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.distance_thresholds = distance_thresholds
        
        # Accumulate predictions and targets for batch evaluation
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: Dict[str, ms.Tensor], targets: Dict[str, ms.Tensor]):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions (Dict[str, ms.Tensor]): Model predictions
            targets (Dict[str, ms.Tensor]): Ground truth targets
        """
        # Convert to numpy for easier processing
        pred_dict = {k: v.asnumpy() if isinstance(v, ms.Tensor) else v 
                    for k, v in predictions.items()}
        target_dict = {k: v.asnumpy() if isinstance(v, ms.Tensor) else v 
                      for k, v in targets.items()}
        
        self.predictions.append(pred_dict)
        self.targets.append(target_dict)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute detection metrics.
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        if not self.predictions or not self.targets:
            return {}
        
        # Extract detections from predictions
        all_detections = []
        all_ground_truths = []
        
        for pred, target in zip(self.predictions, self.targets):
            detections = self._extract_detections(pred)
            ground_truths = self._extract_ground_truths(target)
            
            all_detections.append(detections)
            all_ground_truths.append(ground_truths)
        
        # Compute AP for different distance thresholds
        metrics = {}
        for dist_thresh in self.distance_thresholds:
            ap = self._compute_average_precision(all_detections, all_ground_truths, dist_thresh)
            metrics[f'AP_{dist_thresh}'] = ap
        
        # Compute mean AP
        metrics['mAP'] = np.mean([metrics[f'AP_{thresh}'] for thresh in self.distance_thresholds])
        
        # Compute other metrics
        metrics.update(self._compute_additional_metrics(all_detections, all_ground_truths))
        
        return metrics
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []
    
    def _extract_detections(self, predictions: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract detections from model predictions.
        
        Args:
            predictions (Dict[str, np.ndarray]): Model predictions
            
        Returns:
            List[Dict[str, Any]]: List of detections
        """
        detections = []
        
        # Get heatmap and extract peaks
        heatmap = predictions.get('heatmap', np.zeros((1, 224, 480)))  # [C, H, W]
        centers = predictions.get('centers', np.zeros((3, 224, 480)))  # [3, H, W]
        offsets = predictions.get('offsets', np.zeros((2, 224, 480)))  # [2, H, W]
        
        # Find peaks in heatmap
        for class_id in range(heatmap.shape[0]):
            class_heatmap = heatmap[class_id]
            
            # Simple peak detection (can be improved with NMS)
            peaks = self._find_peaks(class_heatmap, threshold=self.score_threshold)
            
            for peak_y, peak_x in peaks:
                score = class_heatmap[peak_y, peak_x]
                
                # Get 3D center
                center_3d = centers[:, peak_y, peak_x]
                offset_2d = offsets[:, peak_y, peak_x]
                
                # Adjust 2D position with offset
                adjusted_x = peak_x + offset_2d[0]
                adjusted_y = peak_y + offset_2d[1]
                
                detection = {
                    'class_id': class_id,
                    'score': float(score),
                    'center_3d': center_3d.tolist(),
                    'center_2d': [float(adjusted_x), float(adjusted_y)],
                    'bbox_2d': [float(adjusted_x-10), float(adjusted_y-10), 
                               float(adjusted_x+10), float(adjusted_y+10)]  # Simple bbox
                }
                detections.append(detection)
        
        # Sort by score and keep top detections
        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections[:self.max_detections]
    
    def _extract_ground_truths(self, targets: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract ground truth objects from targets.
        
        Args:
            targets (Dict[str, np.ndarray]): Ground truth targets
            
        Returns:
            List[Dict[str, Any]]: List of ground truth objects
        """
        ground_truths = []
        
        # Extract from targets (format depends on your data structure)
        # This is a simplified version - adjust based on your target format
        if 'gt_boxes' in targets and 'gt_labels' in targets:
            gt_boxes = targets['gt_boxes']  # [N, 7] (x, y, z, w, l, h, yaw)
            gt_labels = targets['gt_labels']  # [N]
            
            for i in range(len(gt_boxes)):
                if gt_labels[i] >= 0:  # Valid object
                    ground_truth = {
                        'class_id': int(gt_labels[i]),
                        'center_3d': gt_boxes[i][:3].tolist(),
                        'size_3d': gt_boxes[i][3:6].tolist(),
                        'yaw': float(gt_boxes[i][6]) if gt_boxes.shape[1] > 6 else 0.0
                    }
                    ground_truths.append(ground_truth)
        
        return ground_truths
    
    def _find_peaks(self, heatmap: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
        """
        Find peaks in heatmap.
        
        Args:
            heatmap (np.ndarray): Input heatmap [H, W]
            threshold (float): Minimum threshold for peaks
            
        Returns:
            List[Tuple[int, int]]: List of peak coordinates (y, x)
        """
        from scipy.ndimage import maximum_filter
        
        # Apply maximum filter to find local maxima
        local_maxima = maximum_filter(heatmap, size=3) == heatmap
        
        # Apply threshold
        above_threshold = heatmap > threshold
        
        # Combine conditions
        peaks_mask = local_maxima & above_threshold
        
        # Get peak coordinates
        peak_coords = np.where(peaks_mask)
        peaks = list(zip(peak_coords[0], peak_coords[1]))
        
        return peaks
    
    def _compute_average_precision(self, 
                                 all_detections: List[List[Dict[str, Any]]], 
                                 all_ground_truths: List[List[Dict[str, Any]]], 
                                 distance_threshold: float) -> float:
        """
        Compute Average Precision for given distance threshold.
        
        Args:
            all_detections (List[List[Dict]]): Detections for all images
            all_ground_truths (List[List[Dict]]): Ground truths for all images
            distance_threshold (float): Distance threshold for matching
            
        Returns:
            float: Average Precision
        """
        # Collect all detections and ground truths
        all_dets = []
        all_gts = []
        
        for img_idx, (dets, gts) in enumerate(zip(all_detections, all_ground_truths)):
            for det in dets:
                det_copy = det.copy()
                det_copy['image_id'] = img_idx
                all_dets.append(det_copy)
            
            for gt in gts:
                gt_copy = gt.copy()
                gt_copy['image_id'] = img_idx
                all_gts.append(gt_copy)
        
        if not all_dets or not all_gts:
            return 0.0
        
        # Sort detections by score
        all_dets.sort(key=lambda x: x['score'], reverse=True)
        
        # Match detections to ground truths
        tp = np.zeros(len(all_dets))
        fp = np.zeros(len(all_dets))
        
        # Group ground truths by image
        gt_by_image = {}
        for gt in all_gts:
            img_id = gt['image_id']
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(gt)
        
        # Track which ground truths have been matched
        gt_matched = {i: [False] * len(gts) for i, gts in gt_by_image.items()}
        
        for det_idx, det in enumerate(all_dets):
            img_id = det['image_id']
            
            if img_id not in gt_by_image:
                fp[det_idx] = 1
                continue
            
            # Find best matching ground truth
            best_distance = float('inf')
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_by_image[img_id]):
                if gt_matched[img_id][gt_idx]:
                    continue
                
                # Compute distance between detection and ground truth
                distance = self._compute_3d_distance(det['center_3d'], gt['center_3d'])
                
                if distance < best_distance and distance <= distance_threshold:
                    best_distance = distance
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                tp[det_idx] = 1
                gt_matched[img_id][best_gt_idx] = True
            else:
                fp[det_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / max(len(all_gts), 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap
    
    def _compute_3d_distance(self, center1: List[float], center2: List[float]) -> float:
        """
        Compute 3D Euclidean distance between two centers.
        
        Args:
            center1 (List[float]): First center [x, y, z]
            center2 (List[float]): Second center [x, y, z]
            
        Returns:
            float: 3D distance
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(center1, center2)))
    
    def _compute_additional_metrics(self, 
                                  all_detections: List[List[Dict[str, Any]]], 
                                  all_ground_truths: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Compute additional detection metrics.
        
        Args:
            all_detections (List[List[Dict]]): All detections
            all_ground_truths (List[List[Dict]]): All ground truths
            
        Returns:
            Dict[str, float]: Additional metrics
        """
        total_detections = sum(len(dets) for dets in all_detections)
        total_ground_truths = sum(len(gts) for gts in all_ground_truths)
        
        metrics = {
            'total_detections': float(total_detections),
            'total_ground_truths': float(total_ground_truths),
            'detection_rate': float(total_detections) / max(total_ground_truths, 1)
        }
        
        return metrics