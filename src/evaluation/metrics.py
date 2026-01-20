"""
Metrics calculation for dental caries detection evaluation.
Provides comprehensive metrics including mAP, Precision, Recall, F1-Score,
and clinical metrics like Sensitivity and Specificity.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class SegmentationMetrics:
    """Container for segmentation evaluation metrics."""
    
    # Primary metrics
    mAP50: float = 0.0           # mAP at IoU=0.50
    mAP50_95: float = 0.0        # mAP at IoU=0.50:0.95
    
    # Precision and Recall
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Per-class metrics
    class_precision: Dict[str, float] = None
    class_recall: Dict[str, float] = None
    class_ap50: Dict[str, float] = None
    
    # Box metrics (detection)
    box_mAP50: float = 0.0
    box_mAP50_95: float = 0.0
    box_precision: float = 0.0
    box_recall: float = 0.0
    
    # Mask metrics (segmentation)
    mask_mAP50: float = 0.0
    mask_mAP50_95: float = 0.0
    mask_precision: float = 0.0
    mask_recall: float = 0.0
    
    # Clinical metrics
    sensitivity: float = 0.0      # Same as recall (True Positive Rate)
    specificity: float = 0.0      # True Negative Rate
    
    # Count metrics
    total_predictions: int = 0
    total_ground_truth: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.class_precision is None:
            self.class_precision = {}
        if self.class_recall is None:
            self.class_recall = {}
        if self.class_ap50 is None:
            self.class_ap50 = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def summary(self) -> str:
        """Generate a formatted summary string."""
        lines = [
            "=" * 50,
            "SEGMENTATION METRICS SUMMARY",
            "=" * 50,
            "",
            "Primary Metrics:",
            f"  mAP@50:      {self.mAP50:.4f} ({self.mAP50*100:.2f}%)",
            f"  mAP@50-95:   {self.mAP50_95:.4f} ({self.mAP50_95*100:.2f}%)",
            "",
            "Precision & Recall:",
            f"  Precision:   {self.precision:.4f} ({self.precision*100:.2f}%)",
            f"  Recall:      {self.recall:.4f} ({self.recall*100:.2f}%)",
            f"  F1-Score:    {self.f1_score:.4f} ({self.f1_score*100:.2f}%)",
            "",
            "Box Metrics (Detection):",
            f"  Box mAP@50:  {self.box_mAP50:.4f}",
            f"  Box P/R:     {self.box_precision:.4f} / {self.box_recall:.4f}",
            "",
            "Mask Metrics (Segmentation):",
            f"  Mask mAP@50: {self.mask_mAP50:.4f}",
            f"  Mask P/R:    {self.mask_precision:.4f} / {self.mask_recall:.4f}",
            "",
            "Clinical Metrics:",
            f"  Sensitivity: {self.sensitivity:.4f} (True Positive Rate)",
            f"  Specificity: {self.specificity:.4f} (True Negative Rate)",
            "",
            "Counts:",
            f"  True Positives:  {self.true_positives}",
            f"  False Positives: {self.false_positives}",
            f"  False Negatives: {self.false_negatives}",
            "=" * 50,
        ]
        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculate comprehensive metrics from YOLO validation results.
    
    Extracts and computes metrics relevant for dental caries detection,
    with emphasis on clinical diagnostic metrics.
    """
    
    CLASS_NAMES = {0: "dental_caries"}
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics: Optional[SegmentationMetrics] = None
        self.raw_results = None
    
    def calculate_from_yolo_results(self, results) -> SegmentationMetrics:
        """
        Calculate metrics from YOLO validation results.
        
        Args:
            results: Results object from model.val()
        
        Returns:
            SegmentationMetrics instance with all metrics.
        """
        self.raw_results = results
        
        # Extract results metrics
        # YOLO stores box and mask metrics separately
        
        metrics = SegmentationMetrics()
        
        # Check if results has the expected attributes
        if hasattr(results, 'box'):
            box = results.box
            metrics.box_mAP50 = float(box.map50) if hasattr(box, 'map50') else 0.0
            metrics.box_mAP50_95 = float(box.map) if hasattr(box, 'map') else 0.0
            
            # Get precision and recall from box results
            if hasattr(box, 'p') and len(box.p) > 0:
                metrics.box_precision = float(np.mean(box.p))
            if hasattr(box, 'r') and len(box.r) > 0:
                metrics.box_recall = float(np.mean(box.r))
        
        if hasattr(results, 'seg'):
            seg = results.seg
            metrics.mask_mAP50 = float(seg.map50) if hasattr(seg, 'map50') else 0.0
            metrics.mask_mAP50_95 = float(seg.map) if hasattr(seg, 'map') else 0.0
            
            # Get precision and recall from mask results
            if hasattr(seg, 'p') and len(seg.p) > 0:
                metrics.mask_precision = float(np.mean(seg.p))
            if hasattr(seg, 'r') and len(seg.r) > 0:
                metrics.mask_recall = float(np.mean(seg.r))
        
        # Set primary metrics (use mask metrics for segmentation task)
        metrics.mAP50 = metrics.mask_mAP50 if metrics.mask_mAP50 > 0 else metrics.box_mAP50
        metrics.mAP50_95 = metrics.mask_mAP50_95 if metrics.mask_mAP50_95 > 0 else metrics.box_mAP50_95
        metrics.precision = metrics.mask_precision if metrics.mask_precision > 0 else metrics.box_precision
        metrics.recall = metrics.mask_recall if metrics.mask_recall > 0 else metrics.box_recall
        
        # Calculate F1 score
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        
        # Clinical metrics
        metrics.sensitivity = metrics.recall  # Sensitivity = Recall = TPR
        
        # Per-class metrics
        if hasattr(results, 'names'):
            for class_id, class_name in results.names.items():
                if hasattr(results, 'box') and hasattr(results.box, 'ap50'):
                    if len(results.box.ap50) > class_id:
                        metrics.class_ap50[class_name] = float(results.box.ap50[class_id])
        
        self.metrics = metrics
        return metrics
    
    def calculate_from_predictions(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> SegmentationMetrics:
        """
        Calculate metrics from raw predictions and ground truths.
        
        Args:
            predictions: List of prediction dictionaries.
            ground_truths: List of ground truth dictionaries.
            iou_threshold: IoU threshold for matching.
        
        Returns:
            SegmentationMetrics instance.
        """
        metrics = SegmentationMetrics()
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, gt in zip(predictions, ground_truths):
            # Count matches based on IoU threshold
            # This is a simplified implementation
            pred_boxes = pred.get("boxes", [])
            gt_boxes = gt.get("boxes", [])
            
            matched = set()
            for p_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx in matched:
                        continue
                    iou = self._calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched.add(best_gt_idx)
                else:
                    total_fp += 1
            
            total_fn += len(gt_boxes) - len(matched)
        
        metrics.true_positives = total_tp
        metrics.false_positives = total_fp
        metrics.false_negatives = total_fn
        metrics.total_predictions = total_tp + total_fp
        metrics.total_ground_truth = total_tp + total_fn
        
        # Calculate precision, recall, F1
        if total_tp + total_fp > 0:
            metrics.precision = total_tp / (total_tp + total_fp)
        if total_tp + total_fn > 0:
            metrics.recall = total_tp / (total_tp + total_fn)
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        
        metrics.sensitivity = metrics.recall
        
        self.metrics = metrics
        return metrics
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def save_metrics(self, save_path: Path):
        """Save metrics to JSON file."""
        if self.metrics is None:
            raise ValueError("No metrics calculated. Run calculate_* first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        print(f"Metrics saved to {save_path}")
    
    def print_summary(self):
        """Print metrics summary."""
        if self.metrics is None:
            raise ValueError("No metrics calculated. Run calculate_* first.")
        
        print(self.metrics.summary())
    
    def meets_target(self, target_mAP50: float = 0.70) -> Tuple[bool, str]:
        """
        Check if metrics meet the target mAP@50.
        
        Args:
            target_mAP50: Target mAP@50 threshold (default 0.70 for 70%).
        
        Returns:
            Tuple of (meets_target, message)
        """
        if self.metrics is None:
            return False, "No metrics calculated"
        
        meets = self.metrics.mAP50 >= target_mAP50
        
        if meets:
            message = f"✅ Target met! mAP@50: {self.metrics.mAP50:.4f} >= {target_mAP50:.4f}"
        else:
            gap = target_mAP50 - self.metrics.mAP50
            message = f"❌ Target not met. mAP@50: {self.metrics.mAP50:.4f} < {target_mAP50:.4f} (gap: {gap:.4f})"
        
        return meets, message


def evaluate_model(weights_path: str, data_yaml: str) -> SegmentationMetrics:
    """
    Quick evaluation function.
    
    Args:
        weights_path: Path to model weights.
        data_yaml: Path to data configuration.
    
    Returns:
        SegmentationMetrics instance.
    """
    from ultralytics import YOLO
    
    model = YOLO(weights_path)
    results = model.val(data=data_yaml)
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_from_yolo_results(results)
    calculator.print_summary()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Metrics Calculator ready.")
    print("Use evaluate_model(weights_path, data_yaml) for quick evaluation.")
