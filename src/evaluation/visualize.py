"""
Visualization utilities for dental caries detection results.
Provides functions for plotting training curves, predictions, and metrics.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

from src.config import SETTINGS


class ResultVisualizer:
    """
    Visualization utilities for training and inference results.
    """
    
    # Color scheme for visualizations
    COLORS = {
        "dental_caries": (255, 0, 0),  # Red for dental caries
        "background": (0, 255, 0),      # Green for background
    }
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations. Defaults to runs/visualizations.
        """
        self.save_dir = Path(save_dir) if save_dir else SETTINGS.runs_dir / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        results_dir: Path,
        save: bool = True,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot training curves from YOLO training results.
        
        Args:
            results_dir: Path to YOLO training results directory.
            save: Whether to save the figure.
            show: Whether to display the figure.
        
        Returns:
            Matplotlib figure if successful.
        """
        results_dir = Path(results_dir)
        results_csv = results_dir / "results.csv"
        
        if not results_csv.exists():
            print(f"Results CSV not found: {results_csv}")
            return None
        
        # Read training results
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Box Loss
        ax1 = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/box_loss'], label='Train', color='blue')
        if 'val/box_loss' in df.columns:
            ax1.plot(df['epoch'], df['val/box_loss'], label='Val', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Box Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Segmentation Loss
        ax2 = axes[0, 1]
        if 'train/seg_loss' in df.columns:
            ax2.plot(df['epoch'], df['train/seg_loss'], label='Train', color='blue')
        if 'val/seg_loss' in df.columns:
            ax2.plot(df['epoch'], df['val/seg_loss'], label='Val', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Segmentation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Classification Loss
        ax3 = axes[0, 2]
        if 'train/cls_loss' in df.columns:
            ax3.plot(df['epoch'], df['train/cls_loss'], label='Train', color='blue')
        if 'val/cls_loss' in df.columns:
            ax3.plot(df['epoch'], df['val/cls_loss'], label='Val', color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Classification Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: mAP@50
        ax4 = axes[1, 0]
        if 'metrics/mAP50(B)' in df.columns:
            ax4.plot(df['epoch'], df['metrics/mAP50(B)'], label='Box mAP50', color='blue')
        if 'metrics/mAP50(M)' in df.columns:
            ax4.plot(df['epoch'], df['metrics/mAP50(M)'], label='Mask mAP50', color='green')
        ax4.axhline(y=0.7, color='red', linestyle='--', label='Target (70%)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('mAP@50')
        ax4.set_title('mAP@50')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Plot 5: Precision & Recall
        ax5 = axes[1, 1]
        if 'metrics/precision(B)' in df.columns:
            ax5.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
        if 'metrics/recall(B)' in df.columns:
            ax5.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='orange')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Score')
        ax5.set_title('Precision & Recall')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
        
        # Plot 6: mAP@50-95
        ax6 = axes[1, 2]
        if 'metrics/mAP50-95(B)' in df.columns:
            ax6.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='Box mAP50-95', color='blue')
        if 'metrics/mAP50-95(M)' in df.columns:
            ax6.plot(df['epoch'], df['metrics/mAP50-95(M)'], label='Mask mAP50-95', color='green')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('mAP@50-95')
        ax6.set_title('mAP@50-95')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        plt.suptitle('Training Progress - Dental Caries Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "training_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_predictions(
        self,
        results,
        max_images: int = 9,
        save: bool = True,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Visualize model predictions with masks.
        
        Args:
            results: YOLO prediction results.
            max_images: Maximum number of images to show.
            save: Whether to save the figure.
            show: Whether to display the figure.
        
        Returns:
            Matplotlib figure.
        """
        num_images = min(len(results), max_images)
        cols = 3
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < num_images:
                result = results[idx]
                
                # Get original image
                img = result.orig_img
                if isinstance(img, np.ndarray):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Plot image
                ax.imshow(img)
                
                # Overlay masks if available
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    for mask in masks:
                        # Resize mask to image size
                        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                        
                        # Create colored overlay
                        overlay = np.zeros_like(img)
                        overlay[mask_resized > 0.5] = [255, 0, 0]  # Red for caries
                        
                        # Blend with original
                        ax.imshow(overlay, alpha=0.3)
                
                # Draw boxes if available
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
                        ax.text(
                            x1, y1 - 5, f'caries: {conf:.2f}',
                            color='red', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                        )
                
                # Get image name
                img_path = result.path if hasattr(result, 'path') else f"Image {idx}"
                ax.set_title(Path(img_path).name[:30] + "..." if len(str(img_path)) > 30 else str(img_path))
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle('Dental Caries Detection Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "predictions.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved predictions to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, float],
        target_mAP50: float = 0.70,
        save: bool = True,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot metrics comparison with target.
        
        Args:
            metrics_dict: Dictionary of metric names to values.
            target_mAP50: Target mAP@50 value.
            save: Whether to save the figure.
            show: Whether to display the figure.
        
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Bar chart of metrics
        ax1 = axes[0]
        metrics_to_plot = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
        values = [metrics_dict.get(m, 0) for m in metrics_to_plot]
        labels = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall', 'F1-Score']
        
        colors = ['green' if m == 'mAP50' and v >= target_mAP50 else 'steelblue' 
                  for m, v in zip(metrics_to_plot, values)]
        
        bars = ax1.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
        ax1.axhline(y=target_mAP50, color='red', linestyle='--', linewidth=2, label=f'Target ({target_mAP50*100:.0f}%)')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Metrics')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Precision-Recall trade-off
        ax2 = axes[1]
        precision = metrics_dict.get('precision', 0)
        recall = metrics_dict.get('recall', 0)
        f1 = metrics_dict.get('f1_score', 0)
        
        # Create a simple visualization
        categories = ['Precision', 'Recall', 'F1-Score']
        values_pr = [precision, recall, f1]
        
        # Radar-like bar chart
        theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values_pr_closed = values_pr + [values_pr[0]]
        theta_closed = np.append(theta, theta[0])
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(theta_closed, values_pr_closed, 'b-', linewidth=2)
        ax2.fill(theta_closed, values_pr_closed, alpha=0.25)
        ax2.set_xticks(theta)
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('Precision-Recall Balance')
        
        plt.suptitle('Dental Caries Detection - Performance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved metrics comparison to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str] = None,
        save: bool = True,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: 2D numpy array of confusion matrix.
            class_names: List of class names.
            save: Whether to save the figure.
            show: Whether to display the figure.
        
        Returns:
            Matplotlib figure.
        """
        if class_names is None:
            class_names = ['Background', 'Dental Caries']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(confusion_matrix, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        if show:
            plt.show()
        
        return fig


if __name__ == "__main__":
    # Example usage
    visualizer = ResultVisualizer()
    print(f"Visualizer ready. Save directory: {visualizer.save_dir}")
