"""
Dataset exploration and statistics for dental caries detection.
Provides utilities for analyzing the dataset structure, label distribution,
and image characteristics.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.config import SETTINGS


class DatasetExplorer:
    """Explore and analyze the dental caries dataset."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize dataset explorer.
        
        Args:
            data_dir: Path to dataset root. Defaults to config setting.
        """
        self.data_dir = Path(data_dir) if data_dir else SETTINGS.data_dir
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        
        self.stats = {}
        
    def get_image_paths(self, split: str = "train") -> List[Path]:
        """Get all image paths for a given split."""
        split_dir = self.train_dir if split == "train" else self.val_dir
        images_dir = split_dir / "images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(images_dir.glob(f"*{ext}"))
            image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_paths)
    
    def get_label_paths(self, split: str = "train") -> List[Path]:
        """Get all label paths for a given split."""
        split_dir = self.train_dir if split == "train" else self.val_dir
        labels_dir = split_dir / "labels"
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        return sorted(labels_dir.glob("*.txt"))
    
    def analyze_images(self, split: str = "train") -> Dict:
        """
        Analyze image statistics for a given split.
        
        Returns:
            Dictionary with image statistics.
        """
        image_paths = self.get_image_paths(split)
        
        widths = []
        heights = []
        aspect_ratios = []
        file_sizes = []
        
        print(f"Analyzing {split} images...")
        for img_path in tqdm(image_paths, desc=f"Processing {split} images"):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
                    file_sizes.append(img_path.stat().st_size / 1024)  # KB
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        stats = {
            "count": len(image_paths),
            "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths), "std": np.std(widths)},
            "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights), "std": np.std(heights)},
            "aspect_ratio": {"min": min(aspect_ratios), "max": max(aspect_ratios), "mean": np.mean(aspect_ratios)},
            "file_size_kb": {"min": min(file_sizes), "max": max(file_sizes), "mean": np.mean(file_sizes)},
        }
        
        return stats
    
    def analyze_labels(self, split: str = "train") -> Dict:
        """
        Analyze label statistics for a given split.
        
        Returns:
            Dictionary with label statistics.
        """
        label_paths = self.get_label_paths(split)
        
        class_counts = defaultdict(int)
        instances_per_image = []
        polygon_sizes = []  # Number of points per polygon
        
        print(f"Analyzing {split} labels...")
        for label_path in tqdm(label_paths, desc=f"Processing {split} labels"):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                instance_count = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:  # Minimum: class + 2 points (4 coords)
                        continue
                    
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    instance_count += 1
                    
                    # Count polygon points (each point is x,y pair)
                    num_coords = len(parts) - 1
                    num_points = num_coords // 2
                    polygon_sizes.append(num_points)
                
                instances_per_image.append(instance_count)
                
            except Exception as e:
                print(f"Error processing {label_path}: {e}")
        
        stats = {
            "num_labels": len(label_paths),
            "class_distribution": dict(class_counts),
            "total_instances": sum(class_counts.values()),
            "instances_per_image": {
                "min": min(instances_per_image) if instances_per_image else 0,
                "max": max(instances_per_image) if instances_per_image else 0,
                "mean": np.mean(instances_per_image) if instances_per_image else 0,
            },
            "polygon_points": {
                "min": min(polygon_sizes) if polygon_sizes else 0,
                "max": max(polygon_sizes) if polygon_sizes else 0,
                "mean": np.mean(polygon_sizes) if polygon_sizes else 0,
            },
        }
        
        return stats
    
    def get_full_stats(self) -> Dict:
        """Get complete dataset statistics."""
        self.stats = {
            "train": {
                "images": self.analyze_images("train"),
                "labels": self.analyze_labels("train"),
            },
            "val": {
                "images": self.analyze_images("val"),
                "labels": self.analyze_labels("val"),
            },
        }
        return self.stats
    
    def print_summary(self):
        """Print a formatted summary of dataset statistics."""
        if not self.stats:
            self.get_full_stats()
        
        print("\n" + "=" * 60)
        print("DENTAL CARIES DATASET SUMMARY")
        print("=" * 60)
        
        for split in ["train", "val"]:
            print(f"\n{'─' * 30}")
            print(f"  {split.upper()} SET")
            print(f"{'─' * 30}")
            
            img_stats = self.stats[split]["images"]
            lbl_stats = self.stats[split]["labels"]
            
            print(f"  Images: {img_stats['count']}")
            print(f"  Labels: {lbl_stats['num_labels']}")
            print(f"  Total Instances: {lbl_stats['total_instances']}")
            print(f"  Avg Instances/Image: {lbl_stats['instances_per_image']['mean']:.2f}")
            print(f"  Image Size: {img_stats['width']['mean']:.0f}x{img_stats['height']['mean']:.0f} (avg)")
            print(f"  Class Distribution: {lbl_stats['class_distribution']}")
        
        print("\n" + "=" * 60)
    
    def visualize_samples(self, split: str = "train", num_samples: int = 6, save_path: Optional[Path] = None):
        """
        Visualize sample images with their annotations.
        
        Args:
            split: Dataset split to visualize.
            num_samples: Number of samples to show.
            save_path: Optional path to save the figure.
        """
        image_paths = self.get_image_paths(split)
        
        # Select random samples
        np.random.seed(42)
        indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
        
        cols = 3
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                img_path = image_paths[indices[idx]]
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"{img_path.stem[:30]}...", fontsize=10)
                ax.axis("off")
            else:
                ax.axis("off")
        
        plt.suptitle(f"Sample Images from {split.upper()} Set", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def plot_statistics(self, save_path: Optional[Path] = None):
        """
        Plot dataset statistics.
        
        Args:
            save_path: Optional path to save the figure.
        """
        if not self.stats:
            self.get_full_stats()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Dataset split comparison
        ax1 = axes[0, 0]
        splits = ["Train", "Validation"]
        images = [self.stats["train"]["images"]["count"], self.stats["val"]["images"]["count"]]
        instances = [self.stats["train"]["labels"]["total_instances"], self.stats["val"]["labels"]["total_instances"]]
        
        x = np.arange(len(splits))
        width = 0.35
        ax1.bar(x - width/2, images, width, label="Images", color="steelblue")
        ax1.bar(x + width/2, instances, width, label="Instances", color="coral")
        ax1.set_xlabel("Dataset Split")
        ax1.set_ylabel("Count")
        ax1.set_title("Dataset Distribution")
        ax1.set_xticks(x)
        ax1.set_xticklabels(splits)
        ax1.legend()
        
        # Add value labels
        for i, (img, inst) in enumerate(zip(images, instances)):
            ax1.annotate(str(img), (i - width/2, img + 5), ha="center")
            ax1.annotate(str(inst), (i + width/2, inst + 5), ha="center")
        
        # Plot 2: Instances per image distribution
        ax2 = axes[0, 1]
        train_labels = self.get_label_paths("train")
        instances_per_img = []
        for lp in train_labels:
            with open(lp, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                instances_per_img.append(len(lines))
        
        ax2.hist(instances_per_img, bins=range(0, max(instances_per_img) + 2), 
                 edgecolor="black", color="steelblue", alpha=0.7)
        ax2.set_xlabel("Number of Instances")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Instances per Image Distribution (Train)")
        ax2.axvline(np.mean(instances_per_img), color="red", linestyle="--", 
                    label=f"Mean: {np.mean(instances_per_img):.2f}")
        ax2.legend()
        
        # Plot 3: Image size distribution
        ax3 = axes[1, 0]
        image_paths = self.get_image_paths("train")
        sizes = []
        for ip in image_paths[:100]:  # Sample first 100
            with Image.open(ip) as img:
                sizes.append(img.size)
        
        widths, heights = zip(*sizes)
        ax3.scatter(widths, heights, alpha=0.6, c="steelblue")
        ax3.set_xlabel("Width (pixels)")
        ax3.set_ylabel("Height (pixels)")
        ax3.set_title("Image Dimensions (Train Sample)")
        
        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis("off")
        
        summary_text = f"""
        DATASET SUMMARY
        {'─' * 30}
        
        Training Set:
          • Images: {self.stats['train']['images']['count']}
          • Total Instances: {self.stats['train']['labels']['total_instances']}
          • Avg Instances/Image: {self.stats['train']['labels']['instances_per_image']['mean']:.2f}
        
        Validation Set:
          • Images: {self.stats['val']['images']['count']}
          • Total Instances: {self.stats['val']['labels']['total_instances']}
          • Avg Instances/Image: {self.stats['val']['labels']['instances_per_image']['mean']:.2f}
        
        Classes: 1 (dental_caries)
        Format: YOLO Segmentation
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
                 verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle("Dental Caries Dataset Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved statistics plot to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Run dataset exploration
    explorer = DatasetExplorer()
    explorer.get_full_stats()
    explorer.print_summary()
    
    # Optional: Visualize samples and statistics
    # explorer.visualize_samples("train", num_samples=6)
    # explorer.plot_statistics()
