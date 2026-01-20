"""
Configuration settings for dental caries detection project.
Centralized hyperparameters, paths, and model configurations.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# Project root directory (ds_assignment/)
# settings.py is at ds_assignment/src/config/settings.py, so 3 parents up
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class Config:
    """Configuration class for training and evaluation."""
    
    # ==================== Paths ====================
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    train_dir: Path = DATA_DIR / "train"
    val_dir: Path = DATA_DIR / "val"
    runs_dir: Path = RUNS_DIR
    data_yaml: Path = SRC_DIR / "config" / "data.yaml"
    
    # ==================== Model ====================
    model_name: str = "yolo11m-seg"  # Upgraded from s to m for better performance
    pretrained: bool = True
    
    # ==================== Training Hyperparameters ====================
    epochs: int = 100
    batch_size: int = 8  # Reduced for 1024px images on Tesla T4/M1
    imgsz: int = 1024    # Increased from 640 to 1024 for finer detail
    patience: int = 25   # Slightly more patience for higher res
    
    # ==================== Preprocessing ====================
    use_clahe: bool = True  # Enable Contrast Limited Adaptive Histogram Equalization
    
    # Learning rate
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate (fraction of lr0)
    
    # Optimizer
    optimizer: str = "AdamW"  # Options: SGD, Adam, AdamW
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # ==================== Augmentation ====================
    # Geometric augmentations
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7    # HSV-Saturation augmentation
    hsv_v: float = 0.4    # HSV-Value augmentation
    degrees: float = 10.0  # Rotation degrees (±)
    translate: float = 0.1  # Translation fraction
    scale: float = 0.5     # Scale augmentation
    shear: float = 0.0     # Shear augmentation
    perspective: float = 0.0  # Perspective augmentation
    
    # Flip augmentations
    flipud: float = 0.0   # Vertical flip (disabled for dental X-rays)
    fliplr: float = 0.5   # Horizontal flip
    
    # Mosaic and mixup
    mosaic: float = 1.0   # Mosaic augmentation
    mixup: float = 0.0    # Mixup augmentation
    copy_paste: float = 0.0  # Copy-paste augmentation
    
    # ==================== Validation ====================
    conf_threshold: float = 0.25  # Confidence threshold for predictions
    iou_threshold: float = 0.7    # IoU threshold for NMS
    
    # ==================== Hardware ====================
    device: Optional[str] = None  # Auto-detect: cuda, mps, or cpu
    workers: int = 8  # Number of dataloader workers
    
    # ==================== Logging ====================
    project_name: str = "dental_caries"
    experiment_name: str = "train"
    verbose: bool = True
    save_period: int = -1  # Save checkpoint every N epochs (-1 for only best/last)
    
    def __post_init__(self):
        """Validate and create directories."""
        # Create runs directory if it doesn't exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if not specified
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    def to_train_args(self) -> dict:
        """Convert config to YOLO training arguments."""
        return {
            "data": str(self.data_yaml),
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.imgsz,
            "patience": self.patience,
            "device": self.device,
            "workers": self.workers,
            "project": str(self.runs_dir / "segment"),
            "name": self.experiment_name,
            "pretrained": self.pretrained,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "verbose": self.verbose,
            "save_period": self.save_period,
        }
    
    def to_val_args(self) -> dict:
        """Convert config to YOLO validation arguments."""
        return {
            "data": str(self.data_yaml),
            "imgsz": self.imgsz,
            "batch": self.batch_size,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "device": self.device,
            "verbose": self.verbose,
        }


# Default settings instance
SETTINGS = Config()


if __name__ == "__main__":
    # Print configuration for verification
    print("=" * 50)
    print("Dental Caries Detection Configuration")
    print("=" * 50)
    print(f"\nProject Root: {SETTINGS.project_root}")
    print(f"Data Directory: {SETTINGS.data_dir}")
    print(f"Device: {SETTINGS.device}")
    print(f"\nModel: {SETTINGS.model_name}")
    print(f"Epochs: {SETTINGS.epochs}")
    print(f"Batch Size: {SETTINGS.batch_size}")
    print(f"Image Size: {SETTINGS.imgsz}")
    print(f"\nTraining Args: {SETTINGS.to_train_args()}")
