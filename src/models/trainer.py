"""
YOLOv11 Segmentation Training Pipeline for Dental Caries Detection.
Handles model initialization, training, and checkpointing.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ultralytics import YOLO

from src.config import SETTINGS, Config
from src.data.augmentation import AugmentationConfig, get_augmentation_config


class DentalCariesTrainer:
    """
    Training pipeline for dental caries segmentation using YOLOv11.
    
    Features:
    - Easy initialization with pretrained models
    - Configurable hyperparameters
    - Training with callbacks and logging
    - Model export capabilities
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        augmentation: Optional[AugmentationConfig] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration. Defaults to SETTINGS.
            augmentation: Augmentation configuration. Defaults to medical preset.
        """
        self.config = config or SETTINGS
        self.augmentation = augmentation or get_augmentation_config("medical")
        self.model: Optional[YOLO] = None
        self.results = None
        
    def load_model(self, weights: Optional[str] = None) -> YOLO:
        """
        Load YOLOv11 segmentation model.
        
        Args:
            weights: Path to weights file or model name.
                    If None, loads pretrained model from config.
        
        Returns:
            YOLO model instance.
        """
        if weights is None:
            # Load pretrained model
            model_name = self.config.model_name
            if not model_name.endswith(".pt"):
                model_name = f"{model_name}.pt"
            print(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        else:
            # Load from weights file
            weights_path = Path(weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            print(f"Loading model from: {weights_path}")
            self.model = YOLO(str(weights_path))
        
        return self.model
    
    def train(
        self,
        resume: bool = False,
        experiment_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on dental caries dataset.
        
        Args:
            resume: Whether to resume from last checkpoint.
            experiment_name: Custom experiment name for this run.
            **kwargs: Additional training arguments to override config.
        
        Returns:
            Training results dictionary.
        """
        if self.model is None:
            self.load_model()
        
        # Build training arguments
        train_args = self.config.to_train_args()
        
        # Add augmentation settings
        train_args.update(self.augmentation.to_dict())
        
        # Override with custom experiment name
        if experiment_name:
            train_args["name"] = experiment_name
        elif "name" not in kwargs:
            # Generate unique name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_args["name"] = f"train_{timestamp}"
        
        # Override with any additional kwargs
        train_args.update(kwargs)
        
        # Handle resume
        if resume:
            train_args["resume"] = True
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {train_args['epochs']}")
        print(f"Batch Size: {train_args['batch']}")
        print(f"Image Size: {train_args['imgsz']}")
        print(f"Device: {train_args['device']}")
        print(f"Experiment: {train_args['name']}")
        print("=" * 60 + "\n")
        
        # Start training
        self.results = self.model.train(**train_args)
        
        return self.results
    
    def validate(self, weights: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Validate the model on validation set.
        
        Args:
            weights: Path to weights file. If None, uses current model.
            **kwargs: Additional validation arguments.
        
        Returns:
            Validation results dictionary.
        """
        if weights:
            self.load_model(weights)
        elif self.model is None:
            raise ValueError("No model loaded. Call load_model() or provide weights path.")
        
        # Build validation arguments
        val_args = self.config.to_val_args()
        val_args.update(kwargs)
        
        print("\n" + "=" * 60)
        print("RUNNING VALIDATION")
        print("=" * 60)
        
        results = self.model.val(**val_args)
        
        return results
    
    def predict(
        self,
        source: str,
        weights: Optional[str] = None,
        save: bool = True,
        conf: Optional[float] = None,
        **kwargs
    ):
        """
        Run prediction on images.
        
        Args:
            source: Path to image, directory, or video.
            weights: Path to weights file. If None, uses current model.
            save: Whether to save predictions.
            conf: Confidence threshold.
            **kwargs: Additional prediction arguments.
        
        Returns:
            Prediction results.
        """
        if weights:
            self.load_model(weights)
        elif self.model is None:
            raise ValueError("No model loaded. Call load_model() or provide weights path.")
        
        predict_args = {
            "source": source,
            "save": save,
            "conf": conf or self.config.conf_threshold,
            "iou": self.config.iou_threshold,
            "device": self.config.device,
            **kwargs
        }
        
        results = self.model.predict(**predict_args)
        
        return results
    
    def export(
        self,
        weights: Optional[str] = None,
        format: str = "onnx",
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            weights: Path to weights file.
            format: Export format (onnx, torchscript, tensorrt, etc.)
            **kwargs: Additional export arguments.
        
        Returns:
            Path to exported model.
        """
        if weights:
            self.load_model(weights)
        elif self.model is None:
            raise ValueError("No model loaded.")
        
        export_path = self.model.export(format=format, **kwargs)
        print(f"Model exported to: {export_path}")
        
        return export_path
    
    def get_best_weights(self) -> Optional[Path]:
        """Get path to best weights from last training run."""
        if self.results is None:
            return None
        
        save_dir = Path(self.results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        
        if best_weights.exists():
            return best_weights
        return None
    
    def get_last_weights(self) -> Optional[Path]:
        """Get path to last weights from last training run."""
        if self.results is None:
            return None
        
        save_dir = Path(self.results.save_dir)
        last_weights = save_dir / "weights" / "last.pt"
        
        if last_weights.exists():
            return last_weights
        return None


def quick_train(
    epochs: int = 100,
    batch_size: int = 16,
    model: str = "yolo11s-seg",
    device: Optional[str] = None,
) -> DentalCariesTrainer:
    """
    Quick training function with minimal configuration.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        model: Model variant (yolo11n-seg, yolo11s-seg, etc.)
        device: Device to use (cuda, mps, cpu).
    
    Returns:
        Trainer instance with results.
    """
    config = Config(
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    
    trainer = DentalCariesTrainer(config=config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Quick test
    print("Dental Caries Trainer initialized.")
    print(f"Model: {SETTINGS.model_name}")
    print(f"Data YAML: {SETTINGS.data_yaml}")
    print(f"Device: {SETTINGS.device}")
