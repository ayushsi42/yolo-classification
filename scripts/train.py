#!/usr/bin/env python3
"""
Training script for dental caries detection using YOLOv11 segmentation.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 150 --batch 8 --model yolo11m-seg
    python scripts/train.py --resume
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config, SETTINGS
from src.models import DentalCariesTrainer
from src.data.augmentation import get_augmentation_config
from src.data.preprocessing import preprocess_dataset
from src.utils import setup_logging, print_system_info, set_seed, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 segmentation model for dental caries detection"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", type=str, default="yolo11s-seg",
        help="Model variant (yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg)"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to pretrained weights (for fine-tuning)"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Learning rate
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    
    # Augmentation
    parser.add_argument(
        "--augment", type=str, default="medical",
        choices=["default", "light", "heavy", "medical"],
        help="Augmentation preset"
    )
    # Preprocessing
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE preprocessing")
    parser.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit")
    
    # Device
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda, mps, cpu)"
    )
    
    # Experiment
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    logger = setup_logging(name="train")
    print_system_info()
    set_seed(args.seed)
    
    # Create configuration
    config = Config(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        lr0=args.lr,
        device=args.device,
        use_clahe=args.clahe,
    )
    
    # Handle CLAHE Preprocessing
    if config.use_clahe:
        logger.info("CLAHE preprocessing enabled. Enhancing dataset...")
        processed_data_dir = config.project_root / "data_clahe"
        if not processed_data_dir.exists():
            preprocess_dataset(config.data_dir, processed_data_dir, clip_limit=args.clip)
        
        # Update config to use preprocessed data
        config.data_dir = processed_data_dir
        config.train_dir = processed_data_dir / "train"
        config.val_dir = processed_data_dir / "val"
        
        # Create a new data_clahe.yaml
        clahe_yaml = config.project_root / "src" / "config" / "data_clahe.yaml"
        with open(clahe_yaml, 'w') as f:
            f.write(f"path: {processed_data_dir}\n")
            f.write(f"train: train/images\n")
            f.write(f"val: val/images\n")
            f.write(f"names:\n  0: dental_caries\n")
            f.write(f"nc: 1\n")
        
        config.data_yaml = clahe_yaml
        logger.info(f"Dataset enhanced and saved to {processed_data_dir}")
    
    # Get augmentation config
    augmentation = get_augmentation_config(args.augment)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model:          {config.model_name}")
    print(f"Epochs:         {config.epochs}")
    print(f"Batch Size:     {config.batch_size}")
    print(f"Image Size:     {config.imgsz}")
    print(f"Learning Rate:  {config.lr0}")
    print(f"Device:         {config.device}")
    print(f"Augmentation:   {args.augment}")
    print(f"Data YAML:      {config.data_yaml}")
    print("=" * 60 + "\n")
    
    # Initialize trainer
    trainer = DentalCariesTrainer(config=config, augmentation=augmentation)
    
    # Load model
    if args.weights:
        trainer.load_model(args.weights)
    else:
        trainer.load_model()
    
    # Start training
    try:
        results = trainer.train(
            resume=args.resume,
            experiment_name=args.name,
        )
        
        # Print results summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        # Get best weights path
        best_weights = trainer.get_best_weights()
        if best_weights:
            print(f"Best weights saved to: {best_weights}")
        
        # Validate on best weights
        print("\nRunning validation on best weights...")
        val_results = trainer.validate(weights=str(best_weights))
        
        # Print key metrics
        if hasattr(val_results, 'seg'):
            print(f"\nMask mAP@50:    {val_results.seg.map50:.4f}")
            print(f"Mask mAP@50-95: {val_results.seg.map:.4f}")
        if hasattr(val_results, 'box'):
            print(f"Box mAP@50:     {val_results.box.map50:.4f}")
            print(f"Box mAP@50-95:  {val_results.box.map:.4f}")
        
        # Check if target is met
        target_mAP = 0.70
        achieved_mAP = val_results.seg.map50 if hasattr(val_results, 'seg') else val_results.box.map50
        if achieved_mAP >= target_mAP:
            print(f"\n✅ TARGET ACHIEVED! mAP@50: {achieved_mAP:.4f} >= {target_mAP:.4f}")
        else:
            print(f"\n⚠️  Target not yet met. mAP@50: {achieved_mAP:.4f} < {target_mAP:.4f}")
            print("Consider: more epochs, larger model, or additional augmentation")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("You can resume training with: python scripts/train.py --resume")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
