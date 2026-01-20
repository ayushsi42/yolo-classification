#!/usr/bin/env python3
"""
Evaluation script for dental caries detection model.

Usage:
    python scripts/evaluate.py --weights runs/segment/train/weights/best.pt
    python scripts/evaluate.py --weights path/to/model.pt --save-metrics
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO

from src.config import SETTINGS
from src.evaluation import MetricsCalculator, ResultVisualizer
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 segmentation model for dental caries detection"
    )
    
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to data YAML (default: use config)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for evaluation"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--save-metrics", action="store_true",
        help="Save metrics to JSON file"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot training curves and metrics"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    logger = setup_logging(name="evaluate")
    
    # Check weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        sys.exit(1)
    
    # Get data YAML path
    data_yaml = args.data or str(SETTINGS.data_yaml)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = weights_path.parent.parent  # Go up from weights/best.pt
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Weights:    {weights_path}")
    print(f"Data YAML:  {data_yaml}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Conf Threshold: {args.conf}")
    print(f"IoU Threshold:  {args.iou}")
    print("=" * 60 + "\n")
    
    # Load model
    logger.info("Loading model...")
    model = YOLO(str(weights_path))
    
    # Run validation
    logger.info("Running validation...")
    results = model.val(
        data=data_yaml,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=True,
    )
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    calculator = MetricsCalculator()
    metrics = calculator.calculate_from_yolo_results(results)
    
    # Print summary
    calculator.print_summary()
    
    # Check target
    meets_target, message = calculator.meets_target(target_mAP50=0.70)
    print(f"\n{message}\n")
    
    # Save metrics
    if args.save_metrics:
        metrics_path = output_dir / "evaluation_metrics.json"
        calculator.save_metrics(metrics_path)
        
        # Also save a human-readable summary
        summary_path = output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(metrics.summary())
            f.write(f"\n\n{message}\n")
        logger.info(f"Summary saved to: {summary_path}")
    
    # Plot results
    if args.plot:
        visualizer = ResultVisualizer(save_dir=output_dir / "visualizations")
        
        # Try to plot training curves if results.csv exists
        results_dir = weights_path.parent.parent
        if (results_dir / "results.csv").exists():
            logger.info("Plotting training curves...")
            visualizer.plot_training_curves(results_dir, save=True, show=False)
        
        # Plot metrics comparison
        logger.info("Plotting metrics comparison...")
        visualizer.plot_metrics_comparison(
            metrics.to_dict(),
            target_mAP50=0.70,
            save=True,
            show=False,
        )
    
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
