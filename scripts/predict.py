#!/usr/bin/env python3
"""
Inference script for dental caries detection.

Usage:
    python scripts/predict.py --weights runs/segment/train/weights/best.pt --source path/to/image.jpg
    python scripts/predict.py --weights path/to/model.pt --source path/to/images/ --save
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO

from src.config import SETTINGS
from src.evaluation import ResultVisualizer
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with YOLOv11 dental caries detection model"
    )
    
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to image, directory, or video"
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
        "--imgsz", type=int, default=640,
        help="Image size"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results"
    )
    parser.add_argument(
        "--save-txt", action="store_true",
        help="Save results as text files"
    )
    parser.add_argument(
        "--save-crop", action="store_true",
        help="Save cropped predictions"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display results"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create visualization grid of predictions"
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup
    logger = setup_logging(name="predict")
    
    # Check paths
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        sys.exit(1)
    
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("DENTAL CARIES DETECTION - INFERENCE")
    print("=" * 60)
    print(f"Weights:    {weights_path}")
    print(f"Source:     {source_path}")
    print(f"Conf Threshold: {args.conf}")
    print(f"IoU Threshold:  {args.iou}")
    print("=" * 60 + "\n")
    
    # Load model
    logger.info("Loading model...")
    model = YOLO(str(weights_path))
    
    # Run inference
    logger.info("Running inference...")
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        project=args.output_dir or str(SETTINGS.runs_dir / "predict"),
        show=args.show,
    )
    
    # Print results summary
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    
    total_detections = 0
    for i, result in enumerate(results):
        num_detections = len(result.boxes) if result.boxes is not None else 0
        total_detections += num_detections
        
        # Get image name
        img_name = Path(result.path).name if hasattr(result, 'path') else f"Image {i}"
        
        if num_detections > 0:
            print(f"  {img_name}: {num_detections} dental caries detected")
            
            # Print confidence scores
            if result.boxes is not None:
                confs = result.boxes.conf.cpu().numpy()
                for j, conf in enumerate(confs):
                    print(f"    - Detection {j+1}: {conf:.2%} confidence")
        else:
            print(f"  {img_name}: No dental caries detected")
    
    print("-" * 40)
    print(f"Total: {total_detections} detections in {len(results)} images")
    print("-" * 40 + "\n")
    
    # Create visualization grid
    if args.visualize and len(results) > 0:
        logger.info("Creating visualization...")
        output_dir = Path(args.output_dir) if args.output_dir else SETTINGS.runs_dir / "predict"
        visualizer = ResultVisualizer(save_dir=output_dir)
        visualizer.visualize_predictions(results, save=True, show=args.show)
    
    # Print save location
    if args.save:
        print(f"Results saved to: {args.output_dir or SETTINGS.runs_dir / 'predict'}")
    
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
