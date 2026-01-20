"""
Preprocessing utilities for dental X-ray images.
Includes CLAHE (Contrast Limited Adaptive Histogram Equalization) and other medical imaging enhancements.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to an image to enhance contrast.
    
    Args:
        image: Input image (BGR or Grayscale).
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of grid for histogram equalization.
        
    Returns:
        Enhanced image.
    """
    # Convert to grayscale if it's BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image
        is_color = False
        
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR if input was color
    if is_color:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
    return enhanced


def preprocess_dataset(data_dir: Path, output_dir: Path, clip_limit: float = 2.0):
    """
    Preprocess an entire dataset by applying CLAHE to all images.
    
    Args:
        data_dir: Root directory of the dataset (containing train/val).
        output_dir: Directory to save processed images.
        clip_limit: CLAHE clip limit.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    image_exts = ['*.jpg', '*.jpeg', '*.png']
    
    for split in ['train', 'val']:
        split_dir = data_dir / split / 'images'
        out_split_dir = output_dir / split / 'images'
        out_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Also copy labels
        label_dir = data_dir / split / 'labels'
        out_label_dir = output_dir / split / 'labels'
        out_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy labels
        print(f"Copying labels for {split}...")
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                content = f.read()
            with open(out_label_dir / label_file.name, 'w') as f:
                f.write(content)
        
        # Process images
        images = []
        for ext in image_exts:
            images.extend(list(split_dir.glob(ext)))
            
        print(f"Applying CLAHE to {len(images)} images in {split}...")
        for img_path in tqdm(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            enhanced = apply_clahe(img, clip_limit=clip_limit)
            cv2.imwrite(str(out_split_dir / img_path.name), enhanced)


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess dental X-rays with CLAHE")
    parser.add_argument("--input", type=str, required=True, help="Input data directory")
    parser.add_argument("--output", type=str, required=True, help="Output data directory")
    parser.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit")
    
    args = parser.parse_args()
    preprocess_dataset(Path(args.input), Path(args.output), args.clip)
