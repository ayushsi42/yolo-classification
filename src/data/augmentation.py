"""
Data augmentation configurations for dental caries detection.
Optimized augmentation strategies for dental X-ray images.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AugmentationConfig:
    """
    Augmentation configuration optimized for dental X-ray images.
    
    Dental X-rays have specific characteristics:
    - Consistent orientation (no vertical flipping needed)
    - Grayscale or limited color range
    - Fine details are important for cavity detection
    """
    
    # HSV augmentation (color space adjustments)
    hsv_h: float = 0.015  # Hue variation
    hsv_s: float = 0.7    # Saturation variation
    hsv_v: float = 0.4    # Value/brightness variation
    
    # Geometric transformations
    degrees: float = 10.0    # Rotation range (±degrees)
    translate: float = 0.1   # Translation as fraction of image size
    scale: float = 0.5       # Scale variation (1 ± scale)
    shear: float = 0.0       # Shear angle (degrees)
    perspective: float = 0.0  # Perspective distortion
    
    # Flip augmentations
    flipud: float = 0.0   # Vertical flip probability (disabled for dental)
    fliplr: float = 0.5   # Horizontal flip probability
    
    # Mosaic and mixup
    mosaic: float = 1.0     # Mosaic augmentation probability
    mixup: float = 0.0      # Mixup augmentation probability
    copy_paste: float = 0.0  # Copy-paste augmentation probability
    
    # Additional augmentations
    erasing: float = 0.0    # Random erasing probability
    crop_fraction: float = 1.0  # Crop fraction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
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
            "erasing": self.erasing,
            "crop_fraction": self.crop_fraction,
        }
    
    @classmethod
    def light(cls) -> "AugmentationConfig":
        """Light augmentation for quick training/testing."""
        return cls(
            degrees=5.0,
            translate=0.05,
            scale=0.2,
            mosaic=0.5,
        )
    
    @classmethod
    def heavy(cls) -> "AugmentationConfig":
        """Heavy augmentation for maximum data diversity."""
        return cls(
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=15.0,
            translate=0.15,
            scale=0.7,
            shear=2.0,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )
    
    @classmethod
    def medical_imaging(cls) -> "AugmentationConfig":
        """
        Augmentation optimized for medical imaging.
        Conservative transforms to preserve diagnostic features.
        """
        return cls(
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=10.0,
            translate=0.1,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,  # No vertical flip for consistent orientation
            fliplr=0.5,  # Horizontal flip is valid (left/right teeth are symmetric)
            mosaic=0.8,
            mixup=0.0,   # No mixup to preserve clear boundaries
            copy_paste=0.0,
        )


# Preset configurations
AUGMENTATION_PRESETS = {
    "default": AugmentationConfig(),
    "light": AugmentationConfig.light(),
    "heavy": AugmentationConfig.heavy(),
    "medical": AugmentationConfig.medical_imaging(),
}


def get_augmentation_config(preset: str = "medical") -> AugmentationConfig:
    """
    Get augmentation configuration by preset name.
    
    Args:
        preset: Preset name ("default", "light", "heavy", "medical")
    
    Returns:
        AugmentationConfig instance
    """
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(AUGMENTATION_PRESETS.keys())}")
    return AUGMENTATION_PRESETS[preset]


if __name__ == "__main__":
    # Print augmentation configurations
    print("Available Augmentation Presets:")
    print("=" * 50)
    
    for name, config in AUGMENTATION_PRESETS.items():
        print(f"\n{name.upper()}:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
