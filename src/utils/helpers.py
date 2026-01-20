"""
Helper utilities for the dental caries detection project.
General-purpose functions for logging, device management, and reproducibility.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "dental_caries"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file.
        name: Logger name.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "recommended_device": "cpu",
    }
    
    if info["cuda_available"]:
        info["recommended_device"] = "cuda"
        info["cuda_devices"] = []
        for i in range(info["cuda_device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "major": props.major,
                "minor": props.minor,
            })
    elif info["mps_available"]:
        info["recommended_device"] = "mps"
    
    return info


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_system_info():
    """Print system and environment information."""
    import platform
    
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    print(f"\nPython Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    device_info = get_device_info()
    print(f"\nRecommended Device: {device_info['recommended_device']}")
    
    if device_info["cuda_available"]:
        print(f"CUDA Available: Yes")
        print(f"CUDA Device Count: {device_info['cuda_device_count']}")
        for i, dev in enumerate(device_info.get("cuda_devices", [])):
            print(f"  GPU {i}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
    else:
        print("CUDA Available: No")
    
    if device_info["mps_available"]:
        print("MPS Available: Yes (Apple Silicon)")
    
    print("=" * 50 + "\n")


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model or YOLO model.
    
    Returns:
        Dictionary with parameter counts.
    """
    if hasattr(model, 'model'):
        # YOLO model
        pytorch_model = model.model
    else:
        pytorch_model = model
    
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_millions": total_params / 1e6,
        "trainable_millions": trainable_params / 1e6,
    }


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path.
    
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    # Test utilities
    print_system_info()
    set_seed(42)
