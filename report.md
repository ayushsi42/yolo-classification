# Dental Caries Detection - Technical Report

## Project Overview

This project implements a deep learning solution for detecting dental caries (cavities) in dental X-ray images using YOLOv11 instance segmentation. The model is designed for clinical diagnostic assistance, prioritizing high recall to minimize missed detections.

---

## Model Architecture

### Why YOLOv11-seg?

| Factor | Reasoning |
|--------|-----------|
| **State-of-the-art** | YOLOv11 is the latest YOLO version with improved accuracy and efficiency |
| **Instance Segmentation** | Native support for polygon masks matches our YOLO segmentation format |
| **Transfer Learning** | COCO pretrained weights provide robust feature extraction for faster convergence |
| **Real-time Capable** | Fast inference suitable for potential clinical deployment |

### Model Variant Selection

We selected **YOLOv11s-seg** (Small variant) as it provides a good balance between:
- Model capacity sufficient for our dataset size (~280 training images)
- Training efficiency on consumer-grade hardware
- Inference speed for practical deployment

| Variant | Parameters | mAP@50 (COCO) | Speed (ms) |
|---------|------------|---------------|------------|
| yolo11n-seg | 2.9M | 38.8 | 1.8 |
| **yolo11s-seg** | 10.1M | 46.1 | 2.5 |
| yolo11m-seg | 22.4M | 51.3 | 5.0 |
| yolo11l-seg | 27.6M | 53.2 | 6.2 |

---

## Training Strategy

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 100 | Sufficient for convergence with early stopping |
| **Batch Size** | 16 | Balance between gradient stability and memory |
| **Image Size** | 640×640 | Standard YOLO size, good resolution for dental details |
| **Learning Rate** | 0.01 (initial) | YOLO default with cosine annealing |
| **Optimizer** | AdamW | Better generalization than SGD |
| **Early Stopping** | 20 epochs | Prevent overfitting |

### Data Augmentation

Augmentation strategy optimized for medical imaging:

| Augmentation | Value | Reasoning |
|--------------|-------|-----------|
| **Horizontal Flip** | 50% | Valid - teeth are bilaterally symmetric |
| **Vertical Flip** | 0% | Disabled - dental X-rays have consistent orientation |
| **Rotation** | ±10° | Slight rotation variations in clinical imaging |
| **Scale** | 0.5-1.5× | Account for different image crops |
| **HSV Adjustments** | Moderate | Handle exposure variations in X-rays |
| **Mosaic** | 80% | Increase effective batch diversity |

---

## Dataset Analysis

### Dataset Statistics

| Split | Images | Total Instances | Avg Instances/Image |
|-------|--------|-----------------|---------------------|
| Train | ~280 | ~450 | ~1.6 |
| Val | ~35 | ~55 | ~1.6 |

### Class Distribution
- Single class: `dental_caries`
- Format: YOLO Segmentation (polygon masks)

---

## Evaluation Metrics

### Primary Metrics

| Metric | Target | Importance |
|--------|--------|------------|
| **mAP@50** | >70% | Primary evaluation metric |
| **Recall** | High | Critical - minimize missed cavities (false negatives) |
| **Precision** | High | Minimize false alarms |

### Additional Metrics Rationale

1. **F1-Score**: Harmonic mean of Precision and Recall, provides a balanced single metric
2. **mAP@50-95**: Stricter IoU thresholds evaluate localization quality
3. **Sensitivity (= Recall)**: Standard clinical diagnostic metric (True Positive Rate)
4. **Specificity**: True Negative Rate - important for avoiding unnecessary treatments

> **Clinical Importance**: In diagnostic applications, **high recall is critical**. A missed cavity (false negative) can lead to disease progression, while a false positive can be verified by a clinician.

---

## Results

*[To be filled after training]*

### Training Performance

| Metric | Value |
|--------|-------|
| mAP@50 | -- |
| mAP@50-95 | -- |
| Precision | -- |
| Recall | -- |
| F1-Score | -- |

### Training Curves

*Training curves will be generated after running training.*

---

## Instructions

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py --epochs 100 --batch 16 --model yolo11s-seg
```

### Evaluation
```bash
python scripts/evaluate.py --weights runs/segment/train/weights/best.pt --plot
```

### Inference
```bash
python scripts/predict.py --weights runs/segment/train/weights/best.pt --source path/to/image.jpg
```

---

## Project Structure

```
ds_assignment/
├── data/                   # Dataset
│   ├── train/
│   └── val/
├── src/                    # Source code
│   ├── config/             # Configuration
│   ├── data/               # Data utilities
│   ├── models/             # Model trainer
│   ├── evaluation/         # Metrics & visualization
│   └── utils/              # Helpers
├── scripts/                # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── runs/                   # Training outputs
├── report.md               # This report
└── requirements.txt        # Dependencies
```

---

## Conclusion

This project demonstrates a complete pipeline for dental caries detection using state-of-the-art deep learning techniques. The modular codebase allows for easy experimentation with different models, hyperparameters, and augmentation strategies.

---

## References

1. Ultralytics YOLOv11 Documentation
2. YOLO Segmentation Format Specification
3. Dental Caries Detection Literature
