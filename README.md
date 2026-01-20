# Dental Caries Detection using YOLOv11 Segmentation

A deep learning project for detecting dental caries (cavities) in dental X-ray images using YOLOv11 instance segmentation.

## Project Structure

```
ds_assignment/
├── data/                   # Dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── src/                    # Source code
│   ├── config/             # Configuration files
│   ├── data/               # Data utilities
│   ├── models/             # Model utilities
│   ├── evaluation/         # Evaluation utilities
│   └── utils/              # General utilities
├── scripts/                # Executable scripts
├── runs/                   # Training outputs
├── report.md               # Final report
└── requirements.txt        # Dependencies
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Explore Dataset
```bash
python -m src.data.explore
```

### Train Model
```bash
python scripts/train.py
```

### Evaluate Model
```bash
python scripts/evaluate.py --weights runs/segment/train/weights/best.pt
```

### Run Inference
```bash
python scripts/predict.py --weights runs/segment/train/weights/best.pt --source path/to/image.jpg
```

## Model

- **Architecture**: YOLOv11s-seg (Small variant)
- **Task**: Instance Segmentation
- **Target Metric**: mAP@50 > 70%

## Results

See [report.md](report.md) for detailed results and analysis.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.3+
- CUDA (recommended for training)
