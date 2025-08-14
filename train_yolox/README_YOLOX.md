# YOLOX Circle Detection Training Pipeline

This document describes the YOLOX-based training pipeline for circle detection, which replaces the previous YOLOv8 training approach.

## 🎯 Overview

**YOLOX** is a high-performance anchor-free YOLO implementation that exceeds YOLOv3~v5 in performance. This pipeline uses **YOLOX-Small** model specifically optimized for circle detection with the following features:

**Note**: This pipeline uses a dedicated `yolox` virtual environment to avoid conflicts with existing dependencies.

- **Model**: YOLOX-Small (balanced speed/accuracy)
- **Input Size**: 640x640 pixels (fixed)
- **Training**: GPU-accelerated with CUDA support
- **Export**: ONNX format for mobile deployment
- **Dataset**: COCO format (converted from YOLO format)
- **Environment**: Isolated `yolox` virtual environment

## 📁 New Scripts

The YOLOX pipeline introduces several new scripts that work alongside your existing project structure:

```
src/
├── yolox_setup.py          # Install YOLOX and dependencies
├── yolox_data_prep.py      # Prepare dataset for YOLOX training
├── yolox_train.py          # Train YOLOX model for circle detection
└── yolox_test.py           # Test trained YOLOX model
```

## 🔧 Environment Management

### Virtual Environment Setup

The YOLOX pipeline uses a dedicated virtual environment to ensure clean dependency management:

```bash
# Create YOLOX environment
python3 -m venv yolox

# Activate environment (required for all operations)
source yolox/bin/activate

# Deactivate when done
deactivate
```

### Environment Requirements

- **Python**: 3.7+ (3.11 recommended)
- **CUDA**: Required for GPU training
- **Dependencies**: All requirements automatically installed via `requirements_yolox.txt`
- **PyTorch**: Automatically installed first during setup (required for YOLOX compilation)

## 🚀 Quick Start

### Step 0: Create YOLOX Virtual Environment

```bash
# From project root (/data/code/image-detector/)
# Create new virtual environment for YOLOX
python3 -m venv yolox

# Activate the environment
source yolox/bin/activate

# Upgrade pip
pip install --upgrade pip

# Verify Python version (3.7+ required)
python --version
```

### Step 1: Setup YOLOX Environment

```bash
# From project root (/data/code/image-detector/)
source yolox/bin/activate

# Install YOLOX and dependencies
python src/yolox_setup.py
```

This script will:
- Check system requirements (Python 3.7+, CUDA)
- Install PyTorch first (required for YOLOX compilation)
- Install YOLOX dependencies from `requirements_yolox.txt`
- Clone YOLOX repository
- Create necessary configuration files
- Verify the setup

### Step 2: Prepare Dataset

```bash
# Convert YOLO labels to COCO format and create dataset splits
python src/yolox_data_prep.py
```

This script will:
- Organize your `data/raw/` images and labels into train/val/test splits
- Convert YOLO format labels to COCO format (required by YOLOX)
- Create `config/yolox_dataset.yaml` configuration
- Verify dataset integrity

### Step 3: Train YOLOX Model

```bash
# Train YOLOX-Small model for circle detection
python src/yolox_train.py
```

Training features:
- **GPU Training**: Automatically uses CUDA if available
- **Fixed Resolution**: 640x640 pixels (optimal for circle detection)
- **Batch Size**: Configurable (default: 8)
- **Epochs**: Configurable (default: 100)
- **Early Stopping**: Built-in with configurable patience

### Step 4: Test Model

```bash
# Test the trained model on validation images
python src/yolox_test.py

# Or test on custom directory
python src/yolox_test.py --test-dir data/test/images

# Validate dataset structure
python src/yolox_test.py --validate
```

#### Example: Test a single image

```bash
source yolox/bin/activate
python src/yolox_test.py \
  --model /data/code/image-detector/train_yolox/YOLOX/YOLOX_outputs/yolox_s_circle/best_ckpt.pth \
  --image /data/code/image-detector/train_yolox/YOLOX/datasets/circle_dataset/test/images/20250723_102806.jpg \
  --confidence 0.3   --show
```

Results are written to `models/yolox_test_results/` (summary at `test_summary.txt`; result image is saved when detections are found).

### Step 5: Export to ONNX (fixed 640x640 input)

Export your trained checkpoint to an ONNX model that accepts only 1x3x640x640 inputs (static shape). Omit `--dynamic` to keep the shape fixed.

```bash
/data/code/image-detector/train_yolox/yolox/bin/python \
/data/code/image-detector/train_yolox/YOLOX/tools/export_onnx.py \
-f /data/code/image-detector/train_yolox/exps/yolox_s_circle.py \
-c /data/code/image-detector/train_yolox/YOLOX/YOLOX_outputs/yolox_s_circle/best_ckpt.pth \
--output-name /data/code/image-detector/train_yolox/models/yolox_circle_detector.onnx \
-o 17
```

Notes:
- The exp sets `test_size=(640, 640)`, so the exported model input is fixed to 1x3x640x640.
- If ONNX simplifier errors, add `--no-onnxsim` to skip simplification.

## ⚙️ Configuration

### Training Parameters

The YOLOX training script supports several configurable parameters:

```bash
python src/yolox_train.py --epochs 150 --batch 16 --workers 8
```

**Available Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size per GPU (default: 8)
- `--workers`: Data loading workers (default: 4)
- `--setup`: Setup YOLOX repository and dependencies

### Model Configuration

YOLOX configurations are automatically created in the `exps/` directory:

- `exps/yolox_s_base.py`: Base YOLOX-Small configuration
- `exps/yolox_s_circle.py`: Circle detection specific configuration

**Key Configuration Features:**
- Single class detection (circle only)
- 640x640 input resolution
- SGD optimizer with momentum
- Step learning rate scheduling
- ONNX export configuration

## 📊 Dataset Format

### Input Format (YOLO)

Your existing YOLO format labels are supported:

```
# data/raw/20250723_102136.txt
0 0.117167 0.146875 0.153667 0.102750
0 0.912667 0.149750 0.134667 0.101000
```

**Format**: `class_id x_center y_center width height`
- All values are normalized (0.0 to 1.0)
- `class_id` should be 0 for circles

### Output Format (COCO)

YOLOX requires COCO format annotations, automatically generated:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "20250723_102136.jpg",
      "width": 640,
      "height": 640
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [75, 94, 98, 66],
      "area": 6468,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "circle",
      "supercategory": "shape"
    }
  ]
}
```

