# YOLOX Object Detection Training Pipeline

This document describes the YOLOX-based training pipeline for object detection, which supports any object class and replaces the previous YOLOv8 training approach.

## 📍 Working Directory

**Important**: All commands and scripts in this pipeline must be executed from the `train_yolox` directory. All file paths referenced in this documentation are relative to this directory.

```bash
# Always start from the train_yolox directory
cd /data/code/image-detector/train_yolox/
```

## 🎯 Overview

**YOLOX** is a high-performance anchor-free YOLO implementation that exceeds YOLOv3~v5 in performance. This pipeline uses **YOLOX-Small** model optimized for object detection with the following features:

**Note**: This pipeline uses a dedicated `yolox` virtual environment to avoid conflicts with existing dependencies.

- **Model**: YOLOX-Small (balanced speed/accuracy)
- **Input Size**: 640x640 pixels (fixed)
- **Training**: GPU-accelerated with CUDA support
- **Export**: ONNX format for mobile deployment
- **Dataset**: COCO format (converted from YOLO format)
- **Environment**: Isolated `yolox` virtual environment
- **Multi-Class Support**: Automatically detects class names from `classes.txt`
- **Flexible Dataset Paths**: Parameterized data directories

## 📁 Scripts Overview

The YOLOX pipeline introduces several new scripts that work alongside your existing project structure:

```
src/
├── yolox_setup.py          # Install YOLOX and dependencies
├── yolox_resize.py         # Resize images with letterboxing and adjust coordinates
├── yolox_data_prep.py      # Prepare dataset for YOLOX training
├── yolox_train.py          # Train YOLOX model for object detection
├── yolox_test.py           # Test trained YOLOX model
└── yolox_export.py         # Export trained model to ONNX format
```

## 🚀 Complete Training Workflow

**Important**: All commands in this workflow must be executed from the `train_yolox` directory. All file paths are relative to this directory.

### Step 1: Environment Setup

#### Create and Activate Virtual Environment

```bash
# Navigate to train_yolox directory first
cd train_yolox/

# Create new virtual environment for YOLOX
python3 -m venv yolox

# Activate the environment
source yolox/bin/activate

# Upgrade pip
pip install --upgrade pip

# Verify Python version (3.7+ required)
python --version
```

#### Install YOLOX and Dependencies

```bash
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

**Environment Requirements:**
- **Python**: 3.7+ (3.11 recommended)
- **CUDA**: Required for GPU training
- **Dependencies**: All requirements automatically installed via `requirements_yolox.txt`
- **PyTorch**: Automatically installed first during setup (required for YOLOX compilation)

### Step 2: Data Preparation

#### Prerequisites

Your dataset directory must contain:
- Images (`.jpg`, `.JPG`, `.png`, `.PNG`)
- YOLO format labels (`.txt` files)
- **`classes.txt` file** with class names (one per line)

The pipeline automatically detects the class name from the first line of `classes.txt` and creates the dataset directory accordingly.

#### Resize Images (if needed)

```bash
# Resize images to 640x640 with letterboxing
python src/yolox_resize.py
```

**Letterboxing Process:**
1. **Scale**: Images are scaled down to fit within 640x640 while preserving aspect ratio
2. **Pad**: Black padding is added to reach exactly 640x640 dimensions
3. **Coordinate Adjustment**: YOLO coordinates are automatically adjusted for the new scale and padding

**Example**: A 1280x960 image becomes:
- Scale factor: 0.667 (640/960)
- New size after scaling: 853x640  
- Padding: 106 pixels on left/right to center the image
- Final size: 640x640 with black bars on sides

#### Convert Dataset to COCO Format

```bash
# Convert YOLO labels to COCO format and create dataset splits
python src/yolox_data_prep.py --raw-dir data/raw_resized
```

**What the Data Prep Script Does:**
- **Class Detection**: Reads the first class name from `classes.txt` (e.g., "probe")
- **Dynamic Configuration**: Creates `exps/yolox_s_{class_name}.py` configuration file
- **Dataset Creation**: Creates `YOLOX/datasets/<class_name>_dataset/` directory
- **Data Organization**: Organizes images and labels into train/val/test splits (70/20/10)
- **Format Conversion**: Converts YOLO format labels to COCO format (required by YOLOX)
- **Coordinate Validation**: Verifies that coordinate conversion is accurate

#### Example Directory Structure

```
data/raw_resized/
├── classes.txt          # Contains class names (e.g., "probe", "circle", "square")
├── image001.jpg         # Your images
├── image001.txt         # YOLO format labels
├── image002.jpg
├── image002.txt
└── ...
```

#### Data Preparation Script Options

```bash
# Basic usage with default data directory (auto-detects class from classes.txt)
python src/yolox_data_prep.py

# Specify custom source directory containing images, labels, and classes.txt
python src/yolox_data_prep.py --raw-dir data/Probes

# Specify both source and target directories
python src/yolox_data_prep.py --raw-dir data/Probes --data-dir YOLOX/datasets/my_probes

# Verify prepared dataset
python src/yolox_data_prep.py --verify

# Validate coordinate conversion accuracy
python src/yolox_data_prep.py --validate-coords --raw-dir data/Probes

# Combine options
python src/yolox_data_prep.py --raw-dir data/custom_dataset --data-dir YOLOX/datasets/custom_objects --validate-coords
```

**Available Options:**
- `--raw-dir`: Path to directory containing raw images, YOLO labels, and classes.txt (default: `data/raw_resized`)
- `--data-dir`: Target dataset directory. If not specified, defaults to `YOLOX/datasets/<class_name>_dataset`
- `--verify`: Verify the prepared dataset structure and file counts
- `--validate-coords`: Validate that COCO coordinates correctly correspond to YOLO coordinates

### Step 3: Train YOLOX Model

```bash
# Train YOLOX-Small model 
python src/yolox_train.py --data-dir YOLOX/datasets/probe_dataset --epochs 50 --batch 8
```

**Training Features:**
- **GPU Training**: Automatically uses CUDA if available
- **Fixed Resolution**: 640x640 pixels (optimal for object detection)
- **Batch Size**: Configurable (default: 8, memory optimized)
- **Epochs**: Configurable (default: 80)
- **Dynamic Class Support**: Adapts to any object class from your dataset

**Available Training Options:**
- `--data-dir`: Path to dataset directory (auto-detects if not specified)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size per GPU (default: 2, memory optimized)
- `--workers`: Data loading workers (default: 1, memory optimized)
- `--setup`: Setup YOLOX repository and dependencies

### Step 4: Test Model

#### Test with PyTorch Checkpoint

```bash
source yolox/bin/activate
# Test with PyTorch checkpoint
python src/yolox_test.py \
  --model YOLOX/YOLOX_outputs/yolox_s_circle/best_ckpt.pth \
  --image YOLOX/datasets/probe_dataset/test/images/IMG_0199.JPG \
  --confidence 0.3 --show
```

#### Test with ONNX Model

```bash
# Test with ONNX model (faster inference)
python src/yolox_test_onnx.py \
  --model models/yolox_probe_detector.onnx \
  --image YOLOX/datasets/probe_dataset/test/images/IMG_0199.JPG \
  --show --confidence 0.5
```

Results are written to `models/yolox_test_results/` (summary at `test_summary.txt`; result images are saved when detections are found).

### Step 5: Export to ONNX

Export your trained checkpoint to an ONNX model that accepts only 1x3x640x640 inputs (static shape).

```bash
# Export trained model to ONNX format (run from train_yolox directory)
python src/yolox_export.py \
  --model YOLOX/YOLOX_outputs/yolox_s_circle/best_ckpt.pth \
  --output models/yolox_probe_detector.onnx \
  --exp exps/yolox_s_circle.py
```

**Path Information:**
- All paths are relative to the `train_yolox` directory where the script is executed
- `--model`: Path to trained checkpoint (relative to execution directory)
- `--output`: Output ONNX file path (relative to execution directory) 
- `--exp`: Experiment configuration file (relative to execution directory)

**Export Options:**
- `--opset`: ONNX opset version (default: 17)
- `--dynamic`: Export with dynamic batch size (default: fixed batch size)
- `--no-simplify`: Skip ONNX simplification step
- `--decode-in-inference`: Include decoding in the exported model

**Notes:**
- The exp sets `test_size=(640, 640)`, so the exported model input is fixed to 1x3x640x640
- If ONNX simplifier errors, add `--no-simplify` to skip simplification
- Output model name can reflect your class (e.g., `yolox_probe_detector.onnx`, `yolox_circle_detector.onnx`)

## 📊 Dataset Format

### Input Format (YOLO)

Your existing YOLO format labels are supported:

```
# data/Probes/IMG_0199.txt
0 0.117167 0.146875 0.153667 0.102750
0 0.912667 0.149750 0.134667 0.101000
```

**Format**: `class_id x_center y_center width height`
- All values are normalized (0.0 to 1.0)
- `class_id` should be 0 for your object class (e.g., circles, probes, squares)

**Classes File**: `classes.txt` defines your object classes:

```
# data/Probes/classes.txt
probe
```

### Output Format (COCO)

YOLOX requires COCO format annotations, automatically generated with your class name:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "IMG_0199.JPG",
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
      "name": "probe",
      "supercategory": "object"
    }
  ]
}
```

## 🎯 Multi-Class Support

The pipeline automatically adapts to any object class:

1. **Place your class name** in `classes.txt` (first line is used)
2. **Dataset directory** is automatically named `YOLOX/datasets/<class_name>_dataset`
3. **COCO annotations** use your class name instead of hardcoded values
4. **Model training** adapts to your specific object type

**Examples:**
- `classes.txt` contains "circle" → creates `YOLOX/datasets/circle_dataset/`
- `classes.txt` contains "probe" → creates `YOLOX/datasets/probe_dataset/`
- `classes.txt` contains "square" → creates `YOLOX/datasets/square_dataset/`

## ⚙️ Configuration

### Model Configuration

YOLOX configurations are **dynamically created** during data preparation based on your class name:

- `exps/yolox_s_base.py`: Base YOLOX-Small configuration (created automatically)
- `exps/yolox_s_{class_name}.py`: Class-specific configuration (auto-generated from `classes.txt`)

**Dynamic Configuration Features:**
- **Auto-adapts to your class**: Reads class name from `classes.txt` and creates `yolox_s_{class_name}.py`
- **Single class detection**: Automatically configured for your specific object type
- **640x640 input resolution**: Fixed optimal resolution for object detection
- **Auto dataset path detection**: Automatically configures dataset paths
- **Memory-optimized settings**: Training stability and performance optimization
- **ONNX export configuration**: Ready for mobile deployment

**Examples:**
- `classes.txt` contains "probe" → creates `exps/yolox_s_probe.py`
- `classes.txt` contains "circle" → creates `exps/yolox_s_circle.py`
- `classes.txt` contains "square" → creates `exps/yolox_s_square.py`