# Circle Detection Android App - Complete Guide

A step-by-step tutorial to create an Android application that detects circles in images using YOLOv8 object detection.

## 🎯 Project Overview

**Goal**: Build an Android app that detects circles in images using computer vision
- **Target Device**: Android devices with camera
- **Platform**: Fedora Linux development environment
- **Detection**: Circle detection with confidence scores
- **Approach**: YOLOv8 custom object detection model

## 📁 Project Structure

This project is organized to keep all development files in version control while excluding large datasets, models, and build artifacts:

```
image-detector/                    # Current project directory
├── .gitignore                     # Excludes data, models, builds
├── README.md                      # This guide
├── requirements.txt               # Python dependencies
├── config/
│   └── dataset.yaml              # YOLO dataset configuration
├── src/                          # Python scripts and utilities
│   ├── data_preparation.py       # Dataset organization
│   ├── train_model.py            # Model training
│   ├── test_model.py             # Model testing and validation
│   ├── export_model.py           # Model export for mobile
│   ├── circle_detector.py        # Circle detection utilities
│   └── cleanup_dataset.sh        # Dataset cleanup script
├── circle-detection-app/         # Android app source code
│   ├── app/                      # Android Studio project
│   └── assets/                   # Model assets (excluded from git)
├── data/                         # Dataset (excluded from git)
│   ├── raw/                      # Original images and annotations
│   ├── train/, val/, test/       # Split datasets (images + labels)
│   └── verification/             # Test images for validation
└── models/                       # Trained models (excluded from git)
```

## 🔧 Available Scripts

**📸 Data Preparation:**
- `src/data_preparation.py` - Organize dataset into train/val/test splits
- `src/cleanup_dataset.sh` - Clean processed data while preserving raw images

**🤖 Training & Testing:**
- `src/train_model.py` - Train YOLOv8 model for circle detection
- `src/test_model.py` - Comprehensive model testing and validation
- `src/export_model.py` - Export trained model for mobile deployment

**💡 Key Features:**
- **All scripts enforce project root execution** with clear error messages
- **Virtual environment validation** prevents dependency issues
- **GPU auto-detection** for faster training when available
- **Mobile export optimization** with TensorFlow Lite and ONNX formats

---

## 📋 Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Collection & Annotation](#2-data-collection--annotation)
3. [Model Training](#3-model-training)
4. [Model Testing & Validation](#4-model-testing--validation)
5. [Model Export for Mobile](#5-model-export-for-mobile)
6. [Android App Overview](#6-android-app-overview)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Environment Setup

### 1.1 System Requirements
```bash
# Update system
sudo dnf update -y

# Install essential development tools
sudo dnf groupinstall "Development Tools" -y
sudo dnf install git wget curl unzip tree -y
```

### 1.2 Python Environment Setup
```bash
# Install Python 3.11 and pip (best compatibility for ML/CV packages)
sudo dnf install python3.11 python3.11-pip python3.11-venv -y

# Create virtual environment with Python 3.11 in current directory
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Create Project Structure
```bash
# Create all necessary directories (data excluded from git)
mkdir -p {src,config,data/{raw,train/{images,labels},val/{images,labels},test/{images,labels}},models}

# Verify structure
tree -I 'venv|__pycache__|*.pyc'
```

---

## 2. Data Collection & Annotation

### 2.1 Image Collection Strategy

**You need approximately 200-500 images for good results:**

1. **Diverse Conditions**:
   - Different lighting (bright, dim, natural, artificial)
   - Various backgrounds
   - Different circle positions in images
   - Multiple angles and distances
   - Different circle sizes

2. **Image Requirements**:
   - Resolution: At least 640x640 pixels
   - Format: JPG/PNG
   - Clear, non-blurry images

### 2.2 Manual Image Collection

**📸 Image Collection Options:**

#### Option A: Manual Collection
- Use your phone camera to take photos
- Transfer images to `data/raw/` directory
- Ensure consistent naming (e.g., `circle_image_001.jpg`)

#### Option B: Import Existing Images
- Copy existing circle photos to `data/raw/` directory
- Use consistent naming convention
- Ensure images are in JPG/PNG format

**📐 Image Requirements & Diversity:**

#### Essential Diversity (CRITICAL for good detection):

1. **Different Angles and Distances:**
   - **Straight-on (0°)**: Circle perpendicular to camera
   - **Slight angles (±15°, ±30°)**: Camera tilted left/right
   - **Vertical angles (±15°, ±30°)**: Camera above/below circle
   - **Distance variations**: Close-up to far views

2. **Lighting Conditions:**
   - **Indoor lighting**: Fluorescent, LED, warm lighting
   - **Outdoor lighting**: Sunny, cloudy, golden hour
   - **Mixed lighting**: Window light + indoor lights
   - **Shadows**: Circles casting shadows

3. **Background Variations:**
   - **Clean backgrounds**: White wall, plain surface
   - **Textured backgrounds**: Wood, concrete, carpet
   - **Busy backgrounds**: Cluttered room, outdoor scene
   - **Different colors**: Contrasting and similar to circle color

4. **Circle Variations:**
   - **Different sizes**: Small circles to large circles
   - **Multiple orientations**: Various circle positions
   - **Partial visibility**: Circles partially obscured
   - **Different materials**: Metal, plastic, paper circles

#### Image Quality Standards:

```bash
# Recommended specifications:
Resolution: Minimum 640x640 pixels (1080p+ preferred)
Format: JPG or PNG
Quality: Sharp, non-blurry images
Circle size: Circle should be clearly visible (at least 50x50 pixels)
```

#### Collection Strategy:

**🎯 Systematic Approach (Recommended 200-500 images):**

```bash
# Session 1: Indoor controlled conditions (50-100 images)
# - Good lighting, clean background
# - Various angles and distances
# - Different circle positions

# Session 2: Challenging conditions (50-100 images)  
# - Poor lighting, shadows
# - Busy backgrounds
# - Extreme angles

# Session 3: Real-world conditions (100-300 images)
# - Mix of all conditions
# - Simulate actual usage scenarios
```

**⚠️ Common Mistakes to Avoid:**
- ❌ All images from same angle/distance
- ❌ Only perfect lighting conditions  
- ❌ Same background in all images
- ❌ Circle always in center of image
- ❌ Blurry or low-resolution images
- ❌ Images too small (circle barely visible)

**🔍 Quality Check:**
After collection, verify your dataset:
```bash
# Check image count and quality
ls data/raw/*.jpg | wc -l  # Should be 200+ images
file data/raw/*.jpg | head -5  # Check file types and sizes
```

### 2.3 Data Annotation

**📋 Overview:** Annotate your images using LabelImg tool for YOLO format.

**🎯 Annotation Setup:**

1. **Install LabelImg**:
   ```bash
   # Install annotation tool
   pip install labelImg
   ```

2. **Start LabelImg**:
   ```bash
   # Activate environment
   source venv/bin/activate
   
   # Start LabelImg
   labelImg data/raw data/raw
   ```

3. **Annotation Workflow**:
   - Draw bounding boxes around circles
   - Save annotations in YOLO format
   - Create class file: `echo "circle" > data/raw/classes.txt`

**📝 Annotation Guidelines:**
- **Tight bounding boxes**: Include entire circle
- **Consistent labeling**: Use same class name for all circles
- **Quality control**: Verify annotations are accurate
- **Complete coverage**: Annotate all visible circles in each image

**✅ When You're Done:**
- **Annotation files**: Each `.jpg` has corresponding `.txt` with YOLO format
- **Class file**: `data/raw/classes.txt` with circle class
- **Ready for training**: `python src/data_preparation.py`

### 2.4 Data Organization Script

**📄 Script:** `src/data_preparation.py`

**🚀 Usage:**
```bash
# After completing annotation, organize dataset
python src/data_preparation.py
```

**📊 What it does:**
- **Finds annotated images**: Scans `data/raw/` for image-annotation pairs
- **Creates data splits**: 70% train, 20% validation, 10% test
- **Organizes files**: Copies to `data/train/`, `data/val/`, `data/test/`
- **Updates config**: Sets absolute paths in `config/dataset.yaml`
- **Validates dataset**: Warns if insufficient data (< 10 images)

**📁 Output Structure:**
```
data/
├── train/images/ & train/labels/    # 70% of data
├── val/images/ & val/labels/        # 20% of data
└── test/images/ & test/labels/      # 10% of data
```

### 2.5 Dataset Cleanup Script

**📄 Script:** `src/cleanup_dataset.sh`

**🧹 When to use:**
- Want to re-organize dataset with different splits
- Need to start fresh with data preparation
- Testing different training parameters
- Cleaning up before final model training

**🚀 Usage:**
```bash
# Basic cleanup (preserves raw data + models)
./src/cleanup_dataset.sh

# Also remove trained models
./src/cleanup_dataset.sh --models

# Also remove Android assets
./src/cleanup_dataset.sh --android

# Clean everything except raw data
./src/cleanup_dataset.sh --all
```

**🛡️ Safety Features:**
- **Preserves `data/raw/`**: Your original images and annotations stay safe
- **Shows preview**: Lists exactly what will be deleted before action
- **Confirmation prompt**: Requires manual confirmation
- **Disk space info**: Shows how much space will be freed
- **Smart detection**: Only cleans existing directories

---

## 3. Model Training

### 3.1 Training Script

**📄 Script:** `src/train_model.py`

**🎯 Training Strategy: Circle Detection**
- **Single-class detection**: Model learns to detect circles in images
- **Aspect ratio preserved by default**: Better for real-world image proportions
- **Mobile-optimized**: YOLOv8n model for fast inference on mobile devices

**🚀 Basic Usage:**
```bash
# From project root with venv activated
python src/train_model.py
```

**⚙️ Advanced Usage (Configurable parameters):**
```bash
# High resolution for small circles (recommended)
python src/train_model.py --imgsz 1024 --batch 8 --epochs 100

# Quick training for testing
python src/train_model.py --imgsz 640 --batch 16 --epochs 50

# Force full training without early stopping
python src/train_model.py --epochs 100 --patience 0

# Enable early stopping (stop if no improvement for 30 epochs)
python src/train_model.py --epochs 200 --patience 30

# Use square input instead of preserving aspect ratio
python src/train_model.py --square
```

**🎯 Training Features:**
- **Auto GPU detection**: Uses CUDA if available, falls back to CPU
- **Configurable resolution**: 640px (fast) to 1280px (best accuracy)
- **Smart batch sizing**: Automatically adjusts based on image size and device
- **Mobile export**: Automatically exports to ONNX and TensorFlow Lite
- **Progress monitoring**: Real-time loss tracking and validation
- **Configurable early stopping**: Control or disable early stopping (patience parameter)
- **Aspect ratio preservation**: Default rectangular input (4:3 ratio) for better real-world performance
- **Single-class detection**: Model trained to detect circles in images

**📊 Output:**
- **Models saved to**: `models/circle_detector_*px_rect/weights/`
- **Best model**: `best.pt` (highest validation accuracy)
- **Mobile formats**: `.onnx`, `.tflite` for deployment
- **Training logs**: Complete metrics and progress tracking

### 3.2 Start Training

#### Basic Training (640px resolution):
```bash
# From project root (/data/code/image-detector/)
# Activate environment
source venv/bin/activate

# Ensure data is organized
python src/data_preparation.py

# Start training with default settings (preserves aspect ratio)
python src/train_model.py
```

#### High-Resolution Training (Recommended for small circles):
```bash
# Train with higher resolution for better small object detection
# GPU will be auto-detected and used if available
python src/train_model.py --imgsz 1024 --batch 8 --epochs 100

# Available image sizes and their use cases:
# --imgsz 640   # Default, fast training, good for medium-sized objects
# --imgsz 832   # Better accuracy, 1.7x slower training
# --imgsz 1024  # High accuracy for small objects, 2.5x slower (RECOMMENDED)
# --imgsz 1280  # Maximum accuracy, 4x slower, requires more memory

# Use square input instead of rectangular (not recommended)
python src/train_model.py --square
```

#### GPU vs CPU Training Performance:
| Device | Resolution | Training Speed | Recommended Batch Size |
|--------|------------|----------------|------------------------|
| **GPU (NVIDIA)** | 640px | **10-15x faster** | 16 |
| **GPU (NVIDIA)** | 1024px | **8-12x faster** | 6-8 |
| CPU | 640px | Baseline (slow) | 8-16 |
| CPU | 1024px | Very slow | 4-8 |

**✅ GPU Training Benefits:**
- **10-15x faster training** than CPU
- **Real-time progress monitoring** 
- **Larger batch sizes** possible
- **Better gradient updates**

#### Training Parameters Explained:
- **`--imgsz`**: Training image resolution (higher = better for small circles)
- **`--batch`**: Batch size (automatically reduced for large images)
- **`--epochs`**: Number of training iterations (more = better learning)

#### Performance vs Accuracy Trade-offs:
| Image Size | Training Time | Small Object Detection | Recommended For |
|------------|---------------|------------------------|-----------------|
| 640px | Fast (1x) | Good | Large circles, quick prototyping |
| 832px | Medium (1.7x) | Better | Medium circles, balanced approach |
| **1024px** | **Slow (2.5x)** | **Excellent** | **Small circles (YOUR USE CASE)** |
| 1280px | Very Slow (4x) | Maximum | Tiny objects, research quality |

**Training Tips**:
- Training will take 1-3 hours depending on your CPU
- Monitor the training loss in the terminal
- Models will be saved in `models/circle_detector_*px_rect/`
- Best model is automatically saved as `best.pt`

---

## 4. Model Testing & Validation

### 4.1 Verification Overview

After training your model, you need to verify its performance using multiple methods:

1. **📊 Automated Metrics**: Precision, Recall, mAP scores
2. **👁️ Visual Validation**: Test on sample images
3. **🎥 Real-time Testing**: Live camera feed
4. **📈 Performance Analysis**: Confusion matrix, detection confidence
5. **🎯 Edge Case Testing**: Challenging scenarios

### 4.2 Test Image Preparation

#### Where to put test images:

```bash
# Option 1: Use validation set (automatically created)
data/val/images/          # 15 images from your dataset

# Option 2: Create dedicated test set with new images
data/test_new/
├── challenging_angle_1.jpg
├── poor_lighting_1.jpg
├── busy_background_1.jpg
└── ...

# Option 3: Use completely new images
data/verification/
├── real_world_test_1.jpg
├── different_circles_1.jpg
└── ...
```

#### Create test image directories:

```bash
# Create verification directories
mkdir -p data/{test_new,verification,edge_cases}

# Copy some validation images for quick testing
cp data/val/images/*.jpg data/verification/ 2>/dev/null || echo "No validation images to copy"
```

### 4.3 Comprehensive Test Script

**📄 Script:** `src/test_model.py`

**🧪 Testing Options:**

**Basic Testing (Recommended first):**
```bash
# Quick visual test on validation images
python src/test_model.py

# Full validation metrics
python src/test_model.py --validation
```

**Advanced Testing:**
```bash
# Real-time camera testing
python src/test_model.py --realtime

# Test specific image directory  
python src/test_model.py --visual data/verification

# Complete test report
python src/test_model.py --report
```

### 4.4 Model Testing Features

**🎯 Testing Capabilities:**
- **📈 Validation metrics**: mAP, precision, recall, F1 score
- **👁️ Visual testing**: Shows detections on sample images  
- **🎥 Real-time testing**: Live camera feed with detection overlay
- **🎯 Edge case testing**: Tests challenging scenarios
- **📋 Complete report**: Runs all tests and generates summary

**🎮 Interactive Controls:**
- **Visual mode**: Press any key to continue, 'q' to quit
- **Real-time mode**: 'q' to quit, 's' to save screenshot
- **Automatic resizing**: Large images scaled for display

**📊 Performance Interpretation:**
- **🟢 Excellent**: mAP50 ≥ 0.8, precision ≥ 0.8, recall ≥ 0.8
- **🟡 Good**: mAP50 ≥ 0.6, moderate false positives
- **🔴 Needs work**: mAP50 < 0.5, high false positives or missed detections

### 4.5 How to Run Model Verification

#### Step 1: Activate Environment
```bash
# From project root (/data/code/image-detector/)
source venv/bin/activate
```

#### Step 2: Choose Your Test Type

**🧪 Quick Visual Test (Recommended first)**:
```bash
python src/test_model.py
# Tests first 10 validation images, shows detection results
```

**📊 Full Validation Metrics**:
```bash
python src/test_model.py --validation
# Gives mAP, precision, recall scores
```

**🎥 Real-time Camera Test**:
```bash
python src/test_model.py --realtime
# Test with live camera feed
```

**🎯 Test Specific Images**:
```bash
python src/test_model.py --visual data/verification
# Test images in specific directory
```

**📋 Complete Report**:
```bash
python src/test_model.py --report
# Runs all tests and generates comprehensive report
```

### 4.6 Expected Results & Interpretation

#### 🟢 Good Model Performance:
- **mAP50 > 0.7**: Excellent circle detection
- **Precision > 0.8**: Few false positives  
- **Recall > 0.7**: Finds most circles
- **Confidence > 0.5**: Reliable detections

#### 🟡 Acceptable Performance:
- **mAP50: 0.5-0.7**: Good for most use cases
- **Precision > 0.6**: Some false positives
- **Recall > 0.5**: Misses some circles

#### 🔴 Needs Improvement:
- **mAP50 < 0.5**: Poor detection accuracy
- **Low Precision**: Many false positives
- **Low Recall**: Misses many circles

### 4.7 Troubleshooting Poor Performance

**If model performs poorly:**

1. **Check training data quality**:
   ```bash
   # Verify annotations
   ls data/raw/*.txt | wc -l  # Should match image count
   head -5 data/raw/*.txt     # Check format
   ```

2. **Add more training data**:
   - Collect 100+ more images
   - Focus on challenging scenarios
   - Ensure diverse conditions

3. **Retrain with different parameters**:
   ```bash
   # Increase epochs or change model size
   # Edit src/train_model.py: epochs=200, or use yolov8s.pt
   ```

4. **Check for overfitting**:
   - Validation loss higher than training loss
   - Good validation metrics but poor real-world performance

### 4.8 Performance Benchmarks

**Target Performance for Circle Detection:**
- **Production Ready**: mAP50 > 0.8, Precision > 0.85
- **Prototype Ready**: mAP50 > 0.6, Precision > 0.7
- **Needs Work**: mAP50 < 0.5

**Real-world Testing Checklist:**
- [ ] Detects circles at different angles (±30°)
- [ ] Works in various lighting conditions
- [ ] Handles different backgrounds
- [ ] Minimal false positives
- [ ] Consistent confidence scores (>0.5)
- [ ] Real-time performance acceptable (>10 FPS on device)

---

## 5. Model Export for Mobile

### 5.1 Export Script

**📄 Script:** `src/export_model.py`

**💡 Export your trained model for mobile deployment!**

### Export Options:
- **TensorFlow Lite**: Optimized for Android deployment
- **ONNX**: Cross-platform format for various devices
- **Quantized models**: Smaller size and faster inference

### Export Process:
```bash
# Export model for mobile deployment
python src/export_model.py

# Or specify a specific model
python src/export_model.py models/circle_detector_640px_rect/weights/best.pt
```

### Export Features:
- **Multiple formats**: Exports Float16 TFLite, Int8 TFLite, and ONNX
- **Automatic copying**: Copies models to Android app assets
- **Size optimization**: Int8 quantization for fastest mobile inference
- **Cross-platform**: ONNX format for various deployment options

### Android Integration:
- Models automatically copied to `circle-detection-app/app/src/main/assets/`
- Int8 TFLite model used for fastest inference
- Float16 TFLite available as backup for better accuracy
- ONNX model available for cross-platform deployment

---

## 6. Android App Overview

### 6.1 App Features

The Android app provides a complete circle detection solution with the following capabilities:

**🎯 Core Functionality:**
- **Circle Detection**: Detects multiple circles in images
- **Real-time Processing**: Fast inference using TensorFlow Lite
- **Confidence Display**: Shows confidence scores for each detected circle
- **Visual Overlay**: Draws circles around detected objects with color-coded confidence levels
- **Multiple Circles**: Can detect and display multiple circles simultaneously

**📱 User Interface:**
- **Camera Mode**: Real-time circle detection using device camera
- **Image Mode**: Process images from gallery
- **Mode Switching**: Easy toggle between camera and image modes
- **Results Display**: Clear visualization of detected circles with confidence scores

**🎨 Visual Features:**
- **Color-coded confidence**: Green (high), Yellow (medium), Red (low confidence)
- **Real-time overlay**: Live detection results on camera feed
- **Image processing**: Support for gallery images with proper orientation handling

### 6.2 Technical Implementation

**🤖 Model Integration:**
- **Input Size**: 640x640 pixels (square)
- **Model Format**: TensorFlow Lite (optimized for mobile)
- **Detection Type**: Single class (circle)
- **Confidence Threshold**: 0.3 (30%)
- **Performance**: ~100-500ms inference time depending on device

**📱 Android Components:**
- **Package**: `com.example.circledetection`
- **Main Activity**: `MainActivity.java` - Handles camera, image processing, and UI
- **Overlay View**: `DetectionOverlayView.java` - Draws circles and confidence scores
- **Model Files**: `circle_model.tflite` (Int8), `circle_model_float16.tflite`, `circle_model.onnx`

**🔧 Technical Features:**
- **Camera2 API**: Modern Android camera implementation
- **TensorFlow Lite**: Optimized inference engine
- **GPU Acceleration**: Optional GPU delegate for faster processing
- **Memory Optimization**: Efficient image processing and model loading
- **Error Handling**: Robust error handling for various scenarios

### 6.3 App Structure

**📁 Project Organization:**
```
circle-detection-app/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/circledetection/
│   │   │   ├── MainActivity.java          # Main app logic
│   │   │   └── DetectionOverlayView.java  # Circle drawing overlay
│   │   ├── res/
│   │   │   ├── layout/activity_main.xml   # UI layout
│   │   │   ├── values/strings.xml         # App strings
│   │   │   └── values/colors.xml          # Color definitions
│   │   └── assets/
│   │       ├── circle_model.tflite        # Primary model (Int8)
│   │       ├── circle_model_float16.tflite # Backup model
│   │       └── circle_model.onnx          # Cross-platform model
│   └── build.gradle                       # Build configuration
└── README.md                             # App documentation
```

**🎮 User Experience:**
- **Intuitive Interface**: Simple, clean design focused on functionality
- **Mode Switching**: Easy toggle between camera and image modes
- **Real-time Feedback**: Immediate visual feedback for detections
- **Error Handling**: Clear error messages and graceful degradation

### 6.4 Performance Characteristics

**⚡ Speed Optimization:**
- **Int8 Quantization**: Fastest inference with acceptable accuracy
- **GPU Acceleration**: Optional GPU delegate for compatible devices
- **Memory Management**: Efficient image processing and model loading
- **Batch Processing**: Optimized for single-image processing

**📊 Performance Metrics:**
- **Inference Time**: 100-500ms depending on device capabilities
- **Model Size**: ~3MB (Int8 quantized)
- **Memory Usage**: Optimized for mobile devices
- **Battery Impact**: Minimal impact with efficient processing

**🔍 Detection Quality:**
- **Accuracy**: High precision circle detection
- **Confidence Scoring**: Reliable confidence thresholds
- **Multi-circle Support**: Detects multiple circles simultaneously
- **Edge Case Handling**: Robust detection in various conditions

---

## 7. Troubleshooting

### Common Issues:

**Model Training Issues:**
```bash
# If training fails due to memory
# Reduce batch size in train_model.py
batch=8  # instead of 16

# If CUDA not found but you have GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Android Build Issues:**
```bash
# If Gradle build fails
cd circle-detection-app
./gradlew clean
./gradlew build
```

**Camera Permission Issues:**
- Make sure permissions are added to AndroidManifest.xml
- Check device settings allow camera access

**Model Loading Issues:**
- Verify model file is in `assets/` folder
- Check model file size (should be < 50MB for mobile)

### Performance Optimization:

1. **Model Size**: Use YOLOv8n (nano) for better mobile performance
2. **Input Resolution**: Use 640x640 or smaller if needed
3. **Quantization**: Use INT8 quantization for faster inference
4. **Threading**: Process inference on background thread

---

## 🎉 Next Steps

After completing this guide, you'll have:
- ✅ A trained circle detection model
- ✅ An Android app that can detect circles
- ✅ Understanding of the complete ML pipeline

**Potential Improvements:**
1. Collect more training data for better accuracy
2. Implement data augmentation techniques
3. Add circle size classification
4. Improve UI/UX with better visualization
5. Add multiple circle detection with tracking

**Learning Resources:**
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Android Camera2 API](https://developer.android.com/training/camera2)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)

Good luck with your circle detection project! 🎯 