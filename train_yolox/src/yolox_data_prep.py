#!/usr/bin/env python3
"""
YOLOX Data Preparation Script
Converts YOLO format labels to COCO format and prepares dataset for YOLOX training
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

def cleanup_dataset():
    """Clean up existing dataset directories to start fresh"""
    print("🧹 Cleaning up existing dataset directories...")
    
    directories_to_clean = [
        'data/train',
        'data/val', 
        'data/test',
        'data/annotations',
        'YOLOX/datasets/circle_dataset'
    ]
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"   Removed: {directory}")
    
    # Also remove any existing COCO annotation files
    coco_files = [
        'data/train/labels.json',
        'data/val/labels.json', 
        'data/test/labels.json'
    ]
    
    for coco_file in coco_files:
        if os.path.exists(coco_file):
            os.remove(coco_file)
            print(f"   Removed: {coco_file}")
    
    print("✅ Cleanup completed")

def get_image_files(raw_dir):
    """Get only .jpg image files from raw directory"""
    image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.jpg')]
    return sorted(image_files)

def get_matching_labels(image_files, raw_dir):
    """Get label files that correspond to the image files"""
    label_files = []
    for img_file in image_files:
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(raw_dir, label_file)
        if os.path.exists(label_path):
            label_files.append(label_file)
        else:
            print(f"⚠️  Warning: No label file found for {img_file}")
    
    return label_files

def convert_yolo_to_coco(yolo_labels_dir, images_dir, output_file):
    """
    Convert YOLO format labels to COCO format
    
    Args:
        yolo_labels_dir: Directory containing YOLO format .txt files
        images_dir: Directory containing corresponding images
        output_file: Output COCO JSON file path
    """
    
    coco_data = {
        "info": {
            "description": "Circle Detection Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Image Detector Project",
            "date_created": "2025-08-13"
        },
        "licenses": [
            {
                "id": 1,
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "circle",
                "supercategory": "shape"
            }
        ]
    }
    
    annotation_id = 0
    
    # Get all label files
    label_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith('.txt')]
    
    print(f"Converting {len(label_files)} label files to COCO format...")
    
    # Get list of available images
    available_images = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    available_images_set = set(available_images)
    
    processed_count = 0
    skipped_count = 0
    
    for label_file in label_files:
        # Get corresponding image file
        image_name = label_file.replace('.txt', '.jpg')
        
        # Check if image exists
        if image_name not in available_images_set:
            print(f"⚠️  Warning: Image {image_name} not found for label {label_file}, skipping")
            skipped_count += 1
            continue
        
        image_path = os.path.join(images_dir, image_name)
        
        # Get image dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"⚠️  Warning: Could not read image {image_path}: {e}")
            skipped_count += 1
            continue
        
        # Add image info
        image_id = len(coco_data["images"])
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })
        
        # Read YOLO labels
        label_path = os.path.join(yolo_labels_dir, label_file)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️  Warning: Could not read label file {label_path}: {e}")
            skipped_count += 1
            continue
        
        # Convert YOLO format to COCO format
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                if len(parts) != 5:
                    print(f"⚠️  Warning: Invalid YOLO format in {label_path}: {line}")
                    continue
                
                class_id, x_center, y_center, label_width, label_height = parts
                class_id = int(class_id)
                x_center = float(x_center)
                y_center = float(y_center)
                label_width = float(label_width)
                label_height = float(label_height)
                
                # Convert normalized coordinates (0-1) to absolute pixel coordinates
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                width_abs = label_width * width  # Normalized width * image width
                height_abs = label_height * height  # Normalized height * image height
                
                # Convert center coordinates to top-left coordinates for COCO format
                x = x_center_abs - width_abs / 2
                y = y_center_abs - height_abs / 2
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width_abs = min(width_abs, width - x)
                height_abs = min(height_abs, height - y)
                
                # Validate bounding box dimensions
                if width_abs <= 0 or height_abs <= 0:
                    print(f"⚠️  Warning: Invalid bounding box dimensions in {label_path}: width={width_abs:.2f}, height={height_abs:.2f}")
                    continue
                
                if width_abs > width or height_abs > height:
                    print(f"⚠️  Warning: Bounding box exceeds image dimensions in {label_path}: bbox=[{x:.2f}, {y:.2f}, {width_abs:.2f}, {height_abs:.2f}], image={width}x{height}")
                    continue
                
                # Only process class ID 0 (circles) - skip other classes
                if class_id == 0:
                    # Add annotation
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, width_abs, height_abs],
                        "area": width_abs * height_abs,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                else:
                    print(f"⚠️  Warning: Skipping class ID {class_id} in {label_path} (only class 0 is supported)")
                
            except Exception as e:
                print(f"⚠️  Warning: Error processing line in {label_path}: {line}, Error: {e}")
                continue
        
        processed_count += 1
    
    # Save COCO format file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted to COCO format: {output_file}")
    print(f"   Processed: {processed_count} files")
    print(f"   Skipped: {skipped_count} files")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    
    # Print coordinate transformation summary
    if coco_data["annotations"]:
        sample_bbox = coco_data["annotations"][0]["bbox"]
        print(f"   Sample bbox: {sample_bbox} (COCO format: [x, y, width, height])")
        print(f"   Coordinate system: YOLO normalized (0-1) → COCO absolute pixels")
    
    return coco_data

def prepare_yolox_dataset():
    """
    Prepare dataset for YOLOX training by:
    1. Cleaning up existing dataset
    2. Converting YOLO labels to COCO format
    3. Creating train/val/test splits
    4. Organizing files for YOLOX
    """
    
    # Ensure we're in the project root
    if not (os.path.exists('yolox') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: yolox/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        sys.exit(1)
    
    print("🎯 === YOLOX Dataset Preparation ===")
    
    # Clean up existing dataset first
    cleanup_dataset()
    
    # Check if raw data exists
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir):
        print("❌ Error: Raw data directory not found: data/raw/")
        print("   Please place your images and YOLO labels in data/raw/")
        sys.exit(1)
    
    # Find image and label files
    image_files = get_image_files(raw_dir)
    label_files = get_matching_labels(image_files, raw_dir)
    
    if not image_files:
        print("❌ Error: No image files found in data/raw/")
        print("   Please add .jpg files")
        sys.exit(1)
    
    if not label_files:
        print("❌ Error: No label files found in data/raw/")
        print("   Please add .txt files with YOLO format labels")
        sys.exit(1)
    
    # Create proper image-label pairs
    print("🔍 Creating image-label pairs...")
    image_label_pairs = []
    for img_file in image_files:
        label_file = img_file.replace('.jpg', '.txt')
        if label_file in label_files:
            image_label_pairs.append((img_file, label_file))
        else:
            print(f"⚠️  Warning: No label file found for {img_file}, skipping")
    
    if not image_label_pairs:
        print("❌ Error: No valid image-label pairs found")
        sys.exit(1)
    
    print(f"   Found {len(image_label_pairs)} valid image-label pairs")
    
    # Create train/val/test splits (70/20/10)
    total_files = len(image_label_pairs)
    train_count = int(total_files * 0.7)
    val_count = int(total_files * 0.2)
    test_count = total_files - train_count - val_count
    
    print(f"Dataset split: {train_count} train, {val_count} val, {test_count} test")
    
    # Shuffle files for random split
    import random
    random.seed(42)  # For reproducible splits
    random.shuffle(image_label_pairs)
    
    # Split files
    train_files = image_label_pairs[:train_count]
    val_files = image_label_pairs[train_count:train_count + val_count]
    test_files = image_label_pairs[train_count + val_count:]
    
    # Create YOLOX dataset directories where config expects them
    print("📁 Creating YOLOX dataset directories...")
    yolox_data_dir = 'YOLOX/datasets/circle_dataset'
    os.makedirs(f'{yolox_data_dir}/train/images', exist_ok=True)
    os.makedirs(f'{yolox_data_dir}/val/images', exist_ok=True)
    os.makedirs(f'{yolox_data_dir}/test/images', exist_ok=True)
    os.makedirs(f'{yolox_data_dir}/annotations', exist_ok=True)
    
    # Copy files directly to YOLOX dataset directories
    def copy_files_to_yolox(file_list, target_img_dir, split_name):
        print(f"📁 Copying {len(file_list)} files to YOLOX {split_name} set...")
        for img_file, label_file in file_list:
            # Copy image to YOLOX dataset
            src_img = os.path.join(raw_dir, img_file)
            dst_img = os.path.join(target_img_dir, img_file)
            shutil.copy2(src_img, dst_img)
        
        print(f"   ✅ YOLOX {split_name}: {len(file_list)} images copied")
    
    print("📁 Organizing dataset files for YOLOX...")
    copy_files_to_yolox(train_files, f'{yolox_data_dir}/train/images', 'train')
    copy_files_to_yolox(val_files, f'{yolox_data_dir}/val/images', 'validation')
    copy_files_to_yolox(test_files, f'{yolox_data_dir}/test/images', 'test')
    
    print("✅ YOLOX dataset files organized")
    
    # Convert YOLO labels to COCO format directly for YOLOX
    print("🔄 Converting labels to COCO format for YOLOX...")
    
    # Convert train set
    train_coco = convert_yolo_to_coco(
        raw_dir,  # Use raw directory for labels
        f'{yolox_data_dir}/train/images',  # Use YOLOX train images
        f'{yolox_data_dir}/annotations/train_labels.json'
    )
    
    # Convert validation set
    val_coco = convert_yolo_to_coco(
        raw_dir,  # Use raw directory for labels
        f'{yolox_data_dir}/val/images',  # Use YOLOX val images
        f'{yolox_data_dir}/annotations/val_labels.json'
    )
    
    # Convert test set
    test_coco = convert_yolo_to_coco(
        raw_dir,  # Use raw directory for labels
        f'{yolox_data_dir}/test/images',  # Use YOLOX test images
        f'{yolox_data_dir}/annotations/test_labels.json'
    )
    
    print("✅ COCO format files created in YOLOX dataset directory")
    
    # Create YOLOX dataset configuration
    create_yolox_dataset_config()
    
    print("\n🎉 YOLOX dataset preparation completed!")
    print("   Dataset is now ready at: YOLOX/datasets/circle_dataset/")
    print("   You can now run: python src/yolox_train.py")

def create_yolox_dataset_config():
    """Create YOLOX dataset configuration file"""
    
    config_content = '''# YOLOX Circle Detection Dataset Configuration
# This file tells YOLOX where to find your training data

# Dataset path (absolute path)
path: /data/code/image-detector/data

# Training, validation, and test sets
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 1

# Class names
names:
  0: circle

# COCO format annotation files
train_annotations: train/labels.json
val_annotations: val/labels.json
test_annotations: test/labels.json

# Dataset info
download: false
roboflow: false
'''
    
    # Ensure target directory exists
    os.makedirs('config', exist_ok=True)

    with open('config/yolox_dataset.yaml', 'w') as f:
        f.write(config_content)
    
    print("✅ Created YOLOX dataset configuration: config/yolox_dataset.yaml")

def verify_dataset():
    """Verify the prepared dataset"""
    
    print("\n🔍 Verifying dataset...")
    
    # Check YOLOX dataset structure
    yolox_data_dir = 'YOLOX/datasets/circle_dataset'
    
    if os.path.exists(yolox_data_dir):
        print(f"✅ YOLOX dataset directory exists: {yolox_data_dir}")
        
        # Check file counts in YOLOX dataset
        train_images = len(os.listdir(f'{yolox_data_dir}/train/images'))
        val_images = len(os.listdir(f'{yolox_data_dir}/val/images'))
        test_images = len(os.listdir(f'{yolox_data_dir}/test/images'))
        
        print(f"YOLOX Train: {train_images} images")
        print(f"YOLOX Val: {val_images} images")
        print(f"YOLOX Test: {test_images} images")
        
        # Check COCO annotation files in YOLOX dataset
        yolox_coco_files = [
            f'{yolox_data_dir}/annotations/train_labels.json',
            f'{yolox_data_dir}/annotations/val_labels.json',
            f'{yolox_data_dir}/annotations/test_labels.json'
        ]
        
        for coco_file in yolox_coco_files:
            if os.path.exists(coco_file):
                with open(coco_file, 'r') as f:
                    data = json.load(f)
                    print(f"✅ {os.path.basename(coco_file)}: {len(data['images'])} images, {len(data['annotations'])} annotations")
            else:
                print(f"❌ {os.path.basename(coco_file)} not found")
    else:
        print(f"❌ YOLOX dataset directory not found: {yolox_data_dir}")
    
    print("\n🎯 Dataset verification completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare YOLOX Circle Detection Dataset')
    parser.add_argument('--verify', action='store_true', help='Verify the prepared dataset')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset()
    else:
        prepare_yolox_dataset()
