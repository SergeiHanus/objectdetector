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

def cleanup_dataset(data_dir=None):
    """Clean up existing dataset directories to start fresh
    
    Args:
        data_dir: Specific dataset directory to clean. If None, cleans default locations.
    """
    print("🧹 Cleaning up existing dataset directories...")
    
    directories_to_clean = [
        'data/train',
        'data/val', 
        'data/test',
        'data/annotations'
    ]
    
    # Add specific data_dir if provided
    if data_dir:
        directories_to_clean.append(data_dir)
    else:
        # Clean common dataset directories
        # Clean all existing dataset directories
        datasets_dir = 'YOLOX/datasets'
        if os.path.exists(datasets_dir):
            dataset_dirs = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir) 
                          if os.path.isdir(os.path.join(datasets_dir, d)) and d.endswith('_dataset')]
            directories_to_clean.extend(dataset_dirs)
    
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
    """Get .jpg, .JPG, .png, and .PNG image files from raw directory"""
    image_files = [f for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
    return sorted(image_files)

def get_matching_labels(image_files, raw_dir):
    """Get label files that correspond to the image files"""
    label_files = []
    for img_file in image_files:
        # Handle .jpg, .JPG, .png, and .PNG extensions
        if img_file.endswith('.jpg'):
            label_file = img_file.replace('.jpg', '.txt')
        elif img_file.endswith('.JPG'):
            label_file = img_file.replace('.JPG', '.txt')
        elif img_file.endswith('.png'):
            label_file = img_file.replace('.png', '.txt')
        elif img_file.endswith('.PNG'):
            label_file = img_file.replace('.PNG', '.txt')
        else:
            continue
            
        label_path = os.path.join(raw_dir, label_file)
        if os.path.exists(label_path):
            label_files.append(label_file)
        else:
            print(f"⚠️  Warning: No label file found for {img_file}")
    
    return label_files

def convert_yolo_to_coco(yolo_labels_dir, images_dir, output_file, class_name="object"):
    """
    Convert YOLO format labels to COCO format
    
    Args:
        yolo_labels_dir: Directory containing YOLO format .txt files
        images_dir: Directory containing corresponding images
        output_file: Output COCO JSON file path
        class_name: Name of the detection class
    """
    
    coco_data = {
        "info": {
            "description": f"{class_name.title()} Detection Dataset",
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
                "name": class_name,
                "supercategory": "object"
            }
        ]
    }
    
    annotation_id = 0
    
    # Get all label files
    label_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith('.txt')]
    
    print(f"Converting {len(label_files)} label files to COCO format...")
    
    # Get list of available images
    available_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
    available_images_set = set(available_images)
    
    processed_count = 0
    skipped_count = 0
    
    for label_file in label_files:
        # Get corresponding image file - try .jpg, .JPG, .png, and .PNG extensions
        image_name_jpg = label_file.replace('.txt', '.jpg')
        image_name_JPG = label_file.replace('.txt', '.JPG')
        image_name_png = label_file.replace('.txt', '.png')
        image_name_PNG = label_file.replace('.txt', '.PNG')
        
        # Check which image exists
        if image_name_jpg in available_images_set:
            image_name = image_name_jpg
        elif image_name_JPG in available_images_set:
            image_name = image_name_JPG
        elif image_name_png in available_images_set:
            image_name = image_name_png
        elif image_name_PNG in available_images_set:
            image_name = image_name_PNG
        else:
            print(f"⚠️  Warning: Image not found for label {label_file} (tried {image_name_jpg}, {image_name_JPG}, {image_name_png}, and {image_name_PNG}), skipping")
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

def get_class_name_from_file(raw_dir):
    """
    Read the first class name from classes.txt file in the raw directory
    
    Args:
        raw_dir: Path to directory containing classes.txt
        
    Returns:
        str: First class name from classes.txt
    """
    classes_file = os.path.join(raw_dir, 'classes.txt')
    
    if not os.path.exists(classes_file):
        print(f"❌ Error: classes.txt not found in {raw_dir}")
        print(f"   Please create {classes_file} with class names (one per line)")
        sys.exit(1)
    
    try:
        with open(classes_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"❌ Error: {classes_file} is empty")
                print("   Please add class names (one per line)")
                sys.exit(1)
            
            class_name = lines[0].strip()
            if not class_name:
                print(f"❌ Error: First line in {classes_file} is empty")
                print("   Please add class names (one per line)")
                sys.exit(1)
            
            print(f"📝 Using class name: '{class_name}' from {classes_file}")
            return class_name
            
    except Exception as e:
        print(f"❌ Error reading {classes_file}: {e}")
        sys.exit(1)

def create_dynamic_config(class_name):
    """
    Create dynamic YOLOX configuration files based on class name
    
    Args:
        class_name: Name of the detection class
    """
    
    print(f"\n⚙️  Creating dynamic configuration for '{class_name}' class...")
    
    # Create exps directory
    os.makedirs('exps', exist_ok=True)
    
    # Create base YOLOX-S configuration
    base_config = f'''# YOLOX-S base configuration
_base_ = [
    '../yolox_s_300e_coco.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_300e.py',
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    bbox_head=dict(
        num_classes=1,  # Single class detection
    )
)

# Training settings
total_epochs = 100
evaluation = dict(interval=10, metric='bbox')
save_checkpoint_interval = 10
log_interval = 50

# Export settings
export = dict(
    type='onnx',
    input_shape=(1, 3, 640, 640),
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={{
        'input': {{0: 'batch_size'}},
        'output': {{0: 'batch_size'}}
    }}
)
'''
    
    with open('exps/yolox_s_base.py', 'w') as f:
        f.write(base_config)
    
    print("✅ Created base YOLOX configuration: exps/yolox_s_base.py")
    
    # Create class-specific configuration with proper YOLOX experiment class
    class_config = f'''#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 1
        # Default data_dir - will be updated by training script if needed
        self.data_dir = self._get_default_data_dir()
        self.train_ann = "train_labels.json"
        self.val_ann = "val_labels.json"
        self.test_ann = "test_labels.json"
        
        # Memory optimization settings
        self.data_num_workers = 4  # Minimal workers to save memory
        self.batch_size = 8  # Increased batch size for better training performance
        self.input_size = (640, 640)  # Standard YOLOX size for better accuracy
        self.test_size = (640, 640)  # Standard YOLOX size for better accuracy
        self.multiscale_range = 0  # Disable multiscale training to save memory
        self.cache = True
        
        # Training settings
        self.max_epoch = 80  # Restored to normal training epochs
        self.warmup_epochs = 10 # Restored to normal warmup epochs
        self.basic_lr_per_img = 0.0001  # Reduced learning rate for stability
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.937
        self.print_interval = 10  # Print only at the end of each epoch (4 iterations)
        self.eval_interval = 10
        self.save_interval = 10
        
        # Augmentation settings - Reduced for initial training stability
        self.degrees = 5.0  # Reduced rotation
        self.translate = 0.05  # Reduced translation
        self.scale = (0.8, 1.2)  # Reduced scaling range
        self.mscale = (0.9, 1.1)  # Reduced multiscale range
        self.shear = 1.0  # Reduced shear
        self.perspective = 0.0
        self.enable_mixup = False
        
        # Additional properties needed for YOLOX DataLoader
        self.max_labels = 50
        self.flip = 0.5  # flip probability
        self.seed = None  # random seed
        
        # Loss settings
        self.mosaic_prob = 0.0
        self.copy_paste_prob = 0.0
        
        # Class names for {class_name} detection
        self.class_names = ["{class_name}"]
        
        # Export settings
        self.export_input_names = ["input"]
        self.export_output_names = ["output"]
        self.export_dynamic_axes = {{
            "input": {{0: "batch_size"}},
            "output": {{0: "batch_size"}}
        }}
    
    def _get_default_data_dir(self):
        """Get default data directory based on experiment name"""
        # Extract class name from experiment name (yolox_s_{class_name} -> {class_name})
        class_name = self.exp_name.replace("yolox_s_", "")
        return f"datasets/{class_name}_dataset"
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Override get_data_loader method as required by YOLOX documentation"""
        from yolox.data import COCODataset, TrainTransform, MosaicDetection, YoloBatchSampler, DataLoader, InfiniteSampler, worker_init_reset_seed
        
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train/images",
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=self.max_labels, flip_prob=self.flip, hsv_prob=0.015),
            cache=cache_img,
            cache_type="ram",
        )
        
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=self.max_labels, flip_prob=self.flip, hsv_prob=0.015),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
        )
        
        if is_distributed:
            import torch.distributed as dist
            batch_size = batch_size // dist.get_world_size()
        
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        
        dataloader_kwargs = {{"num_workers": self.data_num_workers, "pin_memory": True}}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        
        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Override get_eval_loader method as required by YOLOX documentation"""
        from yolox.data import COCODataset, ValTransform
        import torch
        from torch.utils.data import DataLoader
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val/images" if not testdev else "test/images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        
        if is_distributed:
            batch_size = batch_size // torch.distributed.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {{
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }}
        
        val_loader = DataLoader(valdataset, batch_size=batch_size, **dataloader_kwargs)
        
        return val_loader
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Get evaluator for validation"""
        from yolox.evaluators import COCOEvaluator
        
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
    
    def verify_dataset_config(self):
        """Verify that the dataset configuration is correct"""
        print(f"🔍 Verifying dataset configuration for {{self.exp_name}}...")
        
        # Check data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {{self.data_dir}}")
        
        # Check annotation files exist
        train_ann_path = os.path.join(self.data_dir, "annotations", self.train_ann)
        val_ann_path = os.path.join(self.data_dir, "annotations", self.val_ann)
        
        if not os.path.exists(train_ann_path):
            raise FileNotFoundError(f"Training annotations not found: {{train_ann_path}}")
        
        if not os.path.exists(val_ann_path):
            raise FileNotFoundError(f"Validation annotations not found: {{val_ann_path}}")
        
        # Check image directories exist
        train_img_dir = os.path.join(self.data_dir, "train", "images")
        val_img_dir = os.path.join(self.data_dir, "val", "images")
        
        if not os.path.exists(train_img_dir):
            raise FileNotFoundError(f"Training images directory not found: {{train_img_dir}}")
        
        if not os.path.exists(val_img_dir):
            raise FileNotFoundError(f"Validation images directory not found: {{val_img_dir}}")
        
        # Count images
        train_images = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_images = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"✅ Dataset verification passed:")
        print(f"   Data directory: {{self.data_dir}}")
        print(f"   Training images: {{train_images}}")
        print(f"   Validation images: {{val_images}}")
        print(f"   Class: {{self.class_names[0]}}")
        print(f"   Input size: {{self.input_size}}")
        
        return True
'''
    
    config_filename = f'exps/yolox_s_{class_name}.py'
    
    # Remove existing exp file if it exists
    if os.path.exists(config_filename):
        os.remove(config_filename)
        print(f"🗑️  Removed existing configuration: {config_filename}")
    
    with open(config_filename, 'w') as f:
        f.write(class_config)
    
    print(f"✅ Created {class_name} detection configuration: {config_filename}")
    return config_filename

def prepare_yolox_dataset(raw_dir='data/raw_resized', data_dir=None):
    """
    Prepare dataset for YOLOX training by:
    1. Reading class name from classes.txt in raw_dir
    2. Creating dynamic configuration files
    3. Cleaning up existing dataset
    4. Converting YOLO labels to COCO format
    5. Creating train/val/test splits
    6. Organizing files for YOLOX
    
    Args:
        raw_dir: Path to directory containing raw images, YOLO labels, and classes.txt
        data_dir: Target dataset directory. If None, defaults to YOLOX/datasets/<class_name>
    """
    
    # Ensure we're in the train_yolox project root
    if not (os.path.exists('yolox') and os.path.exists('YOLOX') and os.path.exists('src')):
        print("❌ Error: This script must be run from the train_yolox project root directory")
        print("   Expected structure: yolox/, YOLOX/, src/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: train_yolox/")
        sys.exit(1)
    
    print("🎯 === YOLOX Dataset Preparation ===")
    
    # Check if raw data exists
    if not os.path.exists(raw_dir):
        print(f"❌ Error: Raw data directory not found: {raw_dir}")
        print(f"   Please place your images and YOLO labels in {raw_dir}")
        sys.exit(1)
    
    # Get class name from classes.txt
    class_name = get_class_name_from_file(raw_dir)
    
    # Create dynamic configuration files
    config_filename = create_dynamic_config(class_name)
    
    # Set data_dir if not provided
    if data_dir is None:
        data_dir = f'YOLOX/datasets/{class_name}_dataset'
    
    print(f"📁 Target dataset directory: {data_dir}")
    
    # Clean up existing dataset first
    cleanup_dataset(data_dir)
    
    # Find image and label files
    image_files = get_image_files(raw_dir)
    label_files = get_matching_labels(image_files, raw_dir)
    
    if not image_files:
        print("❌ Error: No image files found in raw_resized")
        print("   Please add .jpg files")
        sys.exit(1)
    
    if not label_files:
        print("❌ Error: No label files found in raw_resized")
        print("   Please add .txt files with YOLO format labels")
        sys.exit(1)
    
    # Create proper image-label pairs
    print("🔍 Creating image-label pairs...")
    image_label_pairs = []
    for img_file in image_files:
        # Handle .jpg, .JPG, .png, and .PNG extensions
        if img_file.endswith('.jpg'):
            label_file = img_file.replace('.jpg', '.txt')
        elif img_file.endswith('.JPG'):
            label_file = img_file.replace('.JPG', '.txt')
        elif img_file.endswith('.png'):
            label_file = img_file.replace('.png', '.txt')
        elif img_file.endswith('.PNG'):
            label_file = img_file.replace('.PNG', '.txt')
        else:
            continue
            
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
    
    # Create YOLOX dataset directories
    print("📁 Creating YOLOX dataset directories...")
    yolox_data_dir = data_dir
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
        f'{yolox_data_dir}/annotations/train_labels.json',
        class_name
    )
    
    # Convert validation set
    val_coco = convert_yolo_to_coco(
        raw_dir,  # Use raw directory for labels
        f'{yolox_data_dir}/val/images',  # Use YOLOX val images
        f'{yolox_data_dir}/annotations/val_labels.json',
        class_name
    )
    
    # Convert test set
    test_coco = convert_yolo_to_coco(
        raw_dir,  # Use raw directory for labels
        f'{yolox_data_dir}/test/images',  # Use YOLOX test images
        f'{yolox_data_dir}/annotations/test_labels.json',
        class_name
    )
    
    print("✅ COCO format files created in YOLOX dataset directory")
    
    # Automatically verify the prepared dataset
    print(f"\n🔍 Verifying prepared dataset...")
    _verify_single_dataset(yolox_data_dir)
    
    # Automatically validate coordinate conversion
    print(f"\n🔍 Validating coordinate conversion accuracy...")
    validation_success = validate_coordinate_conversion(raw_dir)
    
    if validation_success:
        print(f"\n🎉 YOLOX dataset preparation completed successfully!")
        print(f"   Dataset is now ready at: {yolox_data_dir}/")
        print(f"   Class name: {class_name}")
        print(f"   You can now run: python src/yolox_train.py --data-dir {yolox_data_dir}")
    else:
        print(f"\n⚠️  YOLOX dataset preparation completed with coordinate validation warnings!")
        print(f"   Dataset is ready at: {yolox_data_dir}/")
        print(f"   Class name: {class_name}")
        print(f"   Please review coordinate validation errors above.")
        print(f"   You can still run: python src/yolox_train.py --data-dir {yolox_data_dir}")
    
    return yolox_data_dir, class_name


def _verify_single_dataset(yolox_data_dir):
    """Verify a single dataset directory"""
    if os.path.exists(yolox_data_dir):
        print(f"✅ Dataset directory exists: {yolox_data_dir}")
        
        # Check file counts in YOLOX dataset
        train_images = len(os.listdir(f'{yolox_data_dir}/train/images'))
        val_images = len(os.listdir(f'{yolox_data_dir}/val/images'))
        test_images = len(os.listdir(f'{yolox_data_dir}/test/images'))
        
        print(f"   Train: {train_images} images")
        print(f"   Val: {val_images} images")
        print(f"   Test: {test_images} images")
        
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
                    print(f"   ✅ {os.path.basename(coco_file)}: {len(data['images'])} images, {len(data['annotations'])} annotations")
            else:
                print(f"   ❌ {os.path.basename(coco_file)} not found")
    else:
        print(f"❌ Dataset directory not found: {yolox_data_dir}")

def validate_coordinate_conversion(raw_dir='data/raw_resized'):
    """
    Validate that COCO coordinates correctly correspond to YOLO coordinates
    by comparing the original YOLO labels with the converted COCO annotations
    
    Args:
        raw_dir: Path to directory containing raw images and YOLO labels
    """
    print("\n🔍 === COORDINATE VALIDATION ===")
    
    # Auto-detect YOLOX dataset directory
    datasets_dir = 'YOLOX/datasets'
    if not os.path.exists(datasets_dir):
        print("❌ Error: YOLOX/datasets/ directory not found")
        print("   Please run data preparation first: python src/yolox_data_prep.py")
        return False
        
    dataset_dirs = [d for d in os.listdir(datasets_dir) 
                  if os.path.isdir(os.path.join(datasets_dir, d)) and d.endswith('_dataset')]
    
    if not dataset_dirs:
        print("❌ Error: No dataset directories found in YOLOX/datasets/")
        print("   Please run data preparation first: python src/yolox_data_prep.py")
        return False
    
    # Use the first dataset found for validation
    yolox_data_dir = os.path.join(datasets_dir, dataset_dirs[0])
    print(f"📁 Validating dataset: {dataset_dirs[0]}")
    
    # Check if required directories exist
    if not os.path.exists(raw_dir):
        print(f"❌ Error: Raw data directory not found: {raw_dir}")
        return False
    
    if not os.path.exists(yolox_data_dir):
        print(f"❌ Error: YOLOX dataset directory not found: {yolox_data_dir}")
        return False
    
    # Load COCO annotation files
    annotation_files = {
        'train': f'{yolox_data_dir}/annotations/train_labels.json',
        'val': f'{yolox_data_dir}/annotations/val_labels.json',
        'test': f'{yolox_data_dir}/annotations/test_labels.json'
    }
    
    validation_results = {
        'total_checks': 0,
        'passed_checks': 0,
        'failed_checks': 0,
        'errors': []
    }
    
    for split, annotation_file in annotation_files.items():
        if not os.path.exists(annotation_file):
            print(f"⚠️  Warning: {split} annotation file not found: {annotation_file}")
            continue
        
        print(f"📝 Validating {split} set...", end=' ')
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image lookup by filename
        image_lookup = {img['file_name']: img for img in coco_data['images']}
        
        # Create annotations lookup by image_id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Validate each image
        for img_info in coco_data['images']:
            image_filename = img_info['file_name']
            image_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Find corresponding YOLO label file - handle .jpg, .JPG, .png, and .PNG extensions
            if image_filename.endswith('.jpg'):
                yolo_label_file = image_filename.replace('.jpg', '.txt')
            elif image_filename.endswith('.JPG'):
                yolo_label_file = image_filename.replace('.JPG', '.txt')
            elif image_filename.endswith('.png'):
                yolo_label_file = image_filename.replace('.png', '.txt')
            elif image_filename.endswith('.PNG'):
                yolo_label_file = image_filename.replace('.PNG', '.txt')
            else:
                validation_results['errors'].append(f"Unsupported image extension: {image_filename}")
                continue
                
            yolo_label_path = os.path.join(raw_dir, yolo_label_file)
            
            if not os.path.exists(yolo_label_path):
                validation_results['errors'].append(f"YOLO label file not found: {yolo_label_file}")
                continue
            
            # Read YOLO labels
            try:
                with open(yolo_label_path, 'r') as f:
                    yolo_lines = [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                validation_results['errors'].append(f"Error reading {yolo_label_file}: {e}")
                continue
            
            # Get COCO annotations for this image
            coco_annotations = annotations_by_image.get(image_id, [])
            
            # Filter YOLO lines to only include class 0 (circles)
            yolo_class0_lines = [line for line in yolo_lines if line.startswith('0 ')]
            
            # Compare counts
            if len(yolo_class0_lines) != len(coco_annotations):
                validation_results['errors'].append(
                    f"{image_filename}: YOLO has {len(yolo_class0_lines)} class-0 objects, "
                    f"COCO has {len(coco_annotations)} annotations"
                )
                validation_results['failed_checks'] += 1
                continue
            
            # Validate each annotation pair
            image_validation_passed = True
            
            for yolo_line, coco_ann in zip(yolo_class0_lines, coco_annotations):
                validation_results['total_checks'] += 1
                
                # Parse YOLO format
                parts = yolo_line.split()
                if len(parts) != 5:
                    validation_results['errors'].append(f"{image_filename}: Invalid YOLO format: {yolo_line}")
                    validation_results['failed_checks'] += 1
                    image_validation_passed = False
                    continue
                
                _, x_center_norm, y_center_norm, width_norm, height_norm = parts
                x_center_norm = float(x_center_norm)
                y_center_norm = float(y_center_norm)
                width_norm = float(width_norm)
                height_norm = float(height_norm)
                
                # Convert YOLO to absolute coordinates
                x_center_abs = x_center_norm * img_width
                y_center_abs = y_center_norm * img_height
                width_abs = width_norm * img_width
                height_abs = height_norm * img_height
                
                # Convert to COCO format (top-left corner)
                x_coco_expected = x_center_abs - width_abs / 2
                y_coco_expected = y_center_abs - height_abs / 2
                
                # Get actual COCO values
                x_coco_actual, y_coco_actual, width_coco_actual, height_coco_actual = coco_ann['bbox']
                
                # Compare with tolerance (account for floating point precision)
                tolerance = 0.1  # pixels
                
                x_match = abs(x_coco_expected - x_coco_actual) < tolerance
                y_match = abs(y_coco_expected - y_coco_actual) < tolerance
                width_match = abs(width_abs - width_coco_actual) < tolerance
                height_match = abs(height_abs - height_coco_actual) < tolerance
                
                if x_match and y_match and width_match and height_match:
                    validation_results['passed_checks'] += 1
                else:
                    validation_results['failed_checks'] += 1
                    image_validation_passed = False
                    
                    error_msg = (
                        f"{image_filename}: Coordinate mismatch\n"
                        f"  YOLO (norm): center=({x_center_norm:.6f}, {y_center_norm:.6f}), size=({width_norm:.6f}, {height_norm:.6f})\n"
                        f"  Expected COCO: x={x_coco_expected:.2f}, y={y_coco_expected:.2f}, w={width_abs:.2f}, h={height_abs:.2f}\n"
                        f"  Actual COCO: x={x_coco_actual:.2f}, y={y_coco_actual:.2f}, w={width_coco_actual:.2f}, h={height_coco_actual:.2f}\n"
                        f"  Differences: dx={abs(x_coco_expected - x_coco_actual):.2f}, dy={abs(y_coco_expected - y_coco_actual):.2f}, "
                        f"dw={abs(width_abs - width_coco_actual):.2f}, dh={abs(height_abs - height_coco_actual):.2f}"
                    )
                    validation_results['errors'].append(error_msg)
            
            # Only log errors, not every successful validation
            if not image_validation_passed and len(yolo_class0_lines) > 0:
                print(f"  ❌ {image_filename}: Validation failed")
        
        # Show split completion
        split_total = len([img for img in coco_data['images']])
        split_errors = len([error for error in validation_results['errors'] if f"{split} set" in error or any(img['file_name'] in error for img in coco_data['images'])])
        if split_errors > 0:
            print(f"❌ {split_errors} errors found")
        else:
            split_annotations = len(coco_data['annotations'])
            print(f"✅ {split_annotations} annotations validated")
    
    # Print summary
    print(f"\n📊 === VALIDATION SUMMARY ===")
    print(f"Total coordinate checks: {validation_results['total_checks']}")
    print(f"Passed: {validation_results['passed_checks']}")
    print(f"Failed: {validation_results['failed_checks']}")
    
    if validation_results['total_checks'] > 0:
        success_rate = (validation_results['passed_checks'] / validation_results['total_checks']) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Print errors (limit to first 10)
    if validation_results['errors']:
        print(f"\n❌ Errors found ({len(validation_results['errors'])} total):")
        for i, error in enumerate(validation_results['errors'][:10]):
            print(f"{i+1}. {error}")
        
        if len(validation_results['errors']) > 10:
            print(f"... and {len(validation_results['errors']) - 10} more errors")
        
        return False
    else:
        print("✅ All coordinate validations passed!")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare YOLOX Detection Dataset')
    parser.add_argument('--raw-dir', default='data/raw_resized', 
                       help='Path to directory containing raw images, YOLO labels, and classes.txt (default: data/raw_resized)')
    parser.add_argument('--data-dir', default=None,
                       help='Target dataset directory. If not specified, defaults to YOLOX/datasets/<class_name>_dataset')
    
    args = parser.parse_args()
    
    # Always run data preparation with automatic verification
    prepare_yolox_dataset(args.raw_dir, args.data_dir)
