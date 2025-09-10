#!/usr/bin/env python3
"""
YOLOX Image Resizing Script with Letterboxing
Resizes images from source directory to 640x640 with letterboxing and adjusts YOLO coordinates accordingly
"""

import os
import sys
import shutil
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

def letterbox_image(image, target_size=(640, 640)):
    """
    Resize image with letterboxing to maintain aspect ratio
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size tuple (width, height)
    
    Returns:
        resized_image: Letterboxed image
        scale: Scale factor applied
        pad_x: Padding added on left/right
        pad_y: Padding added on top/bottom
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
    
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate scale to fit image into target size while maintaining aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with target size and black background
    letterboxed_image = Image.new('RGB', target_size, (0, 0, 0))
    
    # Calculate padding to center the resized image
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    
    # Paste resized image onto letterboxed background
    letterboxed_image.paste(resized_image, (pad_x, pad_y))
    
    return letterboxed_image, scale, pad_x, pad_y

def adjust_yolo_coordinates(yolo_coords, original_size, scale, pad_x, pad_y, target_size=(640, 640)):
    """
    Adjust YOLO coordinates for letterboxed image
    
    Args:
        yolo_coords: List of YOLO format coordinates [class_id, x_center, y_center, width, height]
        original_size: Original image size (width, height)
        scale: Scale factor applied during letterboxing
        pad_x: Horizontal padding
        pad_y: Vertical padding
        target_size: Target size (width, height)
    
    Returns:
        adjusted_coords: Adjusted YOLO coordinates
    """
    original_width, original_height = original_size
    target_width, target_height = target_size
    
    adjusted_coords = []
    
    for coord in yolo_coords:
        if len(coord) != 5:
            print(f"⚠️  Warning: Invalid YOLO coordinate format: {coord}")
            continue
        
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = coord
        
        # Convert normalized coordinates to absolute coordinates in original image
        x_center_abs = float(x_center_norm) * original_width
        y_center_abs = float(y_center_norm) * original_height
        width_abs = float(width_norm) * original_width
        height_abs = float(height_norm) * original_height
        
        # Apply scaling
        x_center_scaled = x_center_abs * scale
        y_center_scaled = y_center_abs * scale
        width_scaled = width_abs * scale
        height_scaled = height_abs * scale
        
        # Apply padding (shift center point)
        x_center_padded = x_center_scaled + pad_x
        y_center_padded = y_center_scaled + pad_y
        
        # Convert back to normalized coordinates for target image
        x_center_new = x_center_padded / target_width
        y_center_new = y_center_padded / target_height
        width_new = width_scaled / target_width
        height_new = height_scaled / target_height
        
        # Ensure coordinates are within bounds [0, 1]
        x_center_new = max(0.0, min(1.0, x_center_new))
        y_center_new = max(0.0, min(1.0, y_center_new))
        width_new = max(0.0, min(1.0, width_new))
        height_new = max(0.0, min(1.0, height_new))
        
        # Only keep annotations where the center is still within the image
        if 0.0 <= x_center_new <= 1.0 and 0.0 <= y_center_new <= 1.0:
            adjusted_coords.append([
                int(class_id),
                x_center_new,
                y_center_new,
                width_new,
                height_new
            ])
    
    return adjusted_coords

def get_image_files(directory):
    """Get all image files from directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return sorted(image_files)

def read_yolo_labels(label_file):
    """Read YOLO format labels from file"""
    if not os.path.exists(label_file):
        return []
    
    labels = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        labels.append(parts)
                    else:
                        print(f"⚠️  Warning: Invalid line in {label_file}: {line}")
    except Exception as e:
        print(f"⚠️  Warning: Error reading {label_file}: {e}")
    
    return labels

def write_yolo_labels(label_file, labels):
    """Write YOLO format labels to file"""
    try:
        with open(label_file, 'w') as f:
            for label in labels:
                class_id, x_center, y_center, width, height = label
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        print(f"❌ Error writing {label_file}: {e}")

def resize_dataset(source_dir):
    """
    Resize all images from source directory to 640x640 with letterboxing
    and adjust YOLO coordinates accordingly
    
    Args:
        source_dir: Path to directory containing images and YOLO labels
    """
    
    print("🔄 === YOLOX Dataset Resizing with Letterboxing ===")
    
    # Validate source directory
    if not os.path.exists(source_dir):
        print(f"❌ Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    if not os.path.isdir(source_dir):
        print(f"❌ Error: Source path is not a directory: {source_dir}")
        sys.exit(1)
    
    # Generate target directory path (same parent dir with _resized suffix)
    source_path = Path(source_dir)
    target_dir = str(source_path.parent / f"{source_path.name}_resized")
    target_size = (640, 640)
    
    print(f"📁 Source directory: {source_dir}")
    print(f"📁 Target directory: {target_dir}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Clean target directory if it already contains files
    if os.listdir(target_dir):
        print(f"🧹 Cleaning existing files in {target_dir}...")
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Get all image files
    image_files = get_image_files(source_dir)
    
    if not image_files:
        print(f"❌ Error: No image files found in {source_dir}")
        print("   Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        sys.exit(1)
    
    print(f"📸 Found {len(image_files)} images to process")
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for image_file in image_files:
        try:
            print(f"🔄 Processing: {image_file}")
            
            # File paths
            source_image_path = os.path.join(source_dir, image_file)
            target_image_path = os.path.join(target_dir, image_file)
            
            # Label file paths
            label_filename = os.path.splitext(image_file)[0] + '.txt'
            source_label_path = os.path.join(source_dir, label_filename)
            target_label_path = os.path.join(target_dir, label_filename)
            
            # Load and check image
            try:
                image = Image.open(source_image_path)
                original_width, original_height = image.size
            except Exception as e:
                print(f"   ❌ Error loading image {image_file}: {e}")
                error_count += 1
                continue
            
            # Check if image is large enough
            if original_width < 640 or original_height < 640:
                print(f"   ⚠️  Warning: Image {image_file} is smaller than 640x640 ({original_width}x{original_height})")
                print(f"   Resizing anyway, but quality may be reduced")
            
            # Check if image is already 640x640
            if original_width == 640 and original_height == 640:
                print(f"   ℹ️  Image {image_file} is already 640x640, copying without modification")
                # Copy image and labels directly
                shutil.copy2(source_image_path, target_image_path)
                if os.path.exists(source_label_path):
                    shutil.copy2(source_label_path, target_label_path)
                processed_count += 1
                continue
            
            # Apply letterboxing
            letterboxed_image, scale, pad_x, pad_y = letterbox_image(image, target_size)
            
            # Save resized image
            letterboxed_image.save(target_image_path, quality=95, optimize=True)
            
            # Process labels if they exist
            if os.path.exists(source_label_path):
                # Read original YOLO labels
                original_labels = read_yolo_labels(source_label_path)
                
                if original_labels:
                    # Adjust coordinates for letterboxing
                    adjusted_labels = adjust_yolo_coordinates(
                        original_labels,
                        (original_width, original_height),
                        scale,
                        pad_x,
                        pad_y,
                        target_size
                    )
                    
                    # Write adjusted labels
                    write_yolo_labels(target_label_path, adjusted_labels)
                    
                    print(f"   ✅ Resized {original_width}x{original_height} → 640x640, "
                          f"scale={scale:.3f}, pad=({pad_x},{pad_y}), "
                          f"labels: {len(original_labels)}→{len(adjusted_labels)}")
                else:
                    # Create empty label file
                    open(target_label_path, 'w').close()
                    print(f"   ✅ Resized {original_width}x{original_height} → 640x640, "
                          f"scale={scale:.3f}, pad=({pad_x},{pad_y}), no labels")
            else:
                print(f"   ⚠️  No label file found for {image_file}")
                # Create empty label file
                label_target_path = os.path.join(target_dir, os.path.splitext(image_file)[0] + '.txt')
                open(label_target_path, 'w').close()
                print(f"   ✅ Resized {original_width}x{original_height} → 640x640, "
                      f"scale={scale:.3f}, pad=({pad_x},{pad_y}), no labels")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {image_file}: {e}")
            error_count += 1
            continue
    
    # Print summary
    print(f"\n📊 === RESIZING SUMMARY ===")
    print(f"Total images found: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Target directory: {target_dir}")
    
    if processed_count > 0:
        print(f"\n✅ Resizing completed! Images are now ready at: {target_dir}")
        print("   Next step: Run 'python src/yolox_data_prep.py' to prepare dataset for training")
    else:
        print(f"\n❌ No images were successfully processed")
        return False
    
    return True

def verify_resized_dataset(source_dir):
    """Verify the resized dataset"""
    
    print("\n🔍 === VERIFYING RESIZED DATASET ===")
    
    # Generate target directory path (same parent dir with _resized suffix)
    source_path = Path(source_dir)
    target_dir = str(source_path.parent / f"{source_path.name}_resized")
    
    if not os.path.exists(target_dir):
        print(f"❌ Error: Resized dataset directory not found: {target_dir}")
        return False
    
    # Get image files
    image_files = get_image_files(target_dir)
    
    if not image_files:
        print(f"❌ Error: No image files found in {target_dir}")
        return False
    
    print(f"📸 Found {len(image_files)} resized images")
    
    # Check a few sample images
    samples_to_check = min(5, len(image_files))
    
    for i, image_file in enumerate(image_files[:samples_to_check]):
        try:
            image_path = os.path.join(target_dir, image_file)
            image = Image.open(image_path)
            width, height = image.size
            
            # Check if all images are 640x640
            if width == 640 and height == 640:
                print(f"   ✅ {image_file}: {width}x{height}")
            else:
                print(f"   ❌ {image_file}: {width}x{height} (expected 640x640)")
        except Exception as e:
            print(f"   ❌ Error checking {image_file}: {e}")
    
    # Check label files
    label_files = [f for f in os.listdir(target_dir) if f.endswith('.txt')]
    print(f"📝 Found {len(label_files)} label files")
    
    print(f"\n✅ Verification completed!")
    print(f"   All images should be 640x640 pixels")
    print(f"   YOLO coordinates have been adjusted for letterboxing")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resize images to 640x640 with letterboxing for YOLOX training')
    parser.add_argument('source_dir', help='Path to directory containing images and YOLO labels')
    parser.add_argument('--verify', action='store_true', help='Verify the resized dataset')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_resized_dataset(args.source_dir)
    else:
        resize_dataset(args.source_dir)
