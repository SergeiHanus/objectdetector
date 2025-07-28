from ultralytics import YOLO
import os

def train_circle_detector(image_size=640, batch_size=16, epochs=100, patience=0, preserve_aspect_ratio=False):
    """
    Train YOLOv8 model for circle detection
    
    Args:
        image_size: Training image resolution (640, 832, 1024, 1280)
        batch_size: Batch size (reduce if running out of memory)
        epochs: Number of training epochs
        patience: Early stopping patience (0 = disabled, 20 = stop if no improvement for 20 epochs)
        preserve_aspect_ratio: If True, use rectangular input preserving original proportions (default: True)
    """
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano version for mobile deployment
    
    print("Starting training...")
    print(f"Using dataset config: config/dataset.yaml")
    print("🎯 Training Strategy: Circle detection")
    print("   • Model will learn to detect circles in images")
    print("   • Final output: Circle detections with confidence scores")
    print("   • Multiple circles can be detected in a single image")
    
    if preserve_aspect_ratio:
        # Use rectangular input (e.g., 640x480, 1024x768) based on common aspect ratios
        # You can customize these ratios based on your typical image proportions
        aspect_ratios = {
            640: (640, 480),   # 4:3 ratio
            832: (832, 624),   # 4:3 ratio  
            1024: (1024, 768), # 4:3 ratio
            1280: (1280, 960)  # 4:3 ratio
        }
        img_size = aspect_ratios.get(image_size, (image_size, int(image_size * 0.75)))
        print(f"Training resolution: {img_size[0]}x{img_size[1]} pixels (preserving aspect ratio)")
    else:
        img_size = image_size
        print(f"Training resolution: {image_size}x{image_size} pixels (square)")
    
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping: {'Disabled' if patience == 0 else f'Patience = {patience} epochs'}")
    
    # Adjust batch size based on image size to prevent memory issues
    if image_size >= 1024 and batch_size > 8:
        print(f"⚠️  Large image size detected. Reducing batch size from {batch_size} to 8 for memory efficiency.")
        batch_size = 8
    elif image_size >= 832 and batch_size > 12:
        print(f"⚠️  Medium-large image size. Reducing batch size from {batch_size} to 12.")
        batch_size = 12
    
    # Auto-detect device (GPU if available, otherwise CPU)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Adjust batch size for GPU memory (4GB GPU)
        if image_size >= 1024 and batch_size > 6:
            print(f"⚠️  Large image size + GPU memory limit. Reducing batch size from {batch_size} to 6.")
            batch_size = 6
        elif image_size >= 832 and batch_size > 8:
            print(f"⚠️  Reducing batch size from {batch_size} to 8 for optimal GPU performance.")
            batch_size = 8
    
    # Train the model for circle detection
    results = model.train(
        data='config/dataset.yaml',    # path to dataset config
        epochs=epochs,                 # number of training epochs
        imgsz=img_size,               # image size (can be tuple for rectangular)
        batch=batch_size,              # batch size (adjusted for memory)
        device=device,                 # auto-detected: 'cuda' or 'cpu'
        patience=patience,             # early stopping patience (0 = disabled)
        save=True,                     # save model checkpoints
        project='models',              # project directory
        name=f'circle_detector_{image_size}px{"_rect" if preserve_aspect_ratio else "_square"}',  # experiment name
        exist_ok=True,                 # overwrite existing
        pretrained=True,               # use pre-trained weights
        optimizer='AdamW',             # optimizer
        lr0=0.01,                      # initial learning rate
        weight_decay=0.0005,           # weight decay
        augment=True,                  # use data augmentation
        # Speed optimization parameters
        overlap_mask=False,            # disable mask overlap for speed
        mask_ratio=8,                  # increase mask downsample ratio for speed
        single_cls=True,               # train as single class (circle only)
        rect=False,                    # rectangular training (handled by imgsz)
        cache=True,                    # cache images for faster training
        workers=4,                     # reduce workers for memory efficiency
    )
    
    # Validate the model
    print("Validating model...")
    metrics = model.val()
    
    # Export to different formats for mobile with speed optimization
    print("Exporting model for mobile deployment...")
    model_name = f'circle_detector_{image_size}px{"_rect" if preserve_aspect_ratio else "_square"}'
    best_model_path = f"models/{model_name}/weights/best.pt"
    
    if os.path.exists(best_model_path):
        export_model = YOLO(best_model_path)
        # Export with speed optimizations for mobile
        export_model.export(format='tflite', int8=True, optimize=True, single_cls=True)      # TensorFlow Lite optimized
        
        # Copy TFLite model to Android assets
        os.makedirs('circle-detection-app/app/src/main/assets', exist_ok=True)
        tflite_path = best_model_path.replace('.pt', '.tflite')
        if os.path.exists(tflite_path):
            import shutil
            shutil.copy2(tflite_path, 'circle-detection-app/app/src/main/assets/circle_model.tflite')
            print("Model copied to Android assets directory")
    
    print("Training completed!")
    print(f"Best model saved at: {best_model_path}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    
    # Create a post-processing wrapper for circle detection
    create_circle_detector_wrapper(best_model_path)
    
    return best_model_path  # Return path for other scripts to use

def create_circle_detector_wrapper(model_path):
    """Create a wrapper class for circle detection"""
    wrapper_code = f'''#!/usr/bin/env python3
"""
Fast Circle Detection Wrapper
Uses the trained model to detect circles in images
"""

from ultralytics import YOLO
import numpy as np

class FastCircleDetector:
    def __init__(self, model_path="{model_path}"):
        self.model = YOLO(model_path)
        print(f"✅ Loaded fast circle detector from {{model_path}}")
    
    def detect_circles(self, image, confidence_threshold=0.3):
        """
        Detect circles in the image
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for circle detections
        
        Returns:
            circle_coordinates: List of circle coordinates [(x1,y1,x2,y2), confidence]
        """
        # Run inference
        results = self.model(image)
        
        if not results[0].boxes:
            return []
        
        # Parse detections
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Filter by confidence threshold
        circle_coordinates = []
        for box, conf in zip(boxes, confidences):
            if conf >= confidence_threshold:
                circle_coordinates.append([
                    box.tolist(),  # [x1, y1, x2, y2]
                    conf           # confidence score
                ])
        
        return circle_coordinates
    
    def get_circle_center(self, bbox):
        """Get center point of circle bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def get_circle_radius(self, bbox):
        """Get approximate radius of circle from bounding box"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return min(width, height) / 2
    
# Usage example:
# detector = FastCircleDetector()
# coordinates = detector.detect_circles(image)
# for bbox, confidence in coordinates:
#     print(f"Circle at {{bbox}} with confidence {{confidence}}")
'''
    
    # Save the wrapper to a file
    wrapper_path = 'src/fast_circle_detector.py'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    
    print(f"✅ Created fast circle detector wrapper: {wrapper_path}")
    print("   Usage: from src.fast_circle_detector import FastCircleDetector")

if __name__ == "__main__":
    import sys
    import argparse
    
    # Verify we're running from the correct directory
    if not (os.path.exists('venv') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: venv/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        print("   Usage: python src/train_model.py [--imgsz SIZE] [--batch SIZE] [--epochs NUM] [--patience NUM]")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 Circle Detection Model')
    parser.add_argument('--imgsz', type=int, default=640, 
                       choices=[640, 832, 1024, 1280],
                       help='Training image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16, auto-adjusted for large images)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=0,
                       help='Early stopping patience (default: 0=disabled, 20=stop after 20 epochs without improvement)')
    parser.add_argument('--square', action='store_true',
                       help='Use square input instead of preserving aspect ratio (default: preserve aspect ratio)')
    
    args = parser.parse_args()
    
    print(f"🎯 === Circle Detection Training ===")
    print(f"Image size: {args.imgsz}px")
    print(f"Aspect ratio: {'Square' if args.square else 'Preserved (rectangular)'}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping: {'Disabled' if args.patience == 0 else f'Patience = {args.patience}'}")
    
    if args.imgsz > 640:
        print(f"⚠️  Using higher resolution ({args.imgsz}px) will:")
        print(f"   • Improve accuracy for small circles")
        print(f"   • Increase training time significantly")
        print(f"   • Require more memory")
        print(f"   • May auto-reduce batch size")
    
    if not args.square:
        print(f"📐 Using rectangular input to preserve aspect ratio")
        print(f"   • Better for images with specific proportions")
        print(f"   • May improve accuracy for circular objects")
        print(f"   • Slightly more memory usage")
    
    model_path = train_circle_detector(
        image_size=args.imgsz,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        preserve_aspect_ratio=not args.square  # Default to True, use --square to disable
    )
    
    print(f"\n🎉 Training complete! Model saved at: {model_path}")
    print(f"📦 Fast circle detector wrapper created: src/fast_circle_detector.py")
