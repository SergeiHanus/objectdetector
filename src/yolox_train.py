#!/usr/bin/env python3
"""
YOLOX Circle Detection Training Script
Replaces the YOLOv8 training with YOLOX-Small model
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def train_yolox_circle_detector(epochs=100, batch_size=2, num_workers=1):
    """
    Train YOLOX-Small model for circle detection
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
    """
    
    # Ensure we're in the project root
    if not (os.path.exists('yolox') and os.path.exists('YOLOX') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: yolox/, YOLOX/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        sys.exit(1)
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is required for YOLOX training")
        print("   Please ensure CUDA is installed and PyTorch supports it")
        sys.exit(1)
    
    print("🎯 === YOLOX Circle Detection Training ===")
    print(f"Model: YOLOX-Small")
    print(f"Image size: 640x640 pixels")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: CUDA (GPU)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Verify dataset exists
    if not os.path.exists('YOLOX/datasets/circle_dataset/train/images') or not os.path.exists('YOLOX/datasets/circle_dataset/val/images'):
        print("❌ Error: Dataset not found. Please run data preparation first:")
        print("   python src/yolox_data_prep.py")
        sys.exit(1)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Install YOLOX if not already installed
    try:
        import yolox
        print("✅ YOLOX already installed")
    except ImportError:
        print("📦 Installing YOLOX...")
        os.system("pip install -e .")
    
    # Start training
    print("\n🚀 Starting YOLOX training...")
    
    # Verify dataset configuration first
    print("\n🔍 Verifying dataset configuration...")
    import sys
    sys.path.append('exps')
    from yolox_s_circle import Exp
    exp = Exp()
    exp.verify_dataset_config()
    
    # Training command
    train_cmd = f"""
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/train.py -f {os.path.abspath('exps/yolox_s_circle.py')} \
        -d 1 -b {batch_size} \
        --fp16
    """
    
    print(f"Training command: {train_cmd}")
    
    # Change to YOLOX directory and run training
    if os.path.exists('YOLOX'):
        os.chdir('YOLOX')
        os.system(train_cmd)
        os.chdir('..')
    else:
        print("❌ Error: YOLOX directory not found")
        print("   Please clone YOLOX repository first:")
        print("   git clone https://github.com/Megvii-BaseDetection/YOLOX.git")
        sys.exit(1)
    
    # Copy trained model to models directory
    copy_trained_model()
    
    print("🎉 YOLOX training completed!")
    return True

def copy_trained_model():
    """Copy trained model from YOLOX output to models directory"""
    
    # Look for the trained model in YOLOX output
    yolox_output = 'YOLOX/YOLOX_outputs/yolox_s_circle'
    
    if os.path.exists(yolox_output):
        # Find the best model
        weights_dir = os.path.join(yolox_output, 'checkpoint')
        if os.path.exists(weights_dir):
            # Copy the latest checkpoint
            for file in os.listdir(weights_dir):
                if file.endswith('.pth'):
                    src = os.path.join(weights_dir, file)
                    dst = f'models/yolox_circle_detector_{file}'
                    shutil.copy2(src, dst)
                    print(f"✅ Copied model: {dst}")
                    break
        
        # Copy ONNX export if available
        onnx_file = os.path.join(yolox_output, 'yolox_s_circle.onnx')
        if os.path.exists(onnx_file):
            dst = 'models/yolox_circle_detector.onnx'
            shutil.copy2(onnx_file, dst)
            print(f"✅ Copied ONNX model: {dst}")
            
            # Copy to Android assets
            android_assets = 'circle-detection-app/app/src/main/assets'
            if os.path.exists(android_assets):
                os.makedirs(android_assets, exist_ok=True)
                android_dst = os.path.join(android_assets, 'yolox_circle_model.onnx')
                shutil.copy2(onnx_file, android_dst)
                print(f"✅ Copied to Android assets: {android_dst}")

def setup_yolox():
    """Setup YOLOX repository and dependencies"""
    
    if not os.path.exists('YOLOX'):
        print("📦 Setting up YOLOX...")
        os.system("git clone https://github.com/Megvii-BaseDetection/YOLOX.git")
        os.chdir('YOLOX')
        os.system("pip install -v -e .")
        os.chdir('..')
        print("✅ YOLOX setup completed")
    else:
        print("✅ YOLOX repository already exists")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOX Circle Detection Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=2, help='Batch size per GPU (memory optimized)')
    parser.add_argument('--workers', type=int, default=1, help='Number of data loading workers (memory optimized)')
    parser.add_argument('--setup', action='store_true', help='Setup YOLOX repository and dependencies')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_yolox()
    else:
        train_yolox_circle_detector(
            epochs=args.epochs,
            batch_size=args.batch,
            num_workers=args.workers
        )
