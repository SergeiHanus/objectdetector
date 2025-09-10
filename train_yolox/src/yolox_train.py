#!/usr/bin/env python3
"""
YOLOX Circle Detection Training Script with YOLOX-Small model
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def train_yolox_detector(data_dir=None, epochs=100, batch_size=2, num_workers=1):
    """
    Train YOLOX-Small model for object detection
    
    Args:
        data_dir: Path to dataset directory. If None, defaults to YOLOX/datasets/<class_name>_dataset
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
    """
    
    # Ensure we're in the train_yolox project root
    if not (os.path.exists('yolox') and os.path.exists('YOLOX') and os.path.exists('src')):
        print("❌ Error: This script must be run from the train_yolox project root directory")
        print("   Expected structure: yolox/, YOLOX/, src/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: train_yolox/")
        sys.exit(1)
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is required for YOLOX training")
        print("   Please ensure CUDA is installed and PyTorch supports it")
        sys.exit(1)
    
    # Auto-detect data_dir if not provided
    if data_dir is None:
        # Look for existing dataset directories
        datasets_dir = 'YOLOX/datasets'
        if os.path.exists(datasets_dir):
            dataset_dirs = [d for d in os.listdir(datasets_dir) 
                          if os.path.isdir(os.path.join(datasets_dir, d)) and d.endswith('_dataset')]
            if len(dataset_dirs) == 1:
                data_dir = os.path.join(datasets_dir, dataset_dirs[0])
                print(f"📁 Auto-detected dataset directory: {data_dir}")
            elif len(dataset_dirs) > 1:
                print(f"❌ Error: Multiple dataset directories found: {dataset_dirs}")
                print("   Please specify --data-dir parameter")
                sys.exit(1)
            else:
                print("❌ Error: No dataset directories found in YOLOX/datasets/")
                print("   Please run data preparation first: python src/yolox_data_prep.py")
                sys.exit(1)
        else:
            print("❌ Error: YOLOX/datasets/ directory not found")
            print("   Please run data preparation first: python src/yolox_data_prep.py")
            sys.exit(1)
    
    # Extract class name from data_dir for display
    class_name = os.path.basename(data_dir).replace('_dataset', '') if data_dir.endswith('_dataset') else 'object'
    
    print(f"🎯 === YOLOX {class_name.title()} Detection Training ===")
    print(f"Model: YOLOX-Small")
    print(f"Dataset: {data_dir}")
    print(f"Class: {class_name}")
    print(f"Image size: 640x640 pixels")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: CUDA (GPU)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Verify dataset exists
    if not os.path.exists(f'{data_dir}/train/images') or not os.path.exists(f'{data_dir}/val/images'):
        print(f"❌ Error: Dataset not found at {data_dir}")
        print("   Please run data preparation first: python src/yolox_data_prep.py")
        sys.exit(1)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Install YOLOX if not already installed
    try:
        import yolox
        print("✅ YOLOX already installed")
    except ImportError:
        print("📦 Installing YOLOX (editable) from local repo without build isolation...")
        os.system("PYTHONNOUSERSITE=1 python -m pip install -v --no-build-isolation -e YOLOX")
    
    # Start training
    print("\n🚀 Starting YOLOX training...")
    
    # Determine the configuration file based on class name
    config_filename = f'exps/yolox_s_{class_name}.py'
    if not os.path.exists(config_filename):
        print(f"❌ Error: Configuration file not found: {config_filename}")
        print("   Please run data preparation first: python src/yolox_data_prep.py")
        sys.exit(1)
    
    # Verify dataset configuration first
    print("\n🔍 Verifying dataset configuration...")
    sys.path.append('exps')
    
    # Dynamically import the configuration module
    config_module_name = f'yolox_s_{class_name}'
    try:
        config_module = __import__(config_module_name)
        Exp = config_module.Exp
        exp = Exp()
        # Update the experiment's data_dir
        exp.data_dir = os.path.abspath(data_dir)
        exp.verify_dataset_config()
    except ImportError as e:
        print(f"❌ Error importing configuration module: {e}")
        print(f"   Configuration file: {config_filename}")
        sys.exit(1)
    
    # Training command with data directory environment variable
    train_cmd = f"""
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True YOLOX_DATA_DIR={os.path.abspath(data_dir)} python tools/train.py -f {os.path.abspath(config_filename)} \
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
    copy_trained_model(class_name)
    
    print("🎉 YOLOX training completed!")
    return True

def copy_trained_model(class_name):
    """Copy trained model from YOLOX output to models directory"""
    
    # Look for the trained model in YOLOX output
    yolox_output = f'YOLOX/YOLOX_outputs/yolox_s_{class_name}'
    
    if os.path.exists(yolox_output):
        # Find the best model
        weights_dir = os.path.join(yolox_output, 'checkpoint')
        if os.path.exists(weights_dir):
            # Copy the latest checkpoint
            for file in os.listdir(weights_dir):
                if file.endswith('.pth'):
                    src = os.path.join(weights_dir, file)
                    dst = f'models/yolox_{class_name}_detector_{file}'
                    shutil.copy2(src, dst)
                    print(f"✅ Copied model: {dst}")
                    break
        
        # Copy ONNX export if available
        onnx_file = os.path.join(yolox_output, f'yolox_s_{class_name}.onnx')
        if os.path.exists(onnx_file):
            dst = f'models/yolox_{class_name}_detector.onnx'
            shutil.copy2(onnx_file, dst)
            print(f"✅ Copied ONNX model: {dst}")
            
            # Copy to Android assets (only for circle detection app)
            if class_name == 'circle':
                android_assets = 'circle-detection-app/app/src/main/assets'
                if os.path.exists(android_assets):
                    os.makedirs(android_assets, exist_ok=True)
                    android_dst = os.path.join(android_assets, f'yolox_{class_name}_model.onnx')
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
    parser = argparse.ArgumentParser(description='Train YOLOX Detection Model')
    parser.add_argument('--data-dir', default=None, help='Path to dataset directory. If not specified, auto-detects from YOLOX/datasets/')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=2, help='Batch size per GPU (memory optimized)')
    parser.add_argument('--workers', type=int, default=1, help='Number of data loading workers (memory optimized)')
    parser.add_argument('--setup', action='store_true', help='Setup YOLOX repository and dependencies')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_yolox()
    else:
        train_yolox_detector(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            num_workers=args.workers
        )
