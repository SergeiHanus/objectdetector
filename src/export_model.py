#!/usr/bin/env python3
"""
Export trained model with optimized settings for mobile deployment
"""

from ultralytics import YOLO
import os
import shutil

def export_model_for_mobile(model_path=None):
    """
    Export model with optimized settings for mobile deployment
    
    Args:
        model_path: Path to the trained model (.pt file)
    """
    
    # Find the best model if not specified
    if model_path is None:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            print("❌ No models directory found")
            return
        
        # Look for the most recent circle_detector model
        model_dirs = []
        for item in os.listdir(models_dir):
            if item.startswith('circle_detector') and os.path.isdir(os.path.join(models_dir, item)):
                pt_path = os.path.join(models_dir, item, 'weights', 'best.pt')
                if os.path.exists(pt_path):
                    mtime = os.path.getmtime(pt_path)
                    model_dirs.append((pt_path, mtime, item))
        
        if model_dirs:
            model_dirs.sort(key=lambda x: x[1], reverse=True)
            model_path = model_dirs[0][0]
            print(f"✅ Using model: {model_dirs[0][2]}")
        else:
            print("❌ No trained models found")
            return
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"🔄 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get model info
    print(f"📊 Model info:")
    print(f"   • Input size: {model.overrides.get('imgsz', 'Unknown')}")
    print(f"   • Classes: {model.overrides.get('nc', 'Unknown')}")
    
    # Export directory
    export_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    print(f"\n🚀 Exporting model with optimized settings...")
    
    try:
        # Get the model's actual input size
        model_input_size = model.overrides.get('imgsz', 640)
        print(f"📏 Using model's actual input size: {model_input_size}")
        
        # Export 1: Float16 TFLite (good balance of speed and accuracy)
        print("📱 Exporting Float16 TFLite model...")
        float16_path = model.export(
            format='tflite',
            int8=False,  # Use float16 instead of int8
            optimize=True,
            simplify=True,
            half=True,  # Use float16 precision
            single_cls=True,  # Single class detection
            imgsz=model_input_size
        )
        print(f"✅ Float16 TFLite: {float16_path}")
        
        # Export 2: Int8 TFLite (fastest, smallest)
        print("📱 Exporting Int8 TFLite model...")
        int8_path = model.export(
            format='tflite',
            int8=True,  # Use int8 quantization
            optimize=True,
            simplify=True,
            single_cls=True,  # Single class detection
            imgsz=model_input_size
        )
        print(f"✅ Int8 TFLite: {int8_path}")
        
        # Export 3: ONNX (cross-platform, good performance)
        print("🔄 Exporting ONNX model...")
        onnx_path = model.export(
            format='onnx',
            simplify=True,
            single_cls=True,  # Single class detection
            imgsz=model_input_size
        )
        print(f"✅ ONNX: {onnx_path}")
        
        # Copy the fastest model (int8) to Android assets
        android_assets_dir = 'circle-detection-app/app/src/main/assets'
        os.makedirs(android_assets_dir, exist_ok=True)
        
        # Copy int8 model as the primary model
        android_model_path = os.path.join(android_assets_dir, 'circle_model.tflite')
        shutil.copy2(int8_path, android_model_path)
        print(f"✅ Copied Int8 model to Android assets: {android_model_path}")
        
        # Also copy float16 as backup
        android_model_float16_path = os.path.join(android_assets_dir, 'circle_model_float16.tflite')
        shutil.copy2(float16_path, android_model_float16_path)
        print(f"✅ Copied Float16 model to Android assets: {android_model_float16_path}")
        
        # Copy ONNX model to assets as well
        android_model_onnx_path = os.path.join(android_assets_dir, 'circle_model.onnx')
        shutil.copy2(onnx_path, android_model_onnx_path)
        print(f"✅ Copied ONNX model to Android assets: {android_model_onnx_path}")
        
        # Show file sizes
        print(f"\n📏 Model sizes:")
        if os.path.exists(float16_path):
            size_mb = os.path.getsize(float16_path) / 1024 / 1024
            print(f"   • Float16 TFLite: {size_mb:.1f} MB")
        
        if os.path.exists(int8_path):
            size_mb = os.path.getsize(int8_path) / 1024 / 1024
            print(f"   • Int8 TFLite: {size_mb:.1f} MB")
        
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            print(f"   • ONNX: {size_mb:.1f} MB")
        
        print(f"\n🎯 Recommendations:")
        print(f"   • Use Int8 TFLite for fastest inference (~1-2MB)")
        print(f"   • Use Float16 TFLite for better accuracy (~6MB)")
        print(f"   • Use ONNX for cross-platform deployment (~10-15MB)")
        print(f"   • Update Android app to use the Int8 model for speed")
        print(f"   • Model outputs single-class circle detections")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    import sys
    
    # Verify we're running from the correct directory
    if not (os.path.exists('venv') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: venv/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        print("   Usage: python src/export_model.py [model_path]")
        sys.exit(1)
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    export_model_for_mobile(model_path)

if __name__ == "__main__":
    main() 