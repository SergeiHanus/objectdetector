#!/usr/bin/env python3
"""
YOLOX Object Detection Test Script
Test a trained YOLOX model on a single image
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

def test_yolox_model(model_path, image_path=None, confidence_threshold=0.3, exp_name=None):
    """
    Test YOLOX model on a single image

    Args:
        model_path: Path to trained YOLOX model
        image_path: Path to single image to test
        confidence_threshold: Minimum confidence for detections
        exp_name: Experiment name (e.g. yolox_s_marker) for exps/<exp_name>.py; default yolox_s_marker
    """
    if exp_name is None:
        exp_name = "yolox_s_marker"
    
    # Ensure we're in the project root (train_yolox or image-detector: YOLOX or yolox, data, exps or src)
    has_yolox = os.path.exists("YOLOX") or os.path.exists("yolox")
    has_data = os.path.exists("data")
    has_exps_or_src = os.path.exists("exps") or os.path.exists("src")
    if not (has_yolox and has_data and has_exps_or_src):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: YOLOX/ (or yolox/), data/, exps/ or src/ at same level")
        print("   Current directory:", os.getcwd())
        sys.exit(1)
    
    print("🎯 === YOLOX Object Detection Testing ===")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        print("   Please train the model first: python src/yolox_train.py")
        sys.exit(1)
    
    # Check if image path is provided
    if image_path is None:
        print("❌ Error: Please provide an image path using --image argument")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Testing image: {image_path}")
    
    # Select the best available device.
    # - CUDA on Linux/Windows
    # - MPS on Apple Silicon (macOS)
    # - CPU fallback
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU backend available, using CPU (slower)")
    
    # Load YOLOX model
    try:
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        from yolox.data.data_augment import ValTransform
        from yolox.utils import fuse_model, get_model_info
        
        print("📦 Loading YOLOX model...")
        
        # Load experiment configuration
        exp_file = f"exps/{exp_name}.py"
        if not os.path.isfile(exp_file):
            print(f"❌ Error: Experiment file not found: {exp_file}")
            sys.exit(1)
        exp = get_exp(exp_file, None)
        # Respect exp defaults unless missing, but keep a sane fallback.
        exp.test_size = getattr(exp, "test_size", (640, 640)) or (640, 640)
        
        # Load model
        model = exp.get_model()
        model.eval()

        # Always load checkpoints onto CPU first, then move model.
        # This avoids brittle map_location behavior on some non-CUDA backends (e.g. MPS).
        #
        # Note: PyTorch 2.6+ defaults `weights_only=True` which can fail for common training
        # checkpoints that store more than plain tensors. Explicitly opt out.
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions don't support `weights_only`.
            ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        
        print(f"✅ Model loaded: {model_path}")
        
    except ImportError as e:
        print(f"❌ Error: Could not import YOLOX modules: {e}")
        print("   Please ensure YOLOX is properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        sys.exit(1)
    
    print(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Test on single image
    results = []

    # Match YOLOX evaluation preprocessing:
    # - letterbox resize + pad with 114 (via preproc/ValTransform)
    # - keep pixel scale in [0..255] float32 (non-legacy path)
    # - keep BGR ordering (cv2) as used by YOLOX preproc
    test_size = tuple(int(x) for x in exp.test_size)
    ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    img_preproc, _ = ValTransform(legacy=False)(img, None, test_size)

    # Convert to tensor (NCHW)
    img_tensor = torch.from_numpy(img_preproc).unsqueeze(0).contiguous().to(device=device, dtype=torch.float32)
    
    # Run inference
    with torch.no_grad():
        try:
            conf_thr = float(confidence_threshold)
            outputs = model(img_tensor)

            predictions = postprocess(
                outputs, int(getattr(exp, "num_classes", 1) or 1), conf_thr, exp.nmsthre, class_agnostic=True
            )

            det = predictions[0]
            if det is None:
                detection_count = 0
                img_result = img.copy()
                print(f"No objects detected (confidence={conf_thr})")
            else:
                det_cpu = det.cpu().numpy()
                detection_count = int(det_cpu.shape[0])
                # Draw on original image coordinates (scale back by ratio).
                img_result = img.copy()
                for pred in det_cpu:
                    x1, y1, x2, y2 = pred[:4]
                    obj_conf = float(pred[4]) if len(pred) > 4 else 0.0
                    cls_conf = float(pred[5]) if len(pred) > 5 else 0.0
                    score = obj_conf * cls_conf if (obj_conf and cls_conf) else max(obj_conf, cls_conf)

                    x1 = int(x1 / ratio)
                    y1 = int(y1 / ratio)
                    x2 = int(x2 / ratio)
                    y2 = int(y2 / ratio)
                    color = (0, 255, 0)
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_result,
                        f"Object: {score:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                print(f"Detected {detection_count} objects (confidence={conf_thr})")

            results.append(
                {
                    "image": os.path.basename(image_path),
                    "detections": detection_count,
                    "confidence_threshold": conf_thr,
                }
            )

            # Save result image
            output_dir = "models/yolox_test_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, img_result)
            print(f"Result saved: {output_path}")
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            sys.exit(1)
    
    # Print summary
    print("\n📊 Test Results Summary:")
    print(f"Image: {results[0]['image']}")
    print(f"Detections: {results[0]['detections']}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Save results to file
    results_file = 'models/yolox_test_results/test_summary.txt'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("YOLOX Object Detection Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Image: {results[0]['image']}\n")
        f.write(f"Detections: {results[0]['detections']}\n")
        f.write(f"Confidence threshold: {results[0]['confidence_threshold']}\n")
    
    print(f"\n✅ Test results saved to: {results_file}")
    print("🎉 YOLOX model testing completed!")
    if results and int(results[0].get("detections", 0)) <= 0:
        sys.exit(2)

def validate_dataset():
    """Validate the prepared dataset"""
    
    print("🔍 Validating YOLOX dataset...")
    
    # Check dataset structure
    required_dirs = [
        'data/train/images',
        'data/train/labels',
        'data/val/images', 
        'data/val/labels',
        'data/test/images',
        'data/test/labels'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        else:
            file_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.txt'))])
            print(f"✅ {dir_path}: {file_count} files")
    
    # Check COCO annotation files
    coco_files = [
        'data/train/labels.json',
        'data/val/labels.json',
        'data/test/labels.json'
    ]
    
    for coco_file in coco_files:
        if os.path.exists(coco_file):
            print(f"✅ {coco_file} exists")
        else:
            print(f"❌ Missing: {coco_file}")
            return False
    
    print("✅ Dataset validation completed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a YOLOX object detection model")
    parser.add_argument(
        "--model",
        type=str,
        default="YOLOX/YOLOX_outputs/yolox_s_marker/best_ckpt.pth",
                       help='Path to trained YOLOX model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file to test')
    parser.add_argument(
        "--exp",
        "--exp-name",
        dest="exp",
        type=str,
        default="yolox_s_marker",
        help="Experiment name for exps/<exp>.py (default: yolox_s_marker)",
    )
    parser.add_argument('--confidence', type=float, default=0.05,
                       help='Confidence threshold for detections (default: 0.05)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the prepared dataset')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset()
    else:
        test_yolox_model(
            model_path=args.model,
            image_path=args.image,
            confidence_threshold=args.confidence,
            exp_name=args.exp,
        )
