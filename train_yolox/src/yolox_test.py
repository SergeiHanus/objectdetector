#!/usr/bin/env python3
"""
YOLOX Circle Detection Test Script
Test the trained YOLOX model on validation images
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

def test_yolox_model(model_path, image_path=None, confidence_threshold=0.3, show=False):
    """
    Test YOLOX model on a single image
    
    Args:
        model_path: Path to trained YOLOX model
        image_path: Path to single image to test
        confidence_threshold: Minimum confidence for detections
    """
    
    # Ensure we're in the project root
    if not (os.path.exists('yolox') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: yolox/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: train_yolox/")
        sys.exit(1)
    
    print("🎯 === YOLOX Circle Detection Testing ===")
    
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
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, using CPU (slower)")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load YOLOX model
    try:
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        from yolox.data.data_augment import ValTransform
        from yolox.utils import fuse_model, get_model_info
        
        print("📦 Loading YOLOX model...")
        
        # Load experiment configuration
        exp = get_exp('exps/yolox_s_circle.py', None)
        exp.test_conf = confidence_threshold
        exp.test_size = (640, 640)
        
        # Load model
        model = exp.get_model()
        model.eval()
        
        if device == 'cuda':
            model.cuda()
        
        # Load weights (allow full checkpoint for trusted local file)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        
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
    
    # Preprocess using YOLOX ValTransform (letterbox to 640, fill=114, no /255)
    preproc = ValTransform(legacy=False)
    img_transformed, _ = preproc(img, None, exp.test_size)
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(device).float()
    
    # Run inference
    with torch.no_grad():
        try:
            outputs = model(img_tensor)
            
            # Post-process outputs
            predictions = postprocess(
                outputs, 1, exp.test_conf, exp.nmsthre, class_agnostic=True
            )
            
            if predictions[0] is not None:
                predictions = predictions[0].cpu().numpy()
                
                # Draw detections on original image size
                img_result = img.copy()
                detection_count = 0
                
                for pred in predictions:
                    x1, y1, x2, y2, obj_conf, cls_conf, cls_id = pred
                    conf = float(obj_conf) * float(cls_conf)
                    cls = int(cls_id)
                    
                    if conf >= confidence_threshold:
                        # Scale boxes back from letterboxed scale to original image
                        ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
                        x1, y1, x2, y2 = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(y2 / ratio)
                        
                        # Draw green circle for detections
                        color = (0, 255, 0)  # Green color
                        w, h = x2 - x1, y2 - y1
                        cx, cy = x1 + w // 2, y1 + h // 2
                        radius = max(1, int(min(w, h) * 0.5))
                        cv2.circle(img_result, (cx, cy), radius, color, 2)
                        
                        # Draw confidence score near top-left of bbox
                        label = f"Circle: {conf:.2f}"
                        cv2.putText(img_result, label, (x1, max(0, y1-10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        detection_count += 1
                
                print(f"Detected {detection_count} circles")
                results.append({
                    'image': os.path.basename(image_path),
                    'detections': detection_count,
                    'confidence_threshold': confidence_threshold
                })
                
                # Save result image
                output_dir = 'models/yolox_test_results'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, img_result)
                print(f"Result saved: {output_path}")

                # Optionally show result image
                if show:
                    try:
                        cv2.imshow("YOLOX Circle Detections", img_result)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    except Exception as _:
                        pass
                
            else:
                print("No circles detected")
                # Show original image if requested
                if show:
                    try:
                        cv2.imshow("YOLOX Circle Detections", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    except Exception as _:
                        pass

                results.append({
                    'image': os.path.basename(image_path),
                    'detections': 0,
                    'confidence_threshold': confidence_threshold
                })
                
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
        f.write("YOLOX Circle Detection Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Image: {results[0]['image']}\n")
        f.write(f"Detections: {results[0]['detections']}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
    
    print(f"\n✅ Test results saved to: {results_file}")
    print("🎉 YOLOX model testing completed!")

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
    parser = argparse.ArgumentParser(description='Test YOLOX Circle Detection Model')
    parser.add_argument('--model', type=str, default='YOLOX_outputs/yolox_s_circle/best_ckpt.pth',
                       help='Path to trained YOLOX model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file to test')
    parser.add_argument('--confidence', type=float, default=0.05,
                       help='Confidence threshold for detections (default: 0.05)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the prepared dataset')
    parser.add_argument('--show', action='store_true',
                        help='Display the image with detections in a window')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset()
    else:
        test_yolox_model(
            model_path=args.model,
            image_path=args.image,
            confidence_threshold=args.confidence,
            show=args.show
        )
