from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class CircleModelTester:
    def __init__(self, model_path=None):
        self.model_path = model_path or self.find_best_model()
        self.model = None
        self.load_model()
    
    def find_best_model(self):
        """Find the most recent trained model"""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return 'models/circle_detector_640px_rect/weights/best.pt'  # Default fallback
        
        # Look for circle_detector_*px directories
        model_dirs = []
        for item in os.listdir(models_dir):
            if item.startswith('circle_detector') and os.path.isdir(os.path.join(models_dir, item)):
                model_path = os.path.join(models_dir, item, 'weights', 'best.pt')
                if os.path.exists(model_path):
                    # Get modification time for sorting
                    mtime = os.path.getmtime(model_path)
                    model_dirs.append((model_path, mtime, item))
        
        if model_dirs:
            # Sort by modification time (newest first) and return the newest model
            model_dirs.sort(key=lambda x: x[1], reverse=True)
            newest_model = model_dirs[0]
            print(f"🔍 Found {len(model_dirs)} trained models. Using newest: {newest_model[2]}")
            return newest_model[0]
        
        # Fallback to default path
        return 'models/circle_detector_640px_rect/weights/best.pt'
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            print("Please train the model first: python src/train_model.py")
            return False
        
        print(f"✅ Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        return True
    
    def test_validation_set(self):
        """Test on validation images with detailed metrics"""
        if not self.model:
            return
        
        print("\n🧪 === VALIDATION SET TESTING ===")
        
        # Run validation and get metrics
        metrics = self.model.val(data='config/dataset.yaml')
        
        print(f"📊 Model Performance Metrics:")
        print(f"   • mAP50 (IoU=0.5):     {metrics.box.map50:.3f}")
        print(f"   • mAP50-95:            {metrics.box.map:.3f}")
        print(f"   • Precision:           {metrics.box.mp:.3f}")
        print(f"   • Recall:              {metrics.box.mr:.3f}")
        print(f"   • F1 Score:            {metrics.box.f1:.3f}")
        
        # Interpretation guide
        self.interpret_metrics(metrics.box.map50, metrics.box.mp, metrics.box.mr)
    
    def interpret_metrics(self, map50, precision, recall):
        """Provide interpretation of metrics"""
        print(f"\n📈 Performance Interpretation:")
        
        if map50 >= 0.8:
            print("   🟢 Excellent detection accuracy!")
        elif map50 >= 0.6:
            print("   🟡 Good detection accuracy, some room for improvement")
        elif map50 >= 0.4:
            print("   🟠 Moderate accuracy, consider more training data")
        else:
            print("   🔴 Low accuracy, needs more training data or different approach")
        
        print(f"   • High Precision ({precision:.3f}): {'✅' if precision > 0.7 else '❌'} Few false positives")
        print(f"   • High Recall ({recall:.3f}): {'✅' if recall > 0.7 else '❌'} Finds most circles")
    
    def resize_for_display(self, image, max_width=1200, max_height=800):
        """Resize image for better display while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
    
    def test_sample_images(self, image_dir='data/val/images', max_images=10):
        """Test on sample images with visual output"""
        if not self.model:
            return
        
        print(f"\n🖼️ === VISUAL TESTING ({image_dir}) ===")
        
        if not os.path.exists(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            print("Available test directories:")
            for test_dir in ['data/val/images', 'data/test/images', 'data/verification']:
                if os.path.exists(test_dir):
                    count = len([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"   • {test_dir}: {count} images")
            return
        
        # Get test images
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        print(f"Testing on {min(len(image_files), max_images)} images...")
        
        detection_results = []
        
        for i, img_file in enumerate(image_files[:max_images]):
            img_path = os.path.join(image_dir, img_file)
            
            # Run inference
            results = self.model(img_path)
            
            # Analyze results
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get confidences
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Count circle detections
                circle_detections = len(results[0].boxes)
                total_detections = circle_detections
                
                # Get confidence statistics
                max_confidence = max(confidences) if len(confidences) > 0 else 0.0
                avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
            else:
                circle_detections = 0
                total_detections = 0
                max_confidence = 0.0
                avg_confidence = 0.0
            
            detection_results.append({
                'file': img_file,
                'circle_detections': circle_detections,
                'total_detections': total_detections,
                'max_confidence': max_confidence,
                'avg_confidence': avg_confidence
            })
            
            print(f"   📸 {img_file}: {circle_detections} circles (max conf: {max_confidence:.3f})")
            
            # Display results
            if total_detections > 0:
                annotated = results[0].plot()
                
                # Resize image for display if too large
                display_img = self.resize_for_display(annotated)
                
                window_name = f'Detection {i+1}/{min(len(image_files), max_images)} - {img_file}'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 600)  # Set reasonable window size
                cv2.imshow(window_name, display_img)
                print(f"      Press any key to continue (ESC to skip remaining)...")
                
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow(window_name)
                
                if key == 27:  # ESC key to skip remaining images
                    print("      Skipping remaining images...")
                    break
            else:
                print(f"      ⚠️ No detections")
        
        # Windows are destroyed individually, no cleanup needed here
        
        # Summary statistics
        total_circle_detections = sum(r['circle_detections'] for r in detection_results)
        total_detections = sum(r['total_detections'] for r in detection_results)
        images_with_circles = sum(1 for r in detection_results if r['circle_detections'] > 0)
        images_with_detections = sum(1 for r in detection_results if r['total_detections'] > 0)
        
        # Calculate average confidences
        circle_confidences = [r['max_confidence'] for r in detection_results if r['circle_detections'] > 0]
        avg_circle_conf = np.mean(circle_confidences) if circle_confidences else 0.0
        
        print(f"\n📊 Detection Summary:")
        print(f"   • Images with detections: {images_with_detections}/{len(detection_results)}")
        print(f"   • Images with circles: {images_with_circles}/{len(detection_results)}")
        print(f"   • Total circles detected: {total_circle_detections}")
        print(f"   • Average circle confidence: {avg_circle_conf:.3f}")
    
    def test_realtime(self):
        """Test model with live camera feed"""
        if not self.model:
            return
        
        print(f"\n🎥 === REAL-TIME TESTING ===")
        print("Starting camera... Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Create resizable window
        window_name = 'Real-time Circle Detection - Press Q to quit, S to save'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 700)
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference every 3 frames for better performance
            if frame_count % 3 == 0:
                results = self.model(frame)
                
                # Count circle detections
                if results[0].boxes and len(results[0].boxes) > 0:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    circle_detections = len(results[0].boxes)
                    
                    # Get max confidence
                    max_circle_conf = np.max(confidences) if len(confidences) > 0 else 0.0
                    
                    # Add detection info to frame
                    detection_text = f"Circles: {circle_detections} (Max conf: {max_circle_conf:.3f})"
                    cv2.putText(frame, detection_text, 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    detection_count += circle_detections
                
                # Draw results
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count} | Press Q to quit, S to save", 
                       (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize for display if needed
            display_frame = self.resize_for_display(annotated_frame)
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"detection_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
        print(f"Detection rate: {detection_rate:.1f}% ({detection_count}/{frame_count} frames)")
    
    def test_edge_cases(self):
        """Test challenging scenarios"""
        edge_case_dirs = [
            ('data/edge_cases', 'Custom edge cases'),
            ('data/verification', 'Verification images'),
            ('data/test_new', 'New test images')
        ]
        
        print(f"\n🎯 === EDGE CASE TESTING ===")
        
        for test_dir, description in edge_case_dirs:
            if os.path.exists(test_dir):
                print(f"\nTesting {description} ({test_dir}):")
                self.test_sample_images(test_dir, max_images=5)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n📋 === MODEL VERIFICATION REPORT ===")
        print(f"Model: {self.model_path}")
        
        # Run all tests
        self.test_validation_set()
        self.test_sample_images()
        self.test_edge_cases()

def main():
    import sys
    
    # Verify we're running from the correct directory
    if not (os.path.exists('venv') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: venv/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        print("   Usage: python src/test_model.py [--realtime|--validation|--visual|--edge|--report]")
        sys.exit(1)
    
    tester = CircleModelTester()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == '--realtime':
            tester.test_realtime()
        elif command == '--validation':
            tester.test_validation_set()
        elif command == '--visual':
            image_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/val/images'
            tester.test_sample_images(image_dir)
        elif command == '--edge':
            tester.test_edge_cases()
        elif command == '--report':
            tester.generate_report()
        else:
            print("Usage: python src/test_model.py [--realtime|--validation|--visual|--edge|--report]")
    else:
        # Default: run quick validation test
        tester.test_sample_images()

if __name__ == "__main__":
    main()
