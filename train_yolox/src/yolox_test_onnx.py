#!/usr/bin/env python3
"""
YOLOX Circle Detection - ONNX Inference

Run inference with an exported ONNX model on a single image at 640x640.

Example:
  source yolox/bin/activate
  python src/yolox_test_onnx.py \
    --model /data/code/image-detector/train_yolox/models/yolox_circle_detector.onnx \
    --image /data/code/image-detector/train_yolox/YOLOX/datasets/circle_dataset/test/images/20250723_102806.jpg \
    --confidence 0.3 --show
"""

import argparse
import os
import sys
import cv2
import numpy as np
import time


def run_onnx_inference(model_path: str, image_path: str, confidence_threshold: float = 0.3, show: bool = False, warmup: int = 2, repeat: int = 10) -> None:

	# Basic checks
	if not os.path.exists(model_path):
		print(f"❌ Error: ONNX model not found: {model_path}")
		sys.exit(1)
	if not os.path.exists(image_path):
		print(f"❌ Error: Image not found: {image_path}")
		sys.exit(1)

	# Imports that rely on installed deps
	try:
		import onnxruntime as ort
		from yolox.data.data_augment import preproc as preprocess
		from yolox.utils import mkdir, multiclass_nms, demo_postprocess
	except Exception as e:
		print(f"❌ Error: Missing dependency: {e}")
		print("   Ensure you've activated the YOLOX venv and installed requirements.")
		sys.exit(1)

	# Load image
	origin_img = cv2.imread(image_path)
	if origin_img is None:
		print(f"❌ Error: Could not load image {image_path}")
		sys.exit(1)

	print(f"🎯 Running ONNX inference on: {image_path}")
	print(f"   Image size: {origin_img.shape[1]}x{origin_img.shape[0]}")

	# Fixed input shape (matches exported model)
	input_shape = (640, 640)
	img, ratio = preprocess(origin_img, input_shape)

	# Create ONNX Runtime session (CPU by default)
	session = ort.InferenceSession(model_path)
	input_name = session.get_inputs()[0].name

	# Model expects NCHW float32
	ort_inputs = {input_name: img[None, :, :, :]}  # shape: (1, 3, 640, 640)

	# Warmup runs (not timed)
	for _ in range(max(0, warmup)):
		session.run(None, ort_inputs)

	# Timed runs
	times_ms = []
	for _ in range(max(1, repeat)):
		t0 = time.time()
		ort_outs = session.run(None, ort_inputs)
		times_ms.append((time.time() - t0) * 1000.0)

	# Report latency
	avg_ms = float(np.mean(times_ms))
	p50_ms = float(np.median(times_ms))
	p95_ms = float(np.percentile(times_ms, 95)) if len(times_ms) > 1 else avg_ms
	print(f"⏱️ ONNX inference latency: avg={avg_ms:.2f} ms, p50={p50_ms:.2f} ms, p95={p95_ms:.2f} ms (repeat={len(times_ms)}, warmup={max(0, warmup)})")

	# Decode predictions using YOLOX utilities
	predictions = demo_postprocess(ort_outs[0], input_shape)[0]
	boxes = predictions[:, :4]
	scores = predictions[:, 4:5] * predictions[:, 5:]

	# Convert cx,cy,w,h to x1,y1,x2,y2
	boxes_xyxy = np.ones_like(boxes)
	boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
	boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
	boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
	boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
	boxes_xyxy /= ratio  # scale back to original image

	# NMS
	dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=confidence_threshold)

	result_img = origin_img.copy()
	detection_count = 0
	if dets is not None:
		final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
		for (x1, y1, x2, y2), conf, cls_id in zip(final_boxes, final_scores, final_cls_inds):
			conf = float(conf)
			if conf < confidence_threshold:
				continue
			x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
			w, h = x2 - x1, y2 - y1
			cx, cy = x1 + w // 2, y1 + h // 2
			radius = max(1, int(min(w, h) * 0.5))
			cv2.circle(result_img, (cx, cy), radius, (0, 255, 0), 2)
			label = f"Circle: {conf:.2f}"
			cv2.putText(result_img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			detection_count += 1

	print(f"Detected {detection_count} circles (conf >= {confidence_threshold})")

	# Save output
	output_dir = 'models/yolox_test_results'
	mkdir(output_dir)
	output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
	cv2.imwrite(output_path, result_img)
	print(f"✅ Result saved: {output_path}")

	# Optionally display
	if show:
		try:
			cv2.imshow("YOLOX ONNX Detections", result_img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		except Exception:
			pass

	# Save brief summary
	results_file = 'models/yolox_test_results/test_summary.txt'
	with open(results_file, 'a') as f:
		f.write("YOLOX ONNX Inference Results\n")
		f.write("=" * 40 + "\n")
		f.write(f"Model: {model_path}\n")
		f.write(f"Image: {os.path.basename(image_path)}\n")
		f.write(f"Detections: {detection_count}\n")
		f.write(f"Confidence threshold: {confidence_threshold}\n\n")
	print(f"📄 Summary appended to: {results_file}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Test YOLOX ONNX Model (Circle Detection)')
	parser.add_argument('--model', type=str, required=True, help='Path to ONNX model (.onnx)')
	parser.add_argument('--image', type=str, required=True, help='Path to image file to test')
	parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold for detections')
	parser.add_argument('--show', action='store_true', help='Display the image with detections in a window')
	parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs (not timed)')
	parser.add_argument('--repeat', type=int, default=10, help='Number of timed runs for latency stats')
	args = parser.parse_args()

	run_onnx_inference(
		model_path=args.model,
		image_path=args.image,
		confidence_threshold=args.confidence,
		show=args.show,
		warmup=args.warmup,
		repeat=args.repeat,
	)


