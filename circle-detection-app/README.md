# Circle Detection Android App

This Android application detects circles in images using a trained YOLOv8 model. The app can detect multiple circles in a single image and displays them with confidence scores.

## Features

- **Circle Detection**: Detects multiple circles in images
- **Real-time Processing**: Fast inference using TensorFlow Lite
- **Confidence Display**: Shows confidence scores for each detected circle
- **Visual Overlay**: Draws circles around detected objects with color-coded confidence levels
- **Multiple Circles**: Can detect and display multiple circles simultaneously

## Model Information

The app uses a YOLOv8 model trained specifically for circle detection:
- **Input Size**: 640x640 pixels
- **Model Format**: TensorFlow Lite (optimized for mobile)
- **Detection Type**: Single class (circle)
- **Confidence Threshold**: 0.3 (30%)

## Color Coding

The detected circles are color-coded based on confidence levels:
- **Green**: High confidence (>70%)
- **Yellow**: Medium confidence (50-70%)
- **Red**: Low confidence (<50%)

## Setup

1. **Build the App**:
   ```bash
   cd circle-detection-app
   ./gradlew assembleDebug
   ```

2. **Install on Device**:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

## Usage

1. **Launch the App**: Open "Circle Detection" from your app drawer
2. **Select Image**: Tap "📁 Select Image" to choose an image from your gallery
3. **Detect Circles**: Tap "🔍 Detect Circles" to run the detection
4. **View Results**: The app will display detected circles with confidence scores

## Model Training

To train your own circle detection model:

1. **Prepare Dataset**: Organize your circle images and annotations
2. **Train Model**: Run the training script:
   ```bash
   python src/train_model.py --imgsz 640 --epochs 100
   ```
3. **Export Model**: Export the trained model for mobile:
   ```bash
   python src/export_circle_model.py
   ```

## Technical Details

### Model Output Format

The model outputs detections in the following format:
```python
detections = [
    {
        'bbox': [x1, y1, x2, y2],  # Normalized coordinates (0-1)
        'confidence': 0.92,
        'class': 0  # Circle class
    },
    # ... more detections
]
```

### Android Implementation

- **Package**: `com.example.circledetection`
- **Main Activity**: `MainActivity.java`
- **Overlay View**: `DetectionOverlayView.java` (draws circles)
- **Model File**: `circle_model.tflite` in assets

### Performance

- **Inference Time**: ~100-500ms depending on device
- **Model Size**: ~3MB (Int8 quantized)
- **Memory Usage**: Optimized for mobile devices

## Troubleshooting

### Common Issues

1. **No Circles Detected**:
   - Check if the image contains clear circular objects
   - Try adjusting lighting or image quality
   - Ensure the circles are not too small or too large

2. **Slow Performance**:
   - The app uses Int8 quantization for speed
   - Consider using a more powerful device
   - Reduce image resolution if needed

3. **Model Loading Issues**:
   - Ensure `circle_model.tflite` is in the assets folder
   - Check that the model file is not corrupted

### Debug Information

The app provides detailed logging for debugging:
- Model loading status
- Detection confidence scores
- Processing time
- Coordinate transformations

## Development

### Project Structure

```
circle-detection-app/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/circledetection/
│   │   │   ├── MainActivity.java
│   │   │   └── DetectionOverlayView.java
│   │   ├── res/
│   │   │   ├── layout/activity_main.xml
│   │   │   ├── values/strings.xml
│   │   │   └── values/colors.xml
│   │   └── assets/
│   │       ├── circle_model.tflite
│   │       └── circle_model_float16.tflite
│   └── build.gradle
└── README.md
```

### Key Components

- **MainActivity**: Handles image selection, model inference, and result display
- **DetectionOverlayView**: Custom view for drawing circles and confidence scores
- **TensorFlow Lite**: Used for model inference on Android
- **Image Processing**: Handles image resizing, normalization, and coordinate mapping

## License

This project is part of the image-detector system for circle detection using computer vision and machine learning techniques. 