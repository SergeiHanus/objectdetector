package com.example.circledetection;

import android.Manifest;
import android.os.Build;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import androidx.exifinterface.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "CircleDetection";
    private static final String MODEL_FILENAME = "circle_model.tflite";
    private static final int INPUT_SIZE = 640; // Model input size (square)
    private static final float CONFIDENCE_THRESHOLD = 0.3f; // Adjusted for new model

    private Interpreter tflite;
    private ImageView imageView;
    private DetectionOverlayView overlayView;
    private DetectionOverlayView imageOverlayView;
    private TextView resultsText;
    private Button selectImageButton;
    private Button processButton;
    private Button cameraModeButton;
    private Button imageModeButton;
    private Button startCameraButton;
    private Button stopCameraButton;
    private LinearLayout imageModeControls;
    private LinearLayout cameraModeControls;
    private FrameLayout cameraPreviewContainer;
    private PreviewView cameraPreview;
    private Bitmap selectedImage;
    private Uri selectedImageUri;
    private int originalImageWidth, originalImageHeight;
    
    // Camera-related fields
    private ProcessCameraProvider cameraProvider;
    private ExecutorService cameraExecutor;
    private boolean isCameraActive = false;
    private boolean isCameraMode = true; // Default to camera mode

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    if (isCameraMode) {
                        startCamera();
                    } else {
                        openImagePicker();
                    }
                } else {
                    String message = isCameraMode ? 
                        getString(R.string.camera_permission_required) : 
                        "Permission required to select images. Please grant permission in Settings.";
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show();
                }
            });
            
    private final ActivityResultLauncher<String> requestCameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show();
                }
            });

    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    if (imageUri != null) {
                        loadImageFromUri(imageUri);
                    }
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Log.d(TAG, "=== APP STARTUP ===");
        Log.d(TAG, "MainActivity onCreate() called");
        
        // Add a simple test log that should definitely appear
        // Log.i(TAG, "🔍 TEST LOG - If you see this, logging is working!");
        // System.out.println("🔍 SYSTEM.OUT TEST - CircleDetection app started!");
        
        initializeViews();
        // loadTensorFlowModel();
        initializeCpuOptimized();   
        
        Log.d(TAG, "=== APP STARTUP COMPLETE ===");
    }

    private void initializeViews() {
        Log.d(TAG, "Initializing views...");
        
        imageView = findViewById(R.id.imageView);
        overlayView = findViewById(R.id.overlayView);
        imageOverlayView = findViewById(R.id.imageOverlayView);
        resultsText = findViewById(R.id.resultsText);
        selectImageButton = findViewById(R.id.selectImageButton);
        processButton = findViewById(R.id.processButton);
        cameraModeButton = findViewById(R.id.cameraModeButton);
        imageModeButton = findViewById(R.id.imageModeButton);
        startCameraButton = findViewById(R.id.startCameraButton);
        stopCameraButton = findViewById(R.id.stopCameraButton);
        imageModeControls = findViewById(R.id.imageModeControls);
        cameraModeControls = findViewById(R.id.cameraModeControls);
        cameraPreviewContainer = findViewById(R.id.cameraPreviewContainer);
        cameraPreview = findViewById(R.id.cameraPreview);

        // Set up mode switching
        cameraModeButton.setOnClickListener(v -> switchToCameraMode());
        imageModeButton.setOnClickListener(v -> switchToImageMode());
        
        // Set up camera controls
        startCameraButton.setOnClickListener(v -> checkCameraPermissionAndStart());
        stopCameraButton.setOnClickListener(v -> stopCamera());
        
        // Set up image mode controls
        selectImageButton.setOnClickListener(v -> checkPermissionAndSelectImage());
        processButton.setOnClickListener(v -> processImage());
        processButton.setEnabled(false);
        
        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor();
        
        // Set initial mode (camera mode by default)
        switchToCameraMode();
        
        Log.d(TAG, "Views initialized successfully");
        
        // Test overlay drawing
        // overlayView.setOnClickListener(v -> {
        //     Log.d(TAG, "Testing overlay drawing at center of screen");
        //     overlayView.setDetection(400, 400, 600, 600, 0.95f, true);
        // });

        // Set initial button text
        processButton.setText(R.string.process_image);
        selectImageButton.setText(R.string.select_image);
    }

private void initializeCpuOptimized() {
    Log.d(TAG, "=== MODEL INITIALIZATION START ===");
    try {
        Log.d(TAG, "Setting up CPU-optimized interpreter options...");
        Interpreter.Options options = new Interpreter.Options();
        
        // Optimizations for the new Int8 model (much faster than float16)
        options.setNumThreads(8); // Use all 8 cores
        options.setAllowFp16PrecisionForFp32(true); // Allow FP16 for speed
        options.setUseNNAPI(false); // Explicit CPU-only to avoid overhead
        
        // Use XNNPACK explicitly for optimized CPU inference
        options.setUseXNNPACK(true);
        
        // Additional optimizations for speed
        options.setAllowBufferHandleOutput(true); // Allow direct buffer access
        options.setCancellable(false); // Disable cancellation for speed
        
        Log.d(TAG, "Loading model file: " + MODEL_FILENAME);
        // Load and verify model file
        ByteBuffer modelBuffer = loadModelFile();
        Log.d(TAG, "Model file size: " + modelBuffer.capacity() + " bytes");
        
        // Check if this is the new Int8 model (should be ~3-4MB)
        if (modelBuffer.capacity() > 3_000_000 && modelBuffer.capacity() < 5_000_000) {
            Log.d(TAG, "✅ Detected new Int8 model (size: " + modelBuffer.capacity() + " bytes) - FASTEST");
        } else if (modelBuffer.capacity() > 5_000_000 && modelBuffer.capacity() < 7_000_000) {
            Log.d(TAG, "✅ Detected Float16 model (size: " + modelBuffer.capacity() + " bytes) - MEDIUM SPEED");
        } else {
            Log.w(TAG, "⚠️ Warning: Model size unexpected (" + (modelBuffer.capacity() / 1024 / 1024) + " MB) - expected ~3-4MB for Int8 or ~6MB for Float16");
        }
        
        tflite = new Interpreter(modelBuffer, options);
        
        // Log model input/output details
        Log.d(TAG, "Model input tensor count: " + tflite.getInputTensorCount());
        Log.d(TAG, "Model output tensor count: " + tflite.getOutputTensorCount());
        
        for (int i = 0; i < tflite.getInputTensorCount(); i++) {
            int[] inputShape = tflite.getInputTensor(i).shape();
            Log.d(TAG, "Input tensor " + i + " shape: " + java.util.Arrays.toString(inputShape));
        }
        
        for (int i = 0; i < tflite.getOutputTensorCount(); i++) {
            int[] outputShape = tflite.getOutputTensor(i).shape();
            Log.d(TAG, "Output tensor " + i + " shape: " + java.util.Arrays.toString(outputShape));
        }
        
        Log.d(TAG, "CPU-optimized interpreter initialized with XNNPACK for Int8 model");
        
        // Test the model with a dummy input to verify it works
        try {
            Log.d(TAG, "Testing model with dummy input...");
            
            // Get the actual output shape from the model
            int[] outputShape = tflite.getOutputTensor(0).shape();
            Log.d(TAG, "Model output shape: [" + outputShape[0] + "][" + outputShape[1] + "][" + outputShape[2] + "]");
            
            float[][][] dummyOutput = new float[outputShape[0]][outputShape[1]][outputShape[2]];
            ByteBuffer dummyInput = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
            dummyInput.rewind();
            
            long testStart = System.currentTimeMillis();
            tflite.run(dummyInput, dummyOutput);
            long testTime = System.currentTimeMillis() - testStart;
            
            Log.d(TAG, "✅ Model test successful - inference time: " + testTime + "ms");
            resultsText.setText("New Int8 model loaded successfully\nSize: " + (modelBuffer.capacity() / 1024 / 1024) + "MB\nTest inference: " + testTime + "ms\nCamera mode active - tap 'Start Camera' to begin detection");
        } catch (Exception e) {
            Log.e(TAG, "Model test failed", e);
            resultsText.setText("⚠️ Model loaded but test failed\nSize: " + (modelBuffer.capacity() / 1024 / 1024) + "MB\nError: " + e.getMessage());
        }
    } catch (Exception e) {
        Log.e(TAG, "Error loading TensorFlow Lite model", e);
        resultsText.setText("❌ Error loading model: " + e.getMessage());
        Toast.makeText(this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
    }
}

    private void loadTensorFlowModel() {
        try {
        NnApiDelegate nnApiDelegate = new NnApiDelegate();
        Interpreter.Options options = new Interpreter.Options();
        
        options.addDelegate(nnApiDelegate);
        // Add this line to explicitly try and use NNAPI
        options.setUseNNAPI(true); 
        
        options.setNumThreads(4); // Keep threads for potential CPU fallback for unsupported ops
        
        tflite = new Interpreter(loadModelFile(), options); //
        Log.d(TAG, "TensorFlow Lite model loaded successfully with NNAPI attempt"); //
        resultsText.setText("✅ Model loaded successfully\nSelect an image to detect arrows"); //
        } catch (Exception e) {
            Log.e(TAG, "Error loading TensorFlow Lite model", e);
            resultsText.setText("❌ Error loading model: " + e.getMessage());
            Toast.makeText(this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private ByteBuffer loadModelFile() throws IOException {
        InputStream inputStream = getAssets().open(MODEL_FILENAME);
        byte[] modelBytes = new byte[inputStream.available()];
        inputStream.read(modelBytes);
        inputStream.close();

        // Log model file details
        Log.d(TAG, "Loading model file: " + MODEL_FILENAME);
        Log.d(TAG, "Model file size: " + modelBytes.length + " bytes (" + (modelBytes.length / 1024 / 1024) + " MB)");
        
        // Check if this is the new Int8 model
        if (modelBytes.length > 3_000_000 && modelBytes.length < 5_000_000) {
            Log.d(TAG, "✅ Confirmed: New Int8 model loaded (" + (modelBytes.length / 1024 / 1024) + " MB) - FASTEST");
        } else if (modelBytes.length > 5_000_000 && modelBytes.length < 7_000_000) {
            Log.d(TAG, "✅ Confirmed: Float16 model loaded (" + (modelBytes.length / 1024 / 1024) + " MB) - MEDIUM SPEED");
        } else {
            Log.w(TAG, "⚠️ Warning: Model size unexpected (" + (modelBytes.length / 1024 / 1024) + " MB) - expected ~3-4MB for Int8 or ~6MB for Float16");
        }

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelBytes.length);
        byteBuffer.put(modelBytes);
        byteBuffer.rewind();

        return byteBuffer;
    }

    private void switchToImageMode() {
        isCameraMode = false;
        imageModeButton.setBackgroundTintList(ContextCompat.getColorStateList(this, R.color.primary_color));
        cameraModeButton.setBackgroundTintList(ContextCompat.getColorStateList(this, R.color.secondary_color));
        
        imageModeControls.setVisibility(View.VISIBLE);
        cameraModeControls.setVisibility(View.GONE);
        imageView.setVisibility(View.VISIBLE);
        cameraPreviewContainer.setVisibility(View.GONE);
        
        // Stop camera if active
        if (isCameraActive) {
            stopCamera();
        }
        
        resultsText.setText("Image Mode\nSelect an image to detect circles");
        
        Log.d(TAG, "Switched to image mode");
    }
    
    private void checkCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }
    
    private void startCamera() {
        if (isCameraActive) {
            Log.d(TAG, "Camera already active");
            return;
        }
        
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
                isCameraActive = true;
                startCameraButton.setEnabled(false);
                stopCameraButton.setEnabled(true);
                resultsText.setText(getString(R.string.camera_started));
                Log.d(TAG, "Camera started successfully");
                Log.d(TAG, "Camera preview container visibility after start: " + cameraPreviewContainer.getVisibility());
                Log.d(TAG, "Camera preview visibility after start: " + cameraPreview.getVisibility());
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera", e);
                Toast.makeText(this, "Error starting camera: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void stopCamera() {
        if (!isCameraActive) {
            return;
        }
        
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        isCameraActive = false;
        startCameraButton.setEnabled(true);
        stopCameraButton.setEnabled(false);
        resultsText.setText(getString(R.string.camera_stopped));
        Log.d(TAG, "Camera stopped");
    }

private void bindCameraUseCases() {
    if (cameraProvider == null) {
        return;
    }
    
    // CRITICAL: Use identical resolution and aspect ratio for both preview and analysis
    android.util.Size targetResolution = new android.util.Size(640, 640);
    
    // Preview use case - EXACTLY match analysis settings
    Preview preview = new Preview.Builder()
            .setTargetResolution(targetResolution)
            .setTargetRotation(Surface.ROTATION_0) // Force same rotation
            .build();
    preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());
    
    // Image analysis use case - EXACTLY match preview settings  
    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
            .setTargetResolution(targetResolution)
            .setTargetRotation(Surface.ROTATION_0) // Force same rotation
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build();
    
    imageAnalysis.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
        @Override
        public void analyze(ImageProxy image) {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastAnalysisTime > 500) {
                lastAnalysisTime = currentTime;
                
                // CRITICAL: Log the actual dimensions we receive
                Log.d(TAG, "ImageAnalysis received: " + image.getWidth() + "x" + image.getHeight() + 
                      " rotation: " + image.getImageInfo().getRotationDegrees());
                
                processCameraFrame(image);
            }
            image.close();
        }
    });
    
    CameraSelector cameraSelector = new CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build();
    
    try {
        cameraProvider.unbindAll();
        
        // Bind both use cases together to ensure they get the same stream
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
        
        Log.d(TAG, "Camera bound successfully with synchronized streams");
        
        // Debug: Check actual preview view setup after binding
        cameraPreview.post(() -> {
            Log.d(TAG, "Final preview setup:");
            Log.d(TAG, "  PreviewView size: " + cameraPreview.getWidth() + "x" + cameraPreview.getHeight());
            Log.d(TAG, "  PreviewView scale type: " + cameraPreview.getScaleType());
            
            // Force the preview to use the same scaling as analysis
            cameraPreview.setScaleType(PreviewView.ScaleType.FILL_CENTER);
            
            // debugCoordinateMapping();
        });
        
    } catch (Exception e) {
        Log.e(TAG, "Error binding camera use cases", e);
    }
}

// Update your processCameraFrame method to handle orientation correctly:

private void processCameraFrame(ImageProxy image) {
    try {
        Log.d(TAG, "Processing camera frame: " + image.getWidth() + "x" + image.getHeight() + 
              " rotation: " + image.getImageInfo().getRotationDegrees());
        
        // Convert ImageProxy to Bitmap
        Bitmap bitmap = imageProxyToBitmap(image);
        if (bitmap == null) {
            Log.w(TAG, "Failed to convert camera frame to bitmap");
            return;
        }
        
        Log.d(TAG, "Converted to bitmap: " + bitmap.getWidth() + "x" + bitmap.getHeight());
        
        // Handle rotation - but ONLY if the analysis and preview streams are rotated differently
        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        Bitmap processedBitmap = bitmap;
        
        // For back camera, we typically need to rotate for correct orientation
        if (rotationDegrees != 0) {
            Log.d(TAG, "Applying rotation: " + rotationDegrees + " degrees");
            processedBitmap = rotateBitmap(bitmap, rotationDegrees);
        }
        
        // CRITICAL: Ensure the bitmap is exactly 640x640 before inference
        Bitmap modelInputBitmap;
        if (processedBitmap.getWidth() != 640 || processedBitmap.getHeight() != 640) {
            Log.d(TAG, "Resizing " + processedBitmap.getWidth() + "x" + processedBitmap.getHeight() + " to 640x640");
            modelInputBitmap = Bitmap.createScaledBitmap(processedBitmap, 640, 640, true);
        } else {
            Log.d(TAG, "✅ Bitmap already 640x640, no resize needed");
            modelInputBitmap = processedBitmap;
        }
        
        // Save a test image to verify what the model sees (debugging only)
        // if (System.currentTimeMillis() % 10000 < 1000) { // Every 10 seconds
        //     saveDebugImage(modelInputBitmap, "model_input_debug.jpg");
        // }
        
        // Run detection on background thread
        new Thread(() -> {
            try {
                DetectionResult result = runInference(modelInputBitmap, null);
                runOnUiThread(() -> displayCameraResults(result));
            } catch (Exception e) {
                Log.e(TAG, "Error during camera frame inference", e);
            }
        }).start();
        
    } catch (Exception e) {
        Log.e(TAG, "Error processing camera frame", e);
    }
}

// Add this debug method to save what the model actually sees:

// private void saveDebugImage(Bitmap bitmap, String filename) {
//     try {
//         // Save to app's internal storage for debugging
//         java.io.File debugDir = new java.io.File(getFilesDir(), "debug");
//         if (!debugDir.exists()) {
//             debugDir.mkdirs();
//         }
//         
//         java.io.File debugFile = new java.io.File(debugDir, filename);
//         java.io.FileOutputStream fos = new java.io.FileOutputStream(debugFile);
//         bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
//         fos.close();
//         
//         Log.d(TAG, "Debug image saved to: " + debugFile.getAbsolutePath());
//         runOnUiThread(() -> {
//             Toast.makeText(this, "Debug image saved: " + filename, Toast.LENGTH_SHORT).show();
//         });
//         
//     } catch (Exception e) {
//         Log.e(TAG, "Failed to save debug image", e);
//     }
// }

// Update your initialization to force preview scale type:

private void switchToCameraMode() {
    isCameraMode = true;
    cameraModeButton.setBackgroundTintList(ContextCompat.getColorStateList(this, R.color.primary_color));
    imageModeButton.setBackgroundTintList(ContextCompat.getColorStateList(this, R.color.secondary_color));
    
    cameraModeControls.setVisibility(View.VISIBLE);
    imageModeControls.setVisibility(View.GONE);
    cameraPreviewContainer.setVisibility(View.VISIBLE);
    imageView.setVisibility(View.GONE);
    
    // CRITICAL: Set preview scale type to match how we process the analysis stream
    cameraPreview.setScaleType(PreviewView.ScaleType.FILL_CENTER);
    
    resultsText.setText("Camera Mode\nTap 'Start Camera' to begin detection");
    
    Log.d(TAG, "Switched to camera mode with FILL_CENTER scaling");
}

    private long lastAnalysisTime = 0;
    
    private void debugCoordinateMapping() {
        Log.d(TAG, "=== COORDINATE MAPPING DEBUG ===");
        Log.d(TAG, "PreviewView size: " + cameraPreview.getWidth() + "x" + cameraPreview.getHeight());
        Log.d(TAG, "PreviewView scale type: " + cameraPreview.getScaleType());
        Log.d(TAG, "OverlayView size: " + overlayView.getWidth() + "x" + overlayView.getHeight());
        Log.d(TAG, "Camera preview container size: " + cameraPreviewContainer.getWidth() + "x" + cameraPreviewContainer.getHeight());
        Log.d(TAG, "=== END COORDINATE MAPPING DEBUG ===");
    }
    
    private List<Detection> applyNonMaxSuppression(List<Detection> detections, float iouThreshold) {
        List<Detection> result = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];
        
        // Sort by confidence (highest first)
        List<Detection> sortedDetections = new ArrayList<>(detections);
        sortedDetections.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        
        for (int i = 0; i < sortedDetections.size(); i++) {
            if (suppressed[i]) continue;
            
            Detection current = sortedDetections.get(i);
            result.add(current);
            
            // Suppress overlapping detections
            for (int j = i + 1; j < sortedDetections.size(); j++) {
                if (suppressed[j]) continue;
                
                Detection other = sortedDetections.get(j);
                float iou = calculateIoU(current, other);
                
                if (iou > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        
        Log.d(TAG, "NMS: " + detections.size() + " -> " + result.size() + " detections");
        return result;
    }
    
    private float calculateIoU(Detection a, Detection b) {
        // Calculate intersection
        float left = Math.max(a.left, b.left);
        float top = Math.max(a.top, b.top);
        float right = Math.min(a.left + a.width, b.left + b.width);
        float bottom = Math.min(a.top + a.height, b.top + b.height);
        
        if (right <= left || bottom <= top) {
            return 0.0f; // No intersection
        }
        
        float intersection = (right - left) * (bottom - top);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        float union = areaA + areaB - intersection;
        
        return intersection / union;
    }
    

    
    private Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Log.d(TAG, "Rotated bitmap by " + degrees + " degrees: " + bitmap.getWidth() + "x" + bitmap.getHeight() + 
              " -> " + rotatedBitmap.getWidth() + "x" + rotatedBitmap.getHeight());
        return rotatedBitmap;
    }
    
    private Bitmap resizeBitmapForModel(Bitmap originalBitmap) {
        // Crop and resize to match exactly what's visible in the preview
        int originalWidth = originalBitmap.getWidth();
        int originalHeight = originalBitmap.getHeight();
        
        // PreviewView is 320x320, camera is 640x640
        // We need to crop the camera image to match the preview aspect ratio
        float previewAspectRatio = 1.0f; // 320x320 is square
        float cameraAspectRatio = (float) originalWidth / originalHeight;
        
        int cropWidth, cropHeight, cropX, cropY;
        
        if (cameraAspectRatio > previewAspectRatio) {
            // Camera is wider than preview - crop width
            cropHeight = originalHeight;
            cropWidth = Math.round(originalHeight * previewAspectRatio);
            cropX = (originalWidth - cropWidth) / 2;
            cropY = 0;
        } else {
            // Camera is taller than preview - crop height
            cropWidth = originalWidth;
            cropHeight = Math.round(originalWidth / previewAspectRatio);
            cropX = 0;
            cropY = (originalHeight - cropHeight) / 2;
        }
        
        // Crop the bitmap to match preview aspect ratio
        Bitmap croppedBitmap = Bitmap.createBitmap(originalBitmap, cropX, cropY, cropWidth, cropHeight);
        
        // Resize to 640x640 (model input size)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, INPUT_SIZE, INPUT_SIZE, true);
        
        Log.d(TAG, "Cropped camera frame from " + originalWidth + "x" + originalHeight + 
              " to " + cropWidth + "x" + cropHeight + " (crop: " + cropX + "," + cropY + 
              ") then resized to " + INPUT_SIZE + "x" + INPUT_SIZE);
        
        return resizedBitmap;
    }
    

    
    @OptIn(markerClass = ExperimentalGetImage.class)
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        try {
            android.media.Image mediaImage = image.getImage();
            if (mediaImage == null) {
                Log.w(TAG, "MediaImage is null");
                return null;
            }
            
            // Get the YUV planes
            android.media.Image.Plane[] planes = mediaImage.getPlanes();
            if (planes.length < 3) {
                Log.w(TAG, "Expected 3 planes, got: " + planes.length);
                return null;
            }
            
            ByteBuffer yBuffer = planes[0].getBuffer();
            ByteBuffer uBuffer = planes[1].getBuffer();
            ByteBuffer vBuffer = planes[2].getBuffer();
            
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            
            Log.d(TAG, "YUV sizes: Y=" + ySize + ", U=" + uSize + ", V=" + vSize);
            
            byte[] nv21 = new byte[ySize + uSize + vSize];
            
            // Copy Y plane
            yBuffer.get(nv21, 0, ySize);
            // Copy U and V planes
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            
            android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(
                    nv21,
                    android.graphics.ImageFormat.NV21,
                    image.getWidth(),
                    image.getHeight(),
                    null);
            
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);
            byte[] imageBytes = out.toByteArray();
            
            Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            if (bitmap == null) {
                Log.w(TAG, "Failed to decode bitmap from JPEG");
            }
            return bitmap;
        } catch (Exception e) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e);
            return null;
        }
    }
    
    private void displayCameraResults(DetectionResult result) {
        if (result.detections.isEmpty()) {
            overlayView.clearDetection();
            overlayView.setVisibility(View.INVISIBLE);
            return;
        }

        // Get the actual preview view dimensions
        int previewWidth = cameraPreview.getWidth();
        int previewHeight = cameraPreview.getHeight();
        
        if (previewWidth <= 0 || previewHeight <= 0) {
            Log.w(TAG, "Preview view not ready yet, skipping overlay update");
            return;
        }

        Log.d(TAG, "Preview view size: " + previewWidth + "x" + previewHeight);
        Log.d(TAG, "Overlay view size: " + overlayView.getWidth() + "x" + overlayView.getHeight());

        // Calculate the actual displayed camera preview bounds within the PreviewView
        // CameraX preview uses CENTER_CROP by default, so we need to calculate the visible area
        
        float cameraAspectRatio = 1.0f; // Your camera is 640x640 (square)
        float previewAspectRatio = (float) previewWidth / previewHeight;
        
        float visibleWidth, visibleHeight;
        float offsetX = 0, offsetY = 0;
        
        if (previewAspectRatio > cameraAspectRatio) {
            // Preview is wider than camera - camera fills height, crops width
            visibleHeight = previewHeight;
            visibleWidth = visibleHeight * cameraAspectRatio;
            offsetX = (previewWidth - visibleWidth) / 2;
            offsetY = 0;
        } else {
            // Preview is taller than camera - camera fills width, crops height  
            visibleWidth = previewWidth;
            visibleHeight = visibleWidth / cameraAspectRatio;
            offsetX = 0;
            offsetY = (previewHeight - visibleHeight) / 2;
        }
        
        Log.d(TAG, "Visible camera area: " + visibleWidth + "x" + visibleHeight);
        Log.d(TAG, "Camera preview offset: (" + offsetX + ", " + offsetY + ")");

        // Now map from model coordinates (640x640) to the visible preview area
        List<DetectionOverlayView.DetectionBox> overlayDetections = new ArrayList<>();
        
        float cameraConfidenceThreshold = 0.5f;
        int validDetections = 0;
        
        for (Detection detection : result.detections) {
            if (detection.confidence < cameraConfidenceThreshold) {
                continue;
            }
            
            validDetections++;
            
            // Scale from model coordinates (640x640) to visible preview area
            float scaleX = visibleWidth / 640f;
            float scaleY = visibleHeight / 640f;
            
            // Apply scaling and offset
            float scaledLeft = detection.left * scaleX + offsetX;
            float scaledTop = detection.top * scaleY + offsetY;
            float scaledRight = (detection.left + detection.width) * scaleX + offsetX;
            float scaledBottom = (detection.top + detection.height) * scaleY + offsetY;
            
            // Clamp to preview bounds
            scaledLeft = Math.max(0, Math.min(scaledLeft, previewWidth));
            scaledTop = Math.max(0, Math.min(scaledTop, previewHeight));
            scaledRight = Math.max(0, Math.min(scaledRight, previewWidth));
            scaledBottom = Math.max(0, Math.min(scaledBottom, previewHeight));
            
            Log.d(TAG, "Detection mapping: model(" + detection.left + "," + detection.top + "," + 
                  (detection.left + detection.width) + "," + (detection.top + detection.height) + 
                  ") -> preview(" + scaledLeft + "," + scaledTop + "," + scaledRight + "," + scaledBottom + 
                  ") confidence: " + detection.confidence);
            
            overlayDetections.add(new DetectionOverlayView.DetectionBox(
                scaledLeft, scaledTop, scaledRight, scaledBottom, 
                detection.confidence, false));
        }
        
        Log.d(TAG, "Displaying " + validDetections + " detections");
        
        overlayView.setDetections(overlayDetections);
        overlayView.setVisibility(View.VISIBLE);
        overlayView.invalidate();
    }

    private void checkPermissionAndSelectImage() {
        String permission;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses READ_MEDIA_IMAGES
            permission = Manifest.permission.READ_MEDIA_IMAGES;
        } else {
            // Android 12 and below uses READ_EXTERNAL_STORAGE
            permission = Manifest.permission.READ_EXTERNAL_STORAGE;
        }

        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            openImagePicker();
        } else {
            requestPermissionLauncher.launch(permission);
        }
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        imagePickerLauncher.launch(intent);
    }

    private void loadImageFromUri(Uri imageUri) {
        try {
            // Store the image URI for later use
            selectedImageUri = imageUri;

            // Load image with proper orientation
            selectedImage = getCorrectlyOrientedBitmap(imageUri);

            // Store original dimensions
            originalImageWidth = selectedImage.getWidth();
            originalImageHeight = selectedImage.getHeight();

            imageView.setImageBitmap(selectedImage);

            // Ensure overlay view is properly positioned and visible
            imageOverlayView.setVisibility(View.INVISIBLE); // Start invisible
            imageOverlayView.clearDetection();

            processButton.setEnabled(true);
            processButton.setVisibility(View.VISIBLE);
            resultsText.setText("📸 Image selected\nTap '🔍 Detect Arrows' to process image");

            // Log.d(TAG, "Image loaded successfully, original size: " + originalImageWidth + "x" + originalImageHeight); // Removed
            // Log.d(TAG, "Overlay view size set to: " + originalImageWidth + "x" + originalImageHeight); // Removed

        } catch (Exception e) {
            Log.e(TAG, "Error loading image", e);
            Toast.makeText(this, "Error loading image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private Bitmap getCorrectlyOrientedBitmap(Uri imageUri) throws IOException {
        InputStream inputStream = getContentResolver().openInputStream(imageUri);

        // First, decode with inJustDecodeBounds=true to check dimensions
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(inputStream, null, options);
        inputStream.close();

        // Don't downsample - keep original dimensions
        options.inSampleSize = 1;
        options.inJustDecodeBounds = false;

        // Decode the bitmap
        inputStream = getContentResolver().openInputStream(imageUri);
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream, null, options);
        inputStream.close();

        // Get the orientation from EXIF data
        int orientation = getImageOrientation(imageUri);

        // Rotate the bitmap if needed
        if (orientation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(orientation);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }

        return bitmap;
    }

    private int getImageOrientation(Uri imageUri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            ExifInterface exif = new ExifInterface(inputStream);
            inputStream.close();

            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return 90;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    return 180;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    return 270;
                default:
                    return 0;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error reading EXIF data", e);
            return 0;
        }
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }

        return inSampleSize;
    }

    private void processImage() {
        // Log.v(TAG, "processImage() called"); // Removed

        if (selectedImage == null) {
            // Log.v(TAG, "No image selected"); // Removed
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        if (tflite == null) {
            // Log.v(TAG, "Model not loaded"); // Removed
            Toast.makeText(this, "Model not loaded. Please restart the app.", Toast.LENGTH_LONG).show();
            return;
        }

        // Log.v(TAG, "Starting image processing..."); // Removed

        // Show processing message
        resultsText.setText("🔄 Processing image...\nPlease wait...");

        // Run inference on background thread
        new Thread(() -> {
            // Log.v(TAG, "Starting inference thread"); // Removed
            DetectionResult result = runInference(selectedImage, selectedImageUri);
            // Log.v(TAG, "Inference completed, displaying results"); // Removed
            runOnUiThread(() -> displayResults(result));
        }).start();
    }
    private DetectionResult runInference(Bitmap image, Uri imageUri) {
        long totalStartTime = System.currentTimeMillis();
        long stageStartTime, stageEndTime;

        try {
                    // Log.d(TAG, "=== INFERENCE PROFILING START ===");
        // Log.d(TAG, "Input image size: " + image.getWidth() + "x" + image.getHeight());
        // Log.d(TAG, "Model input size: " + INPUT_SIZE + "x" + INPUT_SIZE);

            // Stage 1: Image Resizing
            stageStartTime = System.currentTimeMillis();
            Bitmap resizedImage = Bitmap.createScaledBitmap(image, INPUT_SIZE, INPUT_SIZE, true);
            stageEndTime = System.currentTimeMillis();
            // Log.d(TAG, "STAGE 1 - Image Resize: " + (stageEndTime - stageStartTime) + "ms");

            // Stage 2: Bitmap to ByteBuffer conversion
            stageStartTime = System.currentTimeMillis();
            ByteBuffer inputBuffer = convertBitmapToByteBuffer(resizedImage);
            stageEndTime = System.currentTimeMillis();
            // Log.d(TAG, "STAGE 2 - Bitmap to ByteBuffer: " + (stageEndTime - stageStartTime) + "ms");

            // Stage 3: Output array preparation
            stageStartTime = System.currentTimeMillis();
            int[] outputShape = tflite.getOutputTensor(0).shape();
            float[][][] outputArray = new float[outputShape[0]][outputShape[1]][outputShape[2]];
            stageEndTime = System.currentTimeMillis();
            // Log.d(TAG, "STAGE 3 - Output Array Prep: " + (stageEndTime - stageStartTime) + "ms");
            // Log.d(TAG, "Output shape: [" + outputShape[0] + "][" + outputShape[1] + "][" + outputShape[2] + "]");

            // Stage 4: Actual TensorFlow Lite inference
            stageStartTime = System.currentTimeMillis();
            tflite.run(inputBuffer, outputArray);
            stageEndTime = System.currentTimeMillis();
            // Log.d(TAG, "STAGE 4 - TFLite Inference: " + (stageEndTime - stageStartTime) + "ms *** CORE INFERENCE ***");

            // Stage 5: Detection parsing
            stageStartTime = System.currentTimeMillis();
            List<Detection> detections = parseDetections(outputArray, image.getWidth(), image.getHeight(), imageUri);
            stageEndTime = System.currentTimeMillis();
            // Log.d(TAG, "STAGE 5 - Detection Parsing: " + (stageEndTime - stageStartTime) + "ms");

            long totalTime = System.currentTimeMillis() - totalStartTime;
            // Log.d(TAG, "=== TOTAL INFERENCE TIME: " + totalTime + "ms ===");
            // Log.d(TAG, "Detections found: " + detections.size());

            return new DetectionResult(detections, totalTime);

        } catch (Exception e) {
            Log.e(TAG, "Error during inference", e);
            return new DetectionResult(new ArrayList<>(), 0);
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        long startTime = System.currentTimeMillis();

        // Sub-stage 2a: ByteBuffer allocation
        long subStart = System.currentTimeMillis();
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        // Log.d(TAG, "  2a - ByteBuffer allocation: " + (System.currentTimeMillis() - subStart) + "ms"); // Removed

        // Sub-stage 2b: Pixel extraction
        subStart = System.currentTimeMillis();
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        // Log.d(TAG, "  2b - Pixel extraction: " + (System.currentTimeMillis() - subStart) + "ms"); // Removed

        // Sub-stage 2c: Normalization and buffer filling
        subStart = System.currentTimeMillis();
        for (int pixel : pixels) {
            byteBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
            byteBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
            byteBuffer.putFloat((pixel & 0xFF) / 255.0f);         // B
        }
        // Log.d(TAG, "  2c - Normalization: " + (System.currentTimeMillis() - subStart) + "ms"); // Removed

        // Log.d(TAG, "  Total convertBitmapToByteBuffer: " + (System.currentTimeMillis() - startTime) + "ms"); // Removed
        return byteBuffer;
    }
    private List<Detection> parseDetections(float[][][] detections, int originalWidth, int originalHeight, Uri imageUri) {
        List<Detection> allDetections = new ArrayList<>();

        // Constants for the new model (trained on 640x640)
        final float MODEL_INPUT_SIZE = 640f;
        final float CONFIDENCE_THRESHOLD = 0.3f;
        final float IOU_THRESHOLD = 0.5f; // IoU threshold for non-maximum suppression

        Log.d(TAG, "=== DETECTION PARSING DEBUG START ===");
        Log.d(TAG, "Input parameters - originalWidth: " + originalWidth + ", originalHeight: " + originalHeight);
        Log.d(TAG, "Model output shape: [" + detections.length + "][" + detections[0].length + "][" + detections[0][0].length + "]");

        // Model output format: [1][5][8400] - single class model
        // Each detection has: [x_center, y_center, width, height, confidence]
        int numChannels = detections[0].length;
        int numDetections = detections[0][0].length;
        
        Log.d(TAG, "Model output analysis:");
        Log.d(TAG, "  - Number of channels: " + numChannels + " (expected: 5)");
        Log.d(TAG, "  - Number of detections: " + numDetections + " (expected: 8400)");
        
        if (numChannels == 5) {
            Log.d(TAG, "  - ✅ Correct single-class YOLO format detected");
            
            // Process each detection
            for (int detectionIdx = 0; detectionIdx < numDetections; detectionIdx++) {
                // Extract bounding box coordinates (normalized 0-1)
                float x_center = detections[0][0][detectionIdx]; // Center x
                float y_center = detections[0][1][detectionIdx]; // Center y
                float width = detections[0][2][detectionIdx];    // Width
                float height = detections[0][3][detectionIdx];   // Height
                float confidence = detections[0][4][detectionIdx]; // Confidence
                
                if (confidence > CONFIDENCE_THRESHOLD) {
                    // Convert normalized coordinates to pixel coordinates
                    // Model was trained on 640x640, so coordinates are normalized to 640x640
                    float modelX = x_center * MODEL_INPUT_SIZE;
                    float modelY = y_center * MODEL_INPUT_SIZE;
                    float modelWidth = width * MODEL_INPUT_SIZE;
                    float modelHeight = height * MODEL_INPUT_SIZE;
                    
                    // Calculate scaling factors to map from model coordinates to original image
                    float scaleX = (float) originalWidth / MODEL_INPUT_SIZE;
                    float scaleY = (float) originalHeight / MODEL_INPUT_SIZE;
                    
                    // Convert to original image coordinates
                    float pixelX = modelX * scaleX;
                    float pixelY = modelY * scaleY;
                    float pixelWidth = modelWidth * scaleX;
                    float pixelHeight = modelHeight * scaleY;
                    
                    // Calculate bounding box coordinates
                    float left = pixelX - pixelWidth / 2;
                    float top = pixelY - pixelHeight / 2;
                    float right = pixelX + pixelWidth / 2;
                    float bottom = pixelY + pixelHeight / 2;
                    
                    // Clamp to image bounds
                    left = Math.max(0, Math.min(left, originalWidth));
                    top = Math.max(0, Math.min(top, originalHeight));
                    right = Math.max(0, Math.min(right, originalWidth));
                    bottom = Math.max(0, Math.min(bottom, originalHeight));
                    
                    allDetections.add(new Detection(left, top, right - left, bottom - top, confidence));
                }
            }
        } else {
            Log.w(TAG, "❌ Unexpected model output format. Channels: " + numChannels + 
                  " (expected 5 for single-class model)");
            
            // Fallback: try to find any high confidence values
            float maxConfidence = 0;
            int maxI = 0, maxJ = 0, maxK = 0;
            
            for (int i = 0; i < detections.length; i++) {
                for (int j = 0; j < detections[i].length; j++) {
                    for (int k = 0; k < detections[i][j].length; k++) {
                        if (detections[i][j][k] > maxConfidence) {
                            maxConfidence = detections[i][j][k];
                            maxI = i; maxJ = j; maxK = k;
                        }
                    }
                }
            }
            
            Log.d(TAG, "Highest confidence value: " + maxConfidence + " at [" + maxI + "][" + maxJ + "][" + maxK + "]");
            
            if (maxConfidence > CONFIDENCE_THRESHOLD) {
                // Create a detection at the center of the image as fallback
                float detectionSize = Math.min(originalWidth, originalHeight) * 0.1f;
                float left = (originalWidth - detectionSize) / 2;
                float top = (originalHeight - detectionSize) / 2;
                
                allDetections.add(new Detection(left, top, detectionSize, detectionSize, maxConfidence));
                Log.d(TAG, "Added fallback detection with confidence: " + maxConfidence);
            }
        }
        
        // Apply non-maximum suppression to remove overlapping detections
        List<Detection> filteredDetections = applyNonMaxSuppression(allDetections, IOU_THRESHOLD);
        
        // Sort by confidence and take top 4
        filteredDetections.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<Detection> finalDetections = filteredDetections.size() > 4 ? 
            filteredDetections.subList(0, 4) : filteredDetections;
        
        // Log final detections
        Log.d(TAG, "=== FINAL DETECTIONS ===");
        for (int i = 0; i < finalDetections.size(); i++) {
            Detection detection = finalDetections.get(i);
            Log.d(TAG, "Detection " + (i + 1) + ": box(" + 
                  String.format("%.1f", detection.left) + "," + String.format("%.1f", detection.top) + "," +
                  String.format("%.1f", detection.left + detection.width) + "," + String.format("%.1f", detection.top + detection.height) + 
                  ") confidence=" + String.format("%.4f", detection.confidence));
        }
        Log.d(TAG, "=== DETECTION PARSING DEBUG END ===");
        Log.d(TAG, "Original detections: " + allDetections.size() + ", After NMS: " + filteredDetections.size() + ", Final: " + finalDetections.size());

        return finalDetections;
    }

    private List<Detection> parseDetectionsAlternative(float[][][] outputArray, int originalWidth, int originalHeight) {
        List<Detection> results = new ArrayList<>();

        // Log.d(TAG, "Trying alternative parsing with output array shape: " + outputArray.length + "x" + outputArray[0].length + "x" + outputArray[0][0].length); // Removed

        // Try different output array interpretations
        for (int i = 0; i < outputArray.length; i++) {
            for (int j = 0; j < outputArray[i].length; j++) {
                float[] detection = outputArray[i][j];

                // Look for high confidence values (0.5 or higher)
                for (int k = 0; k < detection.length; k++) {
                    if (detection[k] > 0.5) {
                        // Log.d(TAG, "Found high confidence value: " + detection[k] + " at [" + i + "][" + j + "][" + k + "]"); // Removed

                        // Try to interpret this as a detection
                        // This is a simplified approach - adjust based on your model's actual output format
                        float confidence = detection[k];
                        float centerX = (j * 32) % originalWidth; // Simplified coordinate calculation
                        float centerY = (j * 32) / originalWidth * 32;
                        float width = 100; // Default width
                        float height = 100; // Default height

                        float left = centerX - width / 2;
                        float top = centerY - height / 2;

                        results.add(new Detection(left, top, width, height, confidence));
                        // Log.d(TAG, "Added alternative detection with confidence: " + confidence); // Removed
                    }
                }
            }
        }

        return results;
    }

    private void dumpModelOutput(float[][][] outputArray) {
        // Log.d(TAG, "=== MODEL OUTPUT DUMP ==="); // Removed
        // Log.d(TAG, "Output array shape: " + outputArray.length + "x" + outputArray[0].length + "x" + outputArray[0][0].length); // Removed

        // Find the highest values in the output
        float maxValue = 0;
        int maxI = 0, maxJ = 0, maxK = 0;

        for (int i = 0; i < outputArray.length; i++) {
            for (int j = 0; j < outputArray[i].length; j++) {
                for (int k = 0; k < outputArray[i][j].length; k++) {
                    if (outputArray[i][j][k] > maxValue) {
                        maxValue = outputArray[i][j][k];
                        maxI = i; maxJ = j; maxK = k;
                    }
                }
            }
        }

        // Log.d(TAG, "Highest value in output: " + maxValue + " at [" + maxI + "][" + maxJ + "][" + maxK + "]"); // Removed

        // Show the first few rows of the output
        // Log.d(TAG, "First 5 rows of output:"); // Removed
        // for (int i = 0; i < Math.min(5, outputArray[0].length); i++) { // Removed
        //     StringBuilder row = new StringBuilder("Row " + i + ": "); // Removed
        //     for (int j = 0; j < Math.min(10, outputArray[0][i].length); j++) { // Removed
        //         row.append(String.format("%.3f ", outputArray[0][i][j])); // Removed
        //     } // Removed
        //     Log.d(TAG, row.toString()); // Removed
        // } // Removed

        // Count how many values are above 0.5
        int highConfidenceCount = 0;
        for (int i = 0; i < outputArray.length; i++) {
            for (int j = 0; j < outputArray[i].length; j++) {
                for (int k = 0; k < outputArray[i][j].length; k++) {
                    if (outputArray[i][j][k] > 0.5) {
                        highConfidenceCount++;
                    }
                }
            }
        }
        // Log.d(TAG, "Values above 0.5: " + highConfidenceCount); // Removed
        // Log.d(TAG, "=== END MODEL OUTPUT DUMP ==="); // Removed
    }

    private List<Detection> analyzeRawOutput(float[][][] outputArray, int originalWidth, int originalHeight) {
        List<Detection> results = new ArrayList<>();

        // Log.d(TAG, "Analyzing raw output array..."); // Removed

        // Find the highest confidence values in the entire array
        float maxConfidence = 0;
        int maxI = 0, maxJ = 0, maxK = 0;

        for (int i = 0; i < outputArray.length; i++) {
            for (int j = 0; j < outputArray[i].length; j++) {
                for (int k = 0; k < outputArray[i][j].length; k++) {
                    if (outputArray[i][j][k] > maxConfidence) {
                        maxConfidence = outputArray[i][j][k];
                        maxI = i; maxJ = j; maxK = k;
                    }
                }
            }
        }

        // Log.d(TAG, "Highest confidence found: " + maxConfidence + " at [" + maxI + "][" + maxJ + "][" + maxK + "]"); // Removed

        // If we found a high confidence value, try to interpret it
        if (maxConfidence > 0.5) {
            // Try to interpret the indices as coordinates
            float centerX = (maxJ * 32) % originalWidth;
            float centerY = (maxJ * 32) / originalWidth * 32;
            float width = 100;
            float height = 100;

            float left = centerX - width / 2;
            float top = centerY - height / 2;

            results.add(new Detection(left, top, width, height, maxConfidence));
            // Log.d(TAG, "Added raw detection with confidence: " + maxConfidence + " at position (" + left + ", " + top + ")"); // Removed
        }

        return results;
    }

    private List<Detection> parseClassificationOutput(float[][][] outputArray, int originalWidth, int originalHeight) {
        List<Detection> results = new ArrayList<>();

        // Log.d(TAG, "Trying classification-style output parsing..."); // Removed

        // If the model outputs classification probabilities, look for high confidence classes
        for (int i = 0; i < outputArray.length; i++) {
            for (int j = 0; j < outputArray[i].length; j++) {
                float[] classProbs = outputArray[i][j];

                // Find the class with highest probability
                float maxProb = 0;
                int maxClass = 0;

                for (int k = 0; k < classProbs.length; k++) {
                    if (classProbs[k] > maxProb) {
                        maxProb = classProbs[k];
                        maxClass = k;
                    }
                }

                // If we found a high confidence class (arrow or target)
                if (maxProb > 0.5) {
                    // Log.d(TAG, "Found high confidence class: " + maxClass + " with probability: " + maxProb); // Removed

                    // Create a detection at the center of the image
                    float centerX = originalWidth / 2.0f;
                    float centerY = originalHeight / 2.0f;
                    float width = Math.min(originalWidth, originalHeight) * 0.3f; // 30% of image
                    float height = width;

                    float left = centerX - width / 2;
                    float top = centerY - height / 2;

                    results.add(new Detection(left, top, width, height, maxProb));
                    // Log.d(TAG, "Added classification detection with confidence: " + maxProb + " for class: " + maxClass); // Removed
                }
            }
        }

        return results;
    }

    private void displayResults(DetectionResult result) {
        Log.d(TAG, "Displaying results: " + result.detections.size() + " detections, " + result.processingTime + "ms");

        StringBuilder sb = new StringBuilder();
        sb.append("🔍 **Detection Results**\n\n");
        sb.append("⏱️ Processing time: ").append(result.processingTime).append(" ms\n\n");

        if (result.detections.isEmpty()) {
            sb.append("❌ No circles detected\n");
            sb.append("Try a different image or check if the image contains circles");
            imageOverlayView.clearDetection();
            imageOverlayView.setVisibility(View.INVISIBLE);
        } else {
            // Convert detections to overlay format
            List<DetectionOverlayView.DetectionBox> overlayDetections = new ArrayList<>();

                // Calculate the actual displayed image size (accounting for centerInside scaling)
                float imageAspectRatio = (float) originalImageWidth / originalImageHeight;
                float viewAspectRatio = (float) imageView.getWidth() / imageView.getHeight();

                float displayedImageWidth, displayedImageHeight;
                float offsetX, offsetY;

                if (imageAspectRatio > viewAspectRatio) {
                    // Image is wider than view - fit to width
                    displayedImageWidth = imageView.getWidth();
                    displayedImageHeight = displayedImageWidth / imageAspectRatio;
                    offsetX = 0;
                    offsetY = (imageView.getHeight() - displayedImageHeight) / 2;
                } else {
                    // Image is taller than view - fit to height
                    displayedImageHeight = imageView.getHeight();
                    displayedImageWidth = displayedImageHeight * imageAspectRatio;
                    offsetX = (imageView.getWidth() - displayedImageWidth) / 2;
                    offsetY = 0;
                }

                // Scale coordinates to match the displayed image size
                float scaleX = displayedImageWidth / originalImageWidth;
                float scaleY = displayedImageHeight / originalImageHeight;

            Log.d(TAG, "Image view size: " + imageView.getWidth() + "x" + imageView.getHeight());
            Log.d(TAG, "Original image size: " + originalImageWidth + "x" + originalImageHeight);
            Log.d(TAG, "Displayed image size: " + displayedImageWidth + "x" + displayedImageHeight);
            Log.d(TAG, "Offset: (" + offsetX + ", " + offsetY + ")");
            Log.d(TAG, "Scale factors: scaleX=" + scaleX + ", scaleY=" + scaleY);

            // Find the detection with highest confidence
            Detection bestDetection = null;
            float maxConfidence = 0;

            for (Detection detection : result.detections) {
                if (detection.confidence > maxConfidence) {
                    maxConfidence = detection.confidence;
                    bestDetection = detection;
                }
                
                // Scale bounding box coordinates
                float scaledLeft = detection.left * scaleX + offsetX;
                float scaledTop = detection.top * scaleY + offsetY;
                float scaledRight = (detection.left + detection.width) * scaleX + offsetX;
                float scaledBottom = (detection.top + detection.height) * scaleY + offsetY;
                
                // For circle detection, we don't need the inTarget concept
                boolean inTarget = false; // Not used for circles
                
                overlayDetections.add(new DetectionOverlayView.DetectionBox(
                    scaledLeft, scaledTop, scaledRight, scaledBottom, 
                    detection.confidence, inTarget));
                
                Log.d(TAG, "Added overlay detection: box(" + 
                      String.format("%.1f", scaledLeft) + "," + String.format("%.1f", scaledTop) + "," +
                      String.format("%.1f", scaledRight) + "," + String.format("%.1f", scaledBottom) + 
                      ") confidence=" + String.format("%.4f", detection.confidence));
            }

            // Set the detections on the overlay
            imageOverlayView.setDetections(overlayDetections);

                // Force the overlay to be visible and on top
                imageOverlayView.bringToFront();
                imageOverlayView.setVisibility(View.VISIBLE);

                // Force immediate redraw
                imageOverlayView.post(() -> {
                    imageOverlayView.invalidate();
                Log.d(TAG, "Forced overlay redraw");
                Log.d(TAG, "Overlay size: " + imageOverlayView.getWidth() + "x" + imageOverlayView.getHeight());

                    // Force another redraw after a short delay to ensure it's visible
                    imageOverlayView.postDelayed(() -> {
                        imageOverlayView.invalidate();
                    Log.d(TAG, "Second forced overlay redraw");
                    }, 100);
                });

            if (bestDetection != null) {
                sb.append("✅ Found ").append(result.detections.size()).append(" circle(s):\n\n");
                
                for (int i = 0; i < result.detections.size(); i++) {
                    Detection detection = result.detections.get(i);
                    sb.append("⭕ Circle ").append(i + 1).append(":\n");
                    sb.append("   Center: (").append(String.format("%.1f", detection.getCenterX()))
                      .append(", ").append(String.format("%.1f", detection.getCenterY())).append(")\n");
                    sb.append("   Radius: ").append(String.format("%.1f", Math.min(detection.width, detection.height) / 2)).append(" px\n");
                    sb.append("   Confidence: ").append(String.format("%.1f%%", detection.confidence * 100)).append("\n\n");
                }
                
                sb.append("(Circles show detected circle objects)");
            }
        }

        resultsText.setText(sb.toString());
        Log.d(TAG, "Results displayed successfully");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
    }

    // Data classes for results
    private static class Detection {
        float left, top, width, height, confidence;

        Detection(float left, float top, float width, float height, float confidence) {
            this.left = left;
            this.top = top;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
        }
        
        // Helper methods for bounding box operations
        float getRight() { return left + width; }
        float getBottom() { return top + height; }
        float getCenterX() { return left + width / 2; }
        float getCenterY() { return top + height / 2; }
    }

    private static class DetectionResult {
        List<Detection> detections;
        long processingTime;

        DetectionResult(List<Detection> detections, long processingTime) {
            this.detections = detections;
            this.processingTime = processingTime;
        }
    }
}
