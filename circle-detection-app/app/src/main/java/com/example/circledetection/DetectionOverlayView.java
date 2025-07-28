package com.example.circledetection;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

public class DetectionOverlayView extends View {
    private static final String TAG = "DetectionOverlay";
    private Paint boxPaint;
    private List<DetectionBox> detections = new ArrayList<>();

    public static class DetectionBox {
        public float left, top, right, bottom;
        public float confidence;
        public boolean inTarget;

        public DetectionBox(float left, float top, float right, float bottom, float confidence, boolean inTarget) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.confidence = confidence;
            this.inTarget = inTarget;
        }
    }

    public DetectionOverlayView(Context context) {
        super(context);
        init();
    }

    public DetectionOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);
        boxPaint.setAntiAlias(true);
    }

    public void setDetections(List<DetectionBox> detections) {
        this.detections = detections;
        // Log.d(TAG, "Setting " + detections.size() + " detections");
        postInvalidate();
    }

    public void setDetection(float left, float top, float right, float bottom, float confidence, boolean inTarget) {
        detections.clear();
        detections.add(new DetectionBox(left, top, right, bottom, confidence, inTarget));
        // Log.d(TAG, "Setting detection: box(" + left + ", " + top + ", " + right + ", " + bottom + 
        //       ") confidence: " + confidence + " inTarget: " + inTarget);
        postInvalidate();
    }

    public void clearDetection() {
        detections.clear();
        // Log.d(TAG, "Clearing detections");
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        // Log.d(TAG, "onDraw called with " + detections.size() + " detections");
        // Log.d(TAG, "View bounds: " + getWidth() + "x" + getHeight());
        
        for (DetectionBox detection : detections) {
            // Check if box is within view bounds
            if (detection.left >= 0 && detection.right <= getWidth() && 
                detection.top >= 0 && detection.bottom <= getHeight()) {
                
                // Calculate circle center and radius
                float centerX = (detection.left + detection.right) / 2;
                float centerY = (detection.top + detection.bottom) / 2;
                float radius = Math.min(detection.right - detection.left, detection.bottom - detection.top) / 2;
                
                // For circle detection, always use green color
                boxPaint.setColor(Color.GREEN);
                
                // Draw circle
                canvas.drawCircle(centerX, centerY, radius, boxPaint);
                // Log.d(TAG, "Drew circle: center(" + centerX + ", " + centerY + ") radius: " + radius);
            } else {
                Log.w(TAG, "Circle coordinates (" + detection.left + ", " + detection.top + 
                      ", " + detection.right + ", " + detection.bottom + 
                      ") outside view bounds (" + getWidth() + "x" + getHeight() + ")");
            }
        }
    }
} 