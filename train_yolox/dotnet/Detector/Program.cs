using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Drawing;
using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        string modelPath = "model.onnx";
        string imagePath = "image.png";
        string outputPath = "output.png";

        using var session = new InferenceSession(modelPath);

        using var orig = Image.Load<Rgba32>(imagePath);
        int targetSize = 640;
        var letterboxed = LetterboxImage(orig, targetSize, targetSize, out float ratio, out int padX, out int padY);

        Tensor<float> input = ImageToTensor(letterboxed);

        var inputs = new[] { NamedOnnxValue.CreateFromTensor("images", input) };
        using var results = session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // The model outputs [1, 8400, 6] format: [batch, detections, features]
        // Features are: [cx, cy, w, h, obj_conf, cls_conf]
        int numPreds = output.Dimensions[1]; // 8400 detections
        int cols = output.Dimensions[2];     // 6 features per detection
        
        Console.WriteLine($"Model output shape: {output.Dimensions[0]} x {output.Dimensions[1]} x {output.Dimensions[2]}");
        
        // Debug: Print first few raw outputs
        Console.WriteLine("First 3 raw outputs:");
        for (int i = 0; i < Math.Min(3, numPreds); i++)
        {
            Console.WriteLine($"  Row {i}: cx={output[0, i, 0]:F3}, cy={output[0, i, 1]:F3}, w={output[0, i, 2]:F3}, h={output[0, i, 3]:F3}, obj={output[0, i, 4]:F3}, cls={output[0, i, 5]:F3}");
        }
        
        List<Detection> detections = new List<Detection>();

        // Implement YOLOX demo_postprocess logic
        // The model outputs relative offsets that need to be added to grid positions
        int[] strides = { 8, 16, 32 }; // YOLOX feature map strides
        List<int> gridSizes = new List<int>();
        
        // Calculate grid sizes for each stride
        foreach (int stride in strides)
        {
            int gridSize = targetSize / stride;
            gridSizes.Add(gridSize);
        }
        
        // Generate grids for each scale
        List<float[]> grids = new List<float[]>();
        List<float[]> expandedStrides = new List<float[]>();
        
        int gridIndex = 0;
        foreach (int gridSize in gridSizes)
        {
            int stride = strides[gridIndex];
            int totalCells = gridSize * gridSize;
            
            // Generate grid coordinates
            float[] grid = new float[totalCells * 2];
            float[] expandedStride = new float[totalCells];
            
            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    int idx = i * gridSize + j;
                    grid[idx * 2] = j;     // x coordinate
                    grid[idx * 2 + 1] = i; // y coordinate
                    expandedStride[idx] = stride;
                }
            }
            
            grids.Add(grid);
            expandedStrides.Add(expandedStride);
            gridIndex++;
        }
        
        // Concatenate all grids
        int totalGridCells = gridSizes.Sum(x => x * x);
        float[] allGrids = new float[totalGridCells * 2];
        float[] allStrides = new float[totalGridCells];
        
        int offset = 0;
        for (int i = 0; i < grids.Count; i++)
        {
            int gridSize = grids[i].Length / 2;
            Array.Copy(grids[i], 0, allGrids, offset * 2, grids[i].Length);
            Array.Copy(expandedStrides[i], 0, allStrides, offset, expandedStrides[i].Length);
            offset += gridSize;
        }
        
        // Process each detection
        for (int i = 0; i < numPreds; i++)
        {
            // Extract raw outputs (these are relative offsets, not absolute coordinates)
            float cx_offset = output[0, i, 0];
            float cy_offset = output[0, i, 1];
            float w_log = output[0, i, 2];
            float h_log = output[0, i, 3];
            float obj = output[0, i, 4];
            float cls = output[0, i, 5];
            
            // Apply YOLOX post-processing: add grid coordinates and multiply by stride
            float cx = (cx_offset + allGrids[i * 2]) * allStrides[i];
            float cy = (cy_offset + allGrids[i * 2 + 1]) * allStrides[i];
            float w = MathF.Exp(w_log) * allStrides[i];
            float h = MathF.Exp(h_log) * allStrides[i];
            
            float score = obj * cls;
            
            // Debug: Print first few processed detections
            if (i < 5)
            {
                Console.WriteLine($"  Processed {i}: cx={cx:F1}, cy={cy:F1}, w={w:F1}, h={h:F1}, score={score:F3}");
            }
            
            // Apply confidence threshold
            if (score < 0.3f) continue;
            
            // Convert center-based coordinates to corner-based
            float x1 = cx - w / 2f;
            float y1 = cy - h / 2f;
            float x2 = cx + w / 2f;
            float y2 = cy + h / 2f;
            
            // Adjust for letterboxing
            float boxCx = (x1 + x2) / 2f - padX;
            float boxCy = (y1 + y2) / 2f - padY;
            float boxW = x2 - x1;
            float boxH = y2 - y1;
            
            // Scale back to original image dimensions
            boxCx /= ratio;
            boxCy /= ratio;
            boxW /= ratio;
            boxH /= ratio;
            
            detections.Add(new Detection
            {
                X = boxCx,
                Y = boxCy,
                W = boxW,
                H = boxH,
                Score = score
            });
        }

        Console.WriteLine($"Raw detections before NMS: {detections.Count}");
        
        var finalDetections = NMS(detections, 0.45f);
        Console.WriteLine($"Final detections after NMS: {finalDetections.Count}");

        var greenPen = Color.Lime;
        foreach (var det in finalDetections)
        {
            Console.WriteLine($"Detection: Center({det.X:F1}, {det.Y:F1}), Size({det.W:F1}x{det.H:F1}), Score: {det.Score:F3}");
            
            // Draw smaller circles for better visualization
            int radius = Math.Max(5, Math.Min(20, (int)(Math.Min(det.W, det.H) / 4)));
            var ellipse = new EllipsePolygon(det.X, det.Y, radius);
            orig.Mutate(ctx => ctx.Draw(greenPen, 2, ellipse));
        }

        orig.Save(outputPath);
        Console.WriteLine($"Saved {outputPath} with {finalDetections.Count} detections.");
    }

    static Tensor<float> ImageToTensor(Image<Rgba32> img)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

        for (int y = 0; y < img.Height; y++)
        for (int x = 0; x < img.Width; x++)
        {
            var px = img[x, y];
            // Keep values in 0-255 range as expected by YOLOX
            tensor[0, 0, y, x] = px.R;
            tensor[0, 1, y, x] = px.G;
            tensor[0, 2, y, x] = px.B;
        }

        return tensor;
    }

    static Image<Rgba32> LetterboxImage(Image<Rgba32> img, int targetW, int targetH, out float ratio, out int padX, out int padY)
    {
        int iw = img.Width;
        int ih = img.Height;
        ratio = Math.Min((float)targetW / iw, (float)targetH / ih);
        int newW = (int)(iw * ratio);
        int newH = (int)(ih * ratio);

        var resized = img.Clone(ctx => ctx.Resize(newW, newH));
        padX = (targetW - newW) / 2;
        padY = (targetH - newH) / 2;

        var letterboxed = new Image<Rgba32>(targetW, targetH);
        // Capture the values before using them in the lambda
        int capturedPadX = padX;
        int capturedPadY = padY;
        letterboxed.Mutate(ctx =>
        {
            ctx.Fill(new Rgba32(114, 114, 114));
            ctx.DrawImage(resized, new Point(capturedPadX, capturedPadY), 1f);
        });

        return letterboxed;
    }

    static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    static List<Detection> NMS(List<Detection> dets, float iouThresh)
    {
        var sorted = dets.OrderByDescending(d => d.Score).ToList();
        List<Detection> results = new List<Detection>();
        while (sorted.Count > 0)
        {
            var best = sorted[0];
            results.Add(best);
            sorted.RemoveAt(0);
            sorted = sorted.Where(d => IoU(best, d) < iouThresh).ToList();
        }
        return results;
    }

    static float IoU(Detection a, Detection b)
    {
        float x1 = Math.Max(a.X - a.W / 2, b.X - b.W / 2);
        float y1 = Math.Max(a.Y - a.H / 2, b.Y - b.H / 2);
        float x2 = Math.Min(a.X + a.W / 2, b.X + b.W / 2);
        float y2 = Math.Min(a.Y + a.H / 2, b.Y + b.H / 2);

        float interW = Math.Max(0, x2 - x1);
        float interH = Math.Max(0, y2 - y1);
        float inter = interW * interH;
        float union = a.W * a.H + b.W * b.H - inter;
        return union <= 0 ? 0 : inter / union;
    }

    class Detection
    {
        public float X, Y, W, H, Score;
    }
}
