# C# ONNX Runtime Object Detection on Fedora 42

This project shows how to build a **C# console application** on **Fedora 42** that loads an image, runs inference with an **ONNX model**, draws green circles over detected objects, and saves the modified image, all without Unity by using **ONNX Runtime** in pure C#.

To get started, update your system and install .NET SDK 8.0 (LTS) plus required libraries:
```bash
sudo dnf upgrade --refresh
sudo dnf install -y dotnet-sdk-8.0 libgdiplus
dotnet --version

Now create an isolated console project:

mkdir onnx-app && cd onnx-app
dotnet new console -n Detector
cd Detector
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package SixLabors.ImageSharp 
dotnet add package SixLabors.ImageSharp.Drawing


Place your ONNX model and test image in the project folder:

Detector/
  Program.cs
  model.onnx
  input.png


Run inside the Detector/ folder with:

dotnet run

If successful, input.png will be processed and output.png will be created with a green circle.