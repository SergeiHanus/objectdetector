#!/usr/bin/env python3
"""
YOLOX Circle Detection - Model Export to ONNX

Export the trained YOLOX model to ONNX format for deployment.

Example:
  source yolox/bin/activate
  python src/yolox_export.py
  
  # Or with custom paths
  python src/yolox_export.py \
    --model /custom/path/to/best_ckpt.pth \
    --output /custom/path/to/output.onnx
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export YOLOX model to ONNX format")
    
    # Default paths (relative to execution directory)
    default_model = "YOLOX/YOLOX_outputs/yolox_s_circle/best_ckpt.pth"
    default_output = "models/yolox_circle_detector.onnx"
    default_exp = "exps/yolox_s_circle.py"
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=default_model,
        help=f"Path to trained model checkpoint (default: {default_model})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_output,
        help=f"Output path for ONNX model (default: {default_output})"
    )
    parser.add_argument(
        "--exp", 
        type=str, 
        default=default_exp,
        help=f"Path to experiment configuration (default: {default_exp})"
    )
    parser.add_argument(
        "--opset", 
        type=int, 
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--dynamic", 
        action="store_true",
        help="Export with dynamic batch size (default: fixed batch size)"
    )
    parser.add_argument(
        "--no-simplify", 
        action="store_true",
        help="Skip ONNX simplification step"
    )
    parser.add_argument(
        "--decode-in-inference", 
        action="store_true",
        help="Include decoding in the exported model"
    )
    
    args = parser.parse_args()
    
    # Use relative paths from execution directory
    model_path = args.model
    exp_path = args.exp
    output_path = args.output
    
    # Validate input files
    if not Path(model_path).exists():
        print(f"❌ Error: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    if not Path(exp_path).exists():
        print(f"❌ Error: Experiment config not found: {exp_path}")
        sys.exit(1)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build the export command with relative paths
    python_path = "yolox/bin/python"
    export_script = "YOLOX/tools/export_onnx.py"
    
    if not Path(python_path).exists():
        print(f"❌ Error: YOLOX Python environment not found: {python_path}")
        print("   Please run 'source yolox/bin/activate' first")
        sys.exit(1)
    
    if not Path(export_script).exists():
        print(f"❌ Error: YOLOX export script not found: {export_script}")
        sys.exit(1)
    
    cmd = [
        python_path,
        export_script,
        "-f", exp_path,
        "-c", model_path,
        "--output-name", output_path,
        "-o", str(args.opset),
    ]
    
    # Add optional flags
    if args.dynamic:
        cmd.append("--dynamic")
    
    if args.no_simplify:
        cmd.append("--no-onnxsim")
    
    if args.decode_in_inference:
        cmd.append("--decode_in_inference")
    
    print("🚀 Exporting YOLOX model to ONNX...")
    print(f"   Model: {model_path}")
    print(f"   Config: {exp_path}")
    print(f"   Output: {output_path}")
    print(f"   ONNX Opset: {args.opset}")
    print(f"   Dynamic batch: {args.dynamic}")
    print(f"   Simplify: {not args.no_simplify}")
    print(f"   Decode in inference: {args.decode_in_inference}")
    print()
    
    # Execute the export command
    try:
        print("📝 Running export command:")
        print(f"   {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Export completed successfully!")
        print("\n📊 Export output:")
        if result.stdout:
            print(result.stdout)
        
        # Verify the output file exists
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"\n✅ ONNX model saved: {output_path}")
            print(f"   File size: {file_size:.2f} MB")
            
            # Try to load with onnx to verify it's valid
            try:
                import onnx
                model = onnx.load(output_path)
                print(f"   Input shape: {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
                print(f"   Output shape: {[d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]}")
                print("   ✅ ONNX model validation passed")
            except ImportError:
                print("   ⚠️  Cannot validate ONNX model (onnx package not available)")
            except Exception as e:
                print(f"   ⚠️  ONNX model validation failed: {e}")
        else:
            print(f"❌ Error: Output file not created: {output_path}")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print("❌ Export failed!")
        print(f"   Exit code: {e.returncode}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
