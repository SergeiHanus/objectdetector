#!/usr/bin/env python3
"""
Export a trained RF-DETR model checkpoint to ONNX.

Usage example:
  /data/code/image-detector/detr/rf-detr/venv/bin/python \
    /data/code/image-detector/detr/tools/export_onnx.py \
    --checkpoint /data/code/image-detector/detr/output_tiled/checkpoint_best_regular.pth \
    --output /data/code/image-detector/detr/output_tiled/model_448.onnx \
    --resolution 448 --batch-size 1 --opset 17 --simplify
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
import shutil


def get_rfdetr_nano():
    """Instantiate RF-DETR Nano (robust to API variants)."""
    try:
        from rfdetr import RFDETRNano  # type: ignore
        return RFDETRNano()
    except Exception:
        from rfdetr import RFDETRBase  # type: ignore
        try:
            return RFDETRBase(model_size="nano")  # if supported
        except TypeError:
            return RFDETRBase()  # final fallback


def load_checkpoint_into_model(model, checkpoint_path: Path) -> None:
    import torch
    import torch.nn as nn

    def resolve_torch_module(obj):
        current = obj
        # Walk common attribute names to find an nn.Module
        for _ in range(4):
            if isinstance(current, nn.Module):
                return current
            for name in ("model", "module", "net"):
                if hasattr(current, name):
                    current = getattr(current, name)
                    break
            else:
                break
        return current

    model_to_load = resolve_torch_module(model)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    candidates = []
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            candidates.append(ckpt["model"])
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            candidates.append(ckpt["state_dict"])
    if isinstance(ckpt, dict):
        candidates.append(ckpt)  # try as a plain state dict as well

    last_error: Exception | None = None
    for state_dict in candidates:
        try:
            missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)  # type: ignore[attr-defined]
            # If nothing loaded, raise to try next variant
            if missing and len(missing) == len(list(model_to_load.state_dict().keys())):
                raise RuntimeError("State dict keys did not match model")
            return
        except Exception as e:  # try next variant
            last_error = e
            continue

    if last_error is not None:
        raise last_error


def try_export_onnx(model, onnx_path: Path, image_size: int, batch_size: int, opset: int, dynamic: bool, torch_only: bool = False) -> None:
    """Attempt multiple export APIs before falling back to torch.onnx.export."""
    # Prefer model-native exporters first, unless forcing torch-only
    if not torch_only:
        try:
            model.export(
                format="onnx",
                output_path=str(onnx_path),
                opset=opset,
                dynamic=dynamic,
                input_size=(image_size, image_size),
                batch_size=batch_size,
            )
            # Some exporters ignore output_path; ensure our target exists
            if not onnx_path.exists():
                fallback = Path("output/inference_model.onnx").resolve()
                if fallback.exists():
                    onnx_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(fallback), str(onnx_path))
            return
        except Exception:
            pass

        try:
            model.export_onnx(
                str(onnx_path),
                opset_version=opset,
                dynamic_axes=dynamic,
                input_size=(image_size, image_size),
                batch_size=batch_size,
            )
            if not onnx_path.exists():
                fallback = Path("output/inference_model.onnx").resolve()
                if fallback.exists():
                    onnx_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(fallback), str(onnx_path))
            return
        except Exception:
            pass

    # Fallback: raw torch.onnx.export
    import torch
    import torch.nn as nn

    def resolve_torch_module(obj):
        current = obj
        for _ in range(4):
            if isinstance(current, nn.Module):
                return current
            for name in ("model", "module", "net"):
                if hasattr(current, name):
                    current = getattr(current, name)
                    break
            else:
                break
        return current

    model_to_export = resolve_torch_module(model)
    if not isinstance(model_to_export, nn.Module):
        raise TypeError("Underlying torch.nn.Module not found on model; cannot export with torch.onnx.export")
    model_to_export.eval()

    dummy = torch.randn(batch_size, 3, image_size, image_size)
    try:
        device = next(model_to_export.parameters()).device  # type: ignore[attr-defined]
        dummy = dummy.to(device)
    except Exception:
        pass

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}

    torch.onnx.export(
        model_to_export,
        dummy,
        str(onnx_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["predictions"],
        dynamic_axes=dynamic_axes,
    )


def maybe_simplify(onnx_path: Path) -> Path:
    try:
        import onnx
        import onnxsim
        model = onnx.load(str(onnx_path))
        model_simplified, check_ok = onnxsim.simplify(model)
        if check_ok:
            sim_path = onnx_path.with_suffix(".sim.onnx")
            onnx.save(model_simplified, str(sim_path))
            return sim_path
        return onnx_path
    except Exception:
        return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RF-DETR model checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--output", type=str, default="", help="Output ONNX file path")
    parser.add_argument("--resolution", type=int, default=640, help="Square input size (e.g., 448)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export dummy input")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic batch dimension")
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier on the exported model")
    parser.add_argument("--torch-only", action="store_true", help="Force torch.onnx.export and skip native exporters")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).resolve() if args.output else ckpt_path.with_name(
        f"model_{args.resolution}.onnx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_rfdetr_nano()

    # Try to load checkpoint weights into the underlying torch model
    try:
        load_checkpoint_into_model(model, ckpt_path)
    except Exception as e:
        print(f"Warning: failed to load checkpoint strictly into model: {e}")
        # Continue; some model.export implementations accept resume weights separately

    # Prefer exporting on CPU for stability
    try:
        import torch
        model_to_export = getattr(model, "model", model)
        model_to_export.to("cpu")
    except Exception:
        pass

    try_export_onnx(
        model=model,
        onnx_path=out_path,
        image_size=args.resolution,
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic=args.dynamic,
        torch_only=args.torch_only,
    )

    final_path = out_path
    if args.simplify:
        final_path = maybe_simplify(out_path)

    print(f"Exported ONNX: {final_path}")


if __name__ == "__main__":
    main()


