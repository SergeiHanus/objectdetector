#!/usr/bin/env python3
"""
Export YOLOX model to ONNX (opset 17 friendly).

Why this exists:
- PyTorch 2.9+ defaults ONNX export to the torch.export/dynamo path, which can
  emit opset 18 and then attempt to down-convert to opset 17 (often fails for
  Resize). This script forces `dynamo=False` so opset 17 is truly honored.
- `onnxsim` can fail on some setups (checker/IR mismatch). We treat simplify as
  best-effort unless explicitly disabled.

This script is designed to replace the heavier bash logic from `src/export.sh`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _resolve_executable(p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    if path.is_file() and os.access(str(path), os.X_OK):
        return str(path)
    # Follow symlink if it exists but points elsewhere
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError:
            return None
        if target.is_file() and os.access(str(target), os.X_OK):
            return str(target)
    return None


def _pick_python(workdir: Path, preferred: str | None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(
        [
            "yolox/bin/python",
            "yolox/bin/python3.11",
            ".venv/bin/python",
            "python3.11",
            "python3",
            "python",
        ]
    )

    for cand in candidates:
        # absolute path
        resolved = _resolve_executable(cand)
        if resolved:
            return resolved

        # relative to workdir
        resolved = _resolve_executable(str(workdir / cand))
        if resolved:
            return resolved

        # PATH lookup
        which = shutil.which(cand)
        resolved = _resolve_executable(which) if which else None
        if resolved:
            return resolved

    raise SystemExit(
        f"Error: could not find a working Python interpreter.\n"
        f"Workdir: {workdir}\n"
        f"Tried: {', '.join(candidates)}\n"
        f"Fix: pass --python /path/to/python (or set PYTHON=...)"
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export YOLOX model to ONNX (dynamo=False).")
    p.add_argument(
        "--workdir",
        default=os.environ.get("WORKDIR", str(Path.home() / "image-detector" / "train_yolox")),
        help="YOLOX working directory (default: $WORKDIR or ~/image-detector/train_yolox)",
    )
    p.add_argument(
        "--python",
        default=os.environ.get("PYTHON", None),
        help="Python interpreter to use (default: $PYTHON or autodetect under workdir)",
    )

    p.add_argument(
        "-f",
        "--exp_file",
        default=os.environ.get("EXP", "exps/yolox_s_marker.py"),
        help="Experiment description file (default: $EXP or exps/yolox_s_marker.py)",
    )
    p.add_argument(
        "-c",
        "--ckpt",
        default=os.environ.get("CKPT", "YOLOX/YOLOX_outputs/yolox_s_marker/latest_ckpt.pth"),
        help="Checkpoint path (default: $CKPT or YOLOX/.../latest_ckpt.pth)",
    )
    p.add_argument(
        "--output-name",
        default=os.environ.get("OUTPUT", "models/yolox_marker_detector.onnx"),
        help="Output .onnx path (default: $OUTPUT or models/yolox_marker_detector.onnx)",
    )
    p.add_argument(
        "-o",
        "--opset",
        type=int,
        default=int(os.environ.get("OPSET", "17")),
        help="ONNX opset version (default: $OPSET or 17)",
    )
    p.add_argument("--input", default="images", type=str, help="ONNX input name")
    p.add_argument("--output", default="output", type=str, help="ONNX output name")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic batch axis")
    p.add_argument(
        "--no-onnxsim",
        action="store_true",
        default=(os.environ.get("SIMPLIFY", "1") == "0"),
        help="Skip onnxsim simplification (or set SIMPLIFY=0)",
    )
    p.add_argument("--decode_in_inference", action="store_true", help="Decode in inference")
    p.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra YOLOX exp options (passed to exp.merge)",
    )
    return p


def _run_export(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.is_dir():
        raise SystemExit(f"Error: workdir does not exist: {workdir}")

    os.chdir(workdir)
    exporter = workdir / "YOLOX" / "tools" / "export_onnx.py"
    if not exporter.exists():
        raise SystemExit(f"Error: cannot find YOLOX exporter at: {exporter}")

    py = _pick_python(workdir, args.python)

    # Import inside workdir so `yolox` package resolves in that env.
    # We still run in *this* process (not subprocess) so we can force dynamo=False.
    sys.path.insert(0, str(workdir / "YOLOX"))

    from loguru import logger  # type: ignore
    import torch  # type: ignore
    from torch import nn  # type: ignore

    from yolox.exp import get_exp  # type: ignore
    from yolox.models.network_blocks import SiLU  # type: ignore
    from yolox.utils import replace_module  # type: ignore

    logger.info("Using python: {}", py)
    logger.info(
        "Export: exp={} ckpt={} output={} opset={} simplify={}",
        args.exp_file,
        args.ckpt,
        args.output_name,
        args.opset,
        (not args.no_onnxsim),
    )

    exp = get_exp(args.exp_file, args.name if hasattr(args, "name") else None)
    exp.merge(args.opts)
    model = exp.get_model()

    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.eval()
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = bool(args.decode_in_inference)

    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    dyn_axes = None
    if args.dynamic:
        dyn_axes = {args.input: {0: "batch"}, args.output: {0: "batch"}}

    # Critical bit: dynamo=False keeps the legacy exporter and avoids opset18+downconvert.
    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes=dyn_axes,
        opset_version=int(args.opset),
        dynamo=False,
    )
    logger.info("Generated ONNX: {}", args.output_name)

    if not args.no_onnxsim:
        try:
            import onnx  # type: ignore
            from onnxsim import simplify  # type: ignore

            onnx_model = onnx.load(args.output_name)
            model_simp, check = simplify(onnx_model)
            if not check:
                logger.warning("Simplified ONNX model could not be validated; keeping original model.")
            else:
                onnx.save(model_simp, args.output_name)
                logger.info("Generated simplified ONNX: {}", args.output_name)
        except Exception as e:
            logger.warning("onnxsim failed; keeping original model: {}", e)


def main() -> None:
    args = make_parser().parse_args()
    _run_export(args)


if __name__ == "__main__":
    main()

