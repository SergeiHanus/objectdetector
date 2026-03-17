#!/usr/bin/env python3
"""
Canonical training entry point. Run from train_yolox/ (canonical runtime root).
Contract: --exp selects exps/<exp>.py and dataset YOLOX/datasets/<name>_dataset;
outputs go to YOLOX/YOLOX_outputs/<exp>/ and are copied to models/.
"""

import os
import sys
import argparse
import shutil
import base64

# Resolve train_yolox root and ensure we run from there
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if os.path.basename(_ROOT) != "train_yolox" or not os.path.isdir(_ROOT):
    _ROOT = os.path.join(_ROOT, "train_yolox") if os.path.isdir(os.path.join(_ROOT, "train_yolox")) else _ROOT
if not os.path.exists(os.path.join(_ROOT, "exps")):
    print("Error: cannot find train_yolox root (exps/ not found)", file=sys.stderr)
    sys.exit(1)
os.chdir(_ROOT)


def exp_to_dataset_name(exp_name: str) -> str:
    if exp_name.startswith("yolox_s_"):
        return exp_name.replace("yolox_s_", "", 1) + "_dataset"
    return exp_name + "_dataset"


def check_layout(exp_name: str) -> bool:
    """Quick layout check for this exp."""
    if not os.path.isdir("YOLOX") or not os.path.isfile("YOLOX/tools/train.py"):
        print("Missing YOLOX/ or YOLOX/tools/train.py", file=sys.stderr)
        return False
    if not os.path.isfile(f"exps/{exp_name}.py"):
        print(f"Missing exps/{exp_name}.py", file=sys.stderr)
        return False
    ds = exp_to_dataset_name(exp_name)
    ds_path = f"YOLOX/datasets/{ds}"
    for sub in ("train/images", "val/images", "annotations/train_labels.json", "annotations/val_labels.json"):
        path = os.path.join(ds_path, sub)
        if not os.path.exists(path):
            print(f"Missing dataset path: {path}", file=sys.stderr)
            return False
    return True


def copy_artifacts(exp_name: str) -> None:
    """Copy YOLOX_outputs/<exp> checkpoints to models/."""
    out_dir = f"YOLOX/YOLOX_outputs/{exp_name}"
    if not os.path.isdir(out_dir):
        return
    os.makedirs("models", exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pth"):
                src = os.path.join(ckpt_dir, f)
                dst = os.path.join("models", f"yolox_{exp_name}_{f}")
                shutil.copy2(src, dst)
                print(f"Copied: {dst}")
    onnx_path = os.path.join(out_dir, f"{exp_name}.onnx")
    if os.path.isfile(onnx_path):
        dst = os.path.join("models", f"yolox_{exp_name}.onnx")
        shutil.copy2(onnx_path, dst)
        print(f"Copied: {dst}")


def _ensure_smoke_placeholder(exp) -> None:
    """
    Some of our datasets/annotation JSONs include a `smoke_placeholder.png` image entry.
    If that file is missing, YOLOX will crash immediately in the DataLoader worker,
    making it look like epochs "finish instantly" with best AP = 0.
    """
    data_dir = getattr(exp, "data_dir", None)
    if not data_dir:
        return
    placeholder_rel = os.path.join(data_dir, "train", "images", "smoke_placeholder.png")
    placeholder = os.path.abspath(placeholder_rel)
    if os.path.exists(placeholder):
        return
    os.makedirs(os.path.dirname(placeholder), exist_ok=True)
    # 1x1 transparent PNG
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/w8AAusB9YlWZpQAAAAASUVORK5CYII="
    )
    with open(placeholder, "wb") as f:
        f.write(base64.b64decode(png_b64))
    print(f"Created missing placeholder: {placeholder_rel}")


def main():
    parser = argparse.ArgumentParser(description="Canonical YOLOX training (run from train_yolox/)")
    parser.add_argument("--exp", default="yolox_s_marker", help="Experiment name (e.g. yolox_s_marker)")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (default: from exp)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--setup", action="store_true", help="Only run setup (not used here; use yolox_setup)")
    args = parser.parse_args()

    if not check_layout(args.exp):
        sys.exit(1)

    # Import device selection and training launcher (will be MPS-aware in todo 2)
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run setup first.", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    print(f"Device: {device}")

    # So YOLOX trainer can use MPS/CPU when not on CUDA (avoids torch.cuda.set_device on Mac)
    env = os.environ.copy()
    if device != "cuda":
        env["YOLOX_DEVICE"] = device
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None) if device != "cuda" else None

    # Verify dataset via exp (optional: load exp and call verify_dataset_config)
    sys.path.insert(0, os.path.join(_ROOT, "exps"))
    try:
        exp_module = __import__(args.exp, fromlist=["Exp"])
        exp = exp_module.Exp()
        _ensure_smoke_placeholder(exp)
        if hasattr(exp, "verify_dataset_config"):
            exp.verify_dataset_config()
    except Exception as e:
        print(f"Dataset verification failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.join(_ROOT, "exps") in sys.path:
            sys.path.remove(os.path.join(_ROOT, "exps"))

    # Build training command: run from YOLOX dir, pass absolute path to exp file
    exp_file = os.path.abspath(f"exps/{args.exp}.py")
    cmd_parts = [sys.executable, "tools/train.py", "-f", exp_file, "-d", "1", "-b", str(args.batch)]
    if args.epochs is not None:
        cmd_parts.extend(["max_epoch", str(args.epochs)])
    if device == "cuda":
        cmd_parts.append("--fp16")

    print("Running:", " ".join(cmd_parts))
    cwd = os.getcwd()
    os.chdir("YOLOX")
    try:
        import subprocess
        proc = subprocess.Popen(
            cmd_parts,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        combined = []
        for line in proc.stdout:
            print(line, end="")
            combined.append(line)
        proc.wait()
        rc = proc.returncode
        combined_str = "".join(combined)
    finally:
        os.chdir(cwd)

    if rc != 0:
        sys.exit(rc)

    # Don't report success if log shows an exception (e.g. child exited 0 but loguru caught it)
    error_indicators = (
        "AttributeError",
        "_cuda_setDevice",
        "Exception in training",
        "ModuleNotFoundError",
        "AssertionError",
        "ValueError:",
        "RuntimeError:",
        "Traceback (most recent call last)",
    )
    if any(ind in combined_str for ind in error_indicators):
        print("Training failed (error in output).", file=sys.stderr)
        sys.exit(1)

    copy_artifacts(args.exp)
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
