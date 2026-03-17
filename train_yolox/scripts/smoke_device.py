#!/usr/bin/env python3
"""
Smoke test: verify PyTorch, YOLOX, and device (MPS/CUDA/CPU).
Run from train_yolox/ or repo root. Exits 0 only if all checks pass.
Output is machine-parseable when --json is set.
"""

import os
import sys
import json
import argparse

def _root():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.dirname(script_dir)
    if os.path.basename(root) == "train_yolox" and os.path.exists(os.path.join(root, "exps")):
        return root
    if os.path.exists(os.path.join(root, "train_yolox", "exps")):
        return os.path.join(root, "train_yolox")
    return root

def run_smoke(json_out: bool) -> bool:
    result = {"ok": False, "python": None, "torch": None, "device": None, "yolox": None, "errors": []}

    result["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        import torch
        result["torch"] = torch.__version__
    except ImportError as e:
        result["errors"].append(f"PyTorch import failed: {e}")
        if not json_out:
            print("FAIL: PyTorch not installed", file=sys.stderr)
        if json_out:
            print(json.dumps(result))
        return False

    # Device
    if torch.cuda.is_available():
        result["device"] = "cuda"
        result["device_name"] = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        result["device"] = "mps"
        result["device_name"] = "Apple Silicon"
    else:
        result["device"] = "cpu"
        result["device_name"] = "CPU"

    # Minimal tensor op on device
    try:
        dev = torch.device(result["device"])
        x = torch.zeros(2, 3, device=dev)
        y = x + 1
        result["tensor_ok"] = True
    except Exception as e:
        result["errors"].append(f"Tensor op on {result['device']} failed: {e}")
        result["tensor_ok"] = False
        if not json_out:
            print(f"FAIL: Tensor op on {result['device']}: {e}", file=sys.stderr)
        if json_out:
            print(json.dumps(result))
        return False

    # YOLOX import (optional for smoke; may need to be in YOLOX or have it installed)
    try:
        import yolox
        result["yolox"] = getattr(yolox, "__version__", "unknown")
    except ImportError:
        result["yolox"] = None
        result["errors"].append("YOLOX not installed (pip install -e YOLOX)")

    result["ok"] = result["tensor_ok"]

    if json_out:
        print(json.dumps(result))
        return result["ok"]

    print(f"Python: {result['python']}")
    print(f"PyTorch: {result['torch']}")
    print(f"Device: {result['device']} ({result.get('device_name', '')})")
    print(f"Tensor op: OK")
    if result["yolox"]:
        print(f"YOLOX: {result['yolox']}")
    else:
        print("YOLOX: not installed")
    if result["errors"]:
        for e in result["errors"]:
            print(f"Warning: {e}", file=sys.stderr)
    print("Smoke check passed.")
    return result["ok"]


def main():
    parser = argparse.ArgumentParser(description="Smoke test PyTorch + device + YOLOX")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args()
    ok = run_smoke(json_out=args.json)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
