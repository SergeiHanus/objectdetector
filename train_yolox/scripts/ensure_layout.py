#!/usr/bin/env python3
"""
Verify canonical layout and dataset for YOLOX training.
Run from train_yolox/ (canonical runtime root).
"""

import os
import sys
import argparse
import json


def get_train_yolox_root():
    """Resolve train_yolox root: CWD or parent if we're in scripts/."""
    cwd = os.path.realpath(os.getcwd())
    if os.path.basename(cwd) == "scripts" and os.path.exists(os.path.join(cwd, "..", "exps")):
        return os.path.dirname(cwd)
    if os.path.exists(os.path.join(cwd, "exps")) and os.path.exists(os.path.join(cwd, "YOLOX")):
        return cwd
    # Maybe we're at repo root and train_yolox is a subdir
    train_yolox = os.path.join(cwd, "train_yolox")
    if os.path.isdir(train_yolox) and os.path.exists(os.path.join(train_yolox, "exps")):
        return train_yolox
    return None


def exp_to_dataset_name(exp_name: str) -> str:
    """yolox_s_marker -> marker_dataset."""
    if exp_name.startswith("yolox_s_"):
        return exp_name.replace("yolox_s_", "", 1) + "_dataset"
    return exp_name + "_dataset"


def check_layout(root: str, exp_name: str, verbose: bool = True) -> bool:
    """Check that required dirs and files exist. Returns True if all pass."""
    ok = True

    # Required dirs under root
    for name in ("YOLOX", "exps", "data"):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            if verbose:
                print(f"Missing directory: {path}")
            ok = False
    if not ok:
        return False

    # YOLOX must contain tools/train.py
    train_py = os.path.join(root, "YOLOX", "tools", "train.py")
    if not os.path.isfile(train_py):
        if verbose:
            print(f"Missing: {train_py}")
        ok = False

    # Experiment config
    exp_py = os.path.join(root, "exps", exp_name + ".py")
    if not os.path.isfile(exp_py):
        if verbose:
            print(f"Missing experiment config: {exp_py}")
        ok = False
    else:
        if verbose:
            print(f"Found experiment: {exp_py}")

    # Dataset for this exp
    ds_name = exp_to_dataset_name(exp_name)
    ds_root = os.path.join(root, "YOLOX", "datasets", ds_name)
    for sub in ("train/images", "val/images", "annotations"):
        path = os.path.join(ds_root, sub)
        if not os.path.isdir(path):
            if verbose:
                print(f"Missing dataset path: {path}")
            ok = False

    for ann in ("train_labels.json", "val_labels.json"):
        path = os.path.join(ds_root, "annotations", ann)
        if not os.path.isfile(path):
            if verbose:
                print(f"Missing annotation file: {path}")
            ok = False

    if ok and verbose:
        print("Layout check passed.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify train_yolox layout and dataset")
    parser.add_argument("--exp", default="yolox_s_marker", help="Experiment name (e.g. yolox_s_marker)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    root = get_train_yolox_root()
    if not root:
        if args.json:
            print(json.dumps({"ok": False, "error": "train_yolox root not found"}))
        else:
            print("Error: must run from train_yolox/ or with train_yolox as current directory")
            print("Current directory:", os.getcwd())
        sys.exit(1)

    if not args.json:
        # Ensure we're in root for downstream scripts
        os.chdir(root)

    passed = check_layout(root, args.exp, verbose=not args.json)
    if args.json:
        print(json.dumps({"ok": passed, "root": root, "exp": args.exp}))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
